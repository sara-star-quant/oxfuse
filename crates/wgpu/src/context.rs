//! The adapter, device and queue a backend runs on.

use oxfuse_core::Error;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long to wait for the device before a run or download fails.
pub(crate) const POLL_TIMEOUT: Duration = Duration::from_secs(30);

/// The longest wait a context accepts. `Duration::MAX` is the natural way to say
/// "no timeout", but a deadline is `Instant::now() + timeout`, which panics on
/// overflow, so anything longer is treated as one year.
pub(crate) const MAX_TIMEOUT: Duration = Duration::from_secs(365 * 24 * 60 * 60);

fn clamp_timeout(timeout: Duration) -> Duration {
    timeout.min(MAX_TIMEOUT)
}

/// Which limits the device is created with. `Downlevel` asks for the smallest
/// limits every supported device guarantees, so a test run proves the kernels
/// fit weaker devices than the one in the machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitsProfile {
    Adapter,
    Downlevel,
    /// The downlevel limits with the dispatch limit lowered to the given number
    /// of workgroups per dimension, so tests can force small shapes onto a folded
    /// grid with spare workgroups.
    Capped(u32),
}

pub struct Context {
    /// Set once by the device-lost callback and never cleared: every later call
    /// fails with this message instead of reaching a dead device.
    lost: Arc<Mutex<Option<String>>>,
    timeout: Duration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    errors: Arc<Mutex<Vec<String>>>,
    /// Compiled pipelines by shader source. The source is a function of the
    /// kernel and its configuration only (shapes travel in a uniform buffer),
    /// so executables on this device share a pipeline instead of compiling the
    /// same shader again.
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    /// How many shaders this device has compiled, for tests.
    compiled: AtomicUsize,
}

/// What is asked of the adapter: the faster one when a machine has several.
fn adapter_options() -> wgpu::RequestAdapterOptions<'static, 'static> {
    wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }
}

impl Context {
    /// Fails with `Error::Backend` when no adapter exists. There is no CPU fallback.
    pub fn new(profile: LimitsProfile) -> Result<Self, Error> {
        Self::from_instance(&wgpu::Instance::default(), profile)
    }

    pub fn from_instance(instance: &wgpu::Instance, profile: LimitsProfile) -> Result<Self, Error> {
        let adapter = pollster::block_on(instance.request_adapter(&adapter_options()))
            .map_err(|e| Error::Backend(format!("no adapter found: {e}")))?;
        let limits = match profile {
            LimitsProfile::Adapter => adapter.limits(),
            LimitsProfile::Downlevel => wgpu::Limits::downlevel_defaults(),
            LimitsProfile::Capped(cap) => wgpu::Limits {
                max_compute_workgroups_per_dimension: cap,
                ..wgpu::Limits::downlevel_defaults()
            },
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| Error::Backend(format!("cannot create a device: {e}")))?;
        // wgpu panics on a validation error unless a handler is installed. Collect
        // them (and device loss) so a run returns `Error::Backend` instead.
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            sink.lock().unwrap().push(e.to_string());
        }));
        let lost = Arc::new(Mutex::new(None));
        let state = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            *state.lock().unwrap() = Some(format!("device lost ({reason:?}): {message}"));
        });
        Ok(Context {
            lost,
            timeout: POLL_TIMEOUT,
            device,
            queue,
            info: adapter.get_info(),
            limits,
            errors,
            pipelines: Mutex::new(HashMap::new()),
            compiled: AtomicUsize::new(0),
        })
    }

    /// Replaces the default 30 second wait for the device in `wait_idle` and
    /// downloads. A slow software adapter or a very large matmul can need longer.
    /// A timeout above one year (for example `Duration::MAX`) counts as one year.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = clamp_timeout(timeout);
        self
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Fails with the device-lost message if the device is gone.
    pub fn ensure_alive(&self) -> Result<(), Error> {
        match &*self.lost.lock().unwrap() {
            Some(message) => Err(Error::Backend(message.clone())),
            None => Ok(()),
        }
    }

    /// Runs `f` with this thread's validation and out-of-memory errors captured
    /// by error scopes, so a failure is reported to the call that caused it and
    /// not to whichever call drains the shared error list next. Fails first if
    /// the device is already lost.
    pub(crate) fn scoped<T>(&self, f: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        self.ensure_alive()?;
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let result = f();
        let memory_error = pollster::block_on(memory.pop());
        let validation_error = pollster::block_on(validation.pop());
        let outcome = match result {
            Err(e) => Err(e),
            Ok(value) => match memory_error.or(validation_error) {
                Some(e) => Err(Error::Backend(e.to_string())),
                None => Ok(value),
            },
        };
        match outcome {
            Ok(value) => {
                self.ensure_alive()?;
                Ok(value)
            }
            // A lost device shows up first as some other failure; the loss is
            // reported by the device-lost callback, which runs while polling.
            Err(e) => {
                let _ = self.device.poll(wgpu::PollType::Poll);
                Err(self.ensure_alive().err().unwrap_or(e))
            }
        }
    }

    /// A buffer of `size` bytes filled by `fill` while it is mapped at creation.
    /// Unlike `create_buffer_init` this reports a lost device as an error and
    /// does not copy the data through a temporary vector.
    pub(crate) fn buffer_init(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        fill: impl FnOnce(&mut wgpu::BufferViewMut),
    ) -> Result<wgpu::Buffer, Error> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|e| Error::Backend(format!("filling a buffer failed: {e}")))?;
            fill(&mut view);
        }
        buffer.unmap();
        Ok(buffer)
    }

    /// A buffer holding `data` as little-endian `f32`, written in chunks so no
    /// second copy of the tensor is built.
    pub(crate) fn buffer_from_f32(
        &self,
        data: &[f32],
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, Error> {
        const CHUNK: usize = 16_384;
        self.buffer_init(data.len() as u64 * 4, usage, |view| {
            let mut bytes = [0u8; CHUNK * 4];
            for (index, chunk) in data.chunks(CHUNK).enumerate() {
                for (dst, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
                    *dst = v.to_le_bytes();
                }
                let start = index * CHUNK * 4;
                view.slice(start..start + chunk.len() * 4)
                    .copy_from_slice(&bytes[..chunk.len() * 4]);
            }
        })
    }

    /// Number of shaders compiled on this device so far.
    #[cfg(test)]
    pub(crate) fn compiled_count(&self) -> usize {
        self.compiled.load(Ordering::Relaxed)
    }

    /// The compute pipeline for `source`, compiled on first use.
    pub(crate) fn pipeline(&self, source: String) -> wgpu::ComputePipeline {
        let mut pipelines = self.pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(&source) {
            return pipeline.clone();
        }
        self.compiled.fetch_add(1, Ordering::Relaxed);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        pipelines.insert(source, pipeline.clone());
        pipeline
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Blocks until all submitted work has finished, or fails after the poll timeout.
    pub fn wait_idle(&self) -> Result<(), Error> {
        self.ensure_alive()?;
        let polled = self
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(self.timeout),
            })
            .map(|_| ())
            .map_err(|e| e.to_string());
        self.finish_wait(polled)
    }

    /// What a wait reports once its poll returned. A device loss comes first:
    /// the device-lost callback runs while polling, so it can only be seen now,
    /// and it explains a failed poll better than the poll's own message. Then
    /// the poll's failure, then the errors collected in the meantime.
    fn finish_wait(&self, polled: Result<(), String>) -> Result<(), Error> {
        self.ensure_alive()?;
        polled.map_err(|e| Error::Backend(format!("waiting for the device failed: {e}")))?;
        match self.take_errors() {
            Some(message) => Err(Error::Backend(message)),
            None => Ok(()),
        }
    }

    #[cfg(test)]
    pub(crate) fn record_error(&self, message: &str) {
        self.errors.lock().unwrap().push(message.to_owned());
    }

    /// Drains the device errors reported since the last call.
    pub fn take_errors(&self) -> Option<String> {
        let mut errors = self.errors.lock().unwrap();
        if errors.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut *errors).join("; "))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_timeout_above_a_year_counts_as_a_year() {
        assert_eq!(MAX_TIMEOUT, Duration::from_secs(31_536_000));
        assert_eq!(clamp_timeout(Duration::MAX), MAX_TIMEOUT);
        assert_eq!(
            clamp_timeout(MAX_TIMEOUT + Duration::from_secs(1)),
            MAX_TIMEOUT
        );
        assert_eq!(clamp_timeout(MAX_TIMEOUT), MAX_TIMEOUT);
        assert_eq!(
            clamp_timeout(Duration::from_secs(30)),
            Duration::from_secs(30)
        );
        assert_eq!(clamp_timeout(Duration::ZERO), Duration::ZERO);
        // The clamped value is safe to add to a clock.
        assert!(std::time::Instant::now().checked_add(MAX_TIMEOUT).is_some());
    }

    #[test]
    fn a_wait_reports_a_loss_before_the_poll_failure_before_collected_errors() {
        let Ok(ctx) = Context::new(LimitsProfile::Adapter) else {
            return; // no adapter on this machine
        };
        // Nothing wrong: fine.
        assert!(ctx.finish_wait(Ok(())).is_ok());
        // A failed poll with a healthy device is reported as the poll's failure.
        let e = ctx
            .finish_wait(Err("timed out".into()))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("waiting for the device failed: timed out"),
            "{e}"
        );
        // An error collected while waiting is reported once.
        ctx.record_error("collected");
        assert!(
            ctx.finish_wait(Ok(()))
                .unwrap_err()
                .to_string()
                .contains("collected")
        );
        assert!(ctx.finish_wait(Ok(())).is_ok());
        // A device lost during the poll wins over a successful or failed poll.
        *ctx.lost.lock().unwrap() = Some("device lost (test): gone".into());
        for polled in [Ok(()), Err("poll broke".into())] {
            let e = ctx.finish_wait(polled).unwrap_err().to_string();
            assert!(e.contains("device lost (test): gone"), "{e}");
            assert!(!e.contains("waiting for the device failed"), "{e}");
        }
    }

    #[test]
    fn the_adapter_request_prefers_the_faster_device() {
        assert_eq!(
            super::adapter_options().power_preference,
            wgpu::PowerPreference::HighPerformance
        );
    }

    use super::*;

    #[test]
    fn no_adapter_is_an_error_not_a_fallback() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        match Context::from_instance(&instance, LimitsProfile::Adapter) {
            Err(Error::Backend(message)) => assert!(message.contains("no adapter"), "{message}"),
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("a context without any backend must not exist"),
        }
    }

    #[test]
    fn a_device_validation_error_is_collected_not_panicked() {
        let Ok(context) = Context::new(LimitsProfile::Adapter) else {
            return; // no adapter on this machine
        };
        assert_eq!(context.take_errors(), None);
        let _ = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl("this is not wgsl".into()),
            });
        let message = context
            .take_errors()
            .expect("the invalid shader was reported");
        assert!(!message.is_empty());
        assert_eq!(context.take_errors(), None, "errors are drained once read");
    }

    #[test]
    fn the_device_has_exactly_the_limits_the_profile_asked_for() {
        for profile in [LimitsProfile::Adapter, LimitsProfile::Downlevel] {
            let Ok(context) = Context::new(profile) else {
                return; // no adapter on this machine
            };
            assert_eq!(context.device().limits(), *context.limits(), "{profile:?}");
        }
    }

    #[test]
    fn wait_idle_reports_a_recorded_device_error_once() {
        let Ok(context) = Context::new(LimitsProfile::Adapter) else {
            return; // no adapter on this machine
        };
        assert_eq!(context.wait_idle(), Ok(()));
        context.record_error("injected failure");
        match context.wait_idle() {
            Err(Error::Backend(m)) => assert!(m.contains("injected failure"), "{m}"),
            other => panic!("the recorded error must surface, got {other:?}"),
        }
        assert_eq!(
            context.wait_idle(),
            Ok(()),
            "errors are drained once reported"
        );
    }

    #[test]
    fn the_downlevel_profile_requests_no_more_than_the_downlevel_limits() {
        let Ok(context) = Context::new(LimitsProfile::Downlevel) else {
            return; // no adapter on this machine
        };
        let floor = wgpu::Limits::downlevel_defaults();
        assert_eq!(
            context.limits().max_compute_workgroups_per_dimension,
            floor.max_compute_workgroups_per_dimension
        );
        assert_eq!(
            context.limits().max_storage_buffer_binding_size,
            floor.max_storage_buffer_binding_size
        );
    }
}
