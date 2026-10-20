//! The adapter, device and queue a backend runs on.

use oxfuse_core::Error;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long to wait for the device before a run or download fails.
pub(crate) const POLL_TIMEOUT: Duration = Duration::from_secs(30);

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
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    errors: Arc<Mutex<Vec<String>>>,
}

impl Context {
    /// Fails with `Error::Backend` when no adapter exists. There is no CPU fallback.
    pub fn new(profile: LimitsProfile) -> Result<Self, Error> {
        Self::from_instance(&wgpu::Instance::default(), profile)
    }

    pub fn from_instance(instance: &wgpu::Instance, profile: LimitsProfile) -> Result<Self, Error> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
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
        let sink = Arc::clone(&errors);
        device.set_device_lost_callback(move |reason, message| {
            sink.lock()
                .unwrap()
                .push(format!("device lost ({reason:?}): {message}"));
        });
        Ok(Context {
            device,
            queue,
            info: adapter.get_info(),
            limits,
            errors,
        })
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
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(POLL_TIMEOUT),
            })
            .map_err(|e| Error::Backend(format!("waiting for the device failed: {e}")))?;
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
