//! Compiling a plan to dispatches and running it on a device.
//!
//! Shapes are static, so `compile` fixes every buffer, pipeline, parameter block
//! and dispatch size. `run_device` only binds, encodes and submits. Intermediate
//! buffers belong to the executable and are reused, so runs are serialized by a
//! mutex; results are copied into fresh buffers before they are returned.

use crate::config::{DeviceCaps, KernelConfig, check_tensor, grid_fold, permute_layout, tile_grid};
use crate::context::{Context, LimitsProfile};
use crate::shaders::{self, Binary, Unary};
use oxfuse_core::{Error, Graph, Op};
use oxfuse_runtime::{Backend, Executable, Plan, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

/// How long to wait for the device before a run or download fails.
const POLL_TIMEOUT: Duration = Duration::from_secs(30);

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend(message.into())
}

pub struct WgpuBackend {
    ctx: Arc<Context>,
    cfg: KernelConfig,
}

impl WgpuBackend {
    /// Fails with `Error::Backend` when there is no adapter.
    pub fn new() -> Result<Self, Error> {
        let ctx = Context::new(LimitsProfile::Adapter)?;
        Ok(Self::with(Arc::new(ctx), KernelConfig::DEFAULT))
    }

    pub fn with(ctx: Arc<Context>, cfg: KernelConfig) -> Self {
        WgpuBackend { ctx, cfg }
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// Like `Backend::compile`, keeping the device API (`upload`, `run_device`, `download`).
    pub fn compile_device(&self, graph: &Graph) -> Result<WgpuExecutable, Error> {
        WgpuExecutable::build(Arc::clone(&self.ctx), self.cfg, Plan::build(graph))
    }
}

impl Backend for WgpuBackend {
    fn compile(&self, graph: &Graph) -> Result<Box<dyn Executable>, Error> {
        Ok(Box::new(self.compile_device(graph)?))
    }
}

/// A tensor in device memory.
#[derive(Debug)]
pub struct DeviceTensor {
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
}

impl DeviceTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kernel {
    Unary(Unary),
    Binary(Binary, bool, bool),
    Transpose2d,
    Permute(usize),
    Matmul,
}

struct Dispatch {
    pipeline: usize,
    params: wgpu::Buffer,
    groups: (u32, u32),
}

/// A kernel chosen for one step, before its pipeline and parameter buffer exist.
struct Planned {
    kernel: Kernel,
    source: String,
    words: Vec<u32>,
    groups: (u32, u32),
}

struct CompiledStep {
    inputs: Vec<usize>,
    output: usize,
    /// `None`: the output shares its input's buffer (Reshape, rank-0 Transpose).
    dispatch: Option<Dispatch>,
}

pub struct WgpuExecutable {
    ctx: Arc<Context>,
    plan: Plan,
    pipelines: Vec<wgpu::ComputePipeline>,
    steps: Vec<CompiledStep>,
    constants: Vec<(usize, wgpu::Buffer)>,
    /// One buffer per computed slot; the mutex serializes runs that reuse them.
    scratch: Mutex<Vec<Option<wgpu::Buffer>>>,
}

fn caps_of(limits: &wgpu::Limits) -> DeviceCaps {
    DeviceCaps {
        max_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
        max_invocations: limits.max_compute_invocations_per_workgroup,
        max_wg_storage_bytes: limits.max_compute_workgroup_storage_size,
        max_binding_bytes: limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size),
    }
}

/// Which operand of an Add or Mul is a rank-0 scalar applied to the other. When
/// both are scalars neither is broadcast: the single element is read directly.
fn scalar_flags(lhs: &[usize], rhs: &[usize]) -> (bool, bool) {
    let both_scalar = lhs.is_empty() && rhs.is_empty();
    (
        lhs.is_empty() && !both_scalar,
        rhs.is_empty() && !both_scalar,
    )
}

fn elements(shape: &[usize]) -> u64 {
    shape.iter().map(|&d| d as u64).product()
}

fn bytes_of_words(words: &[u32]) -> Vec<u8> {
    let mut bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.resize(bytes.len().next_multiple_of(16).max(16), 0);
    bytes
}

impl WgpuExecutable {
    fn build(ctx: Arc<Context>, cfg: KernelConfig, plan: Plan) -> Result<Self, Error> {
        let caps = caps_of(ctx.limits());
        cfg.validate(&caps).map_err(backend_error)?;
        for (slot, shape) in plan.slot_shapes.iter().enumerate() {
            check_tensor(elements(shape), &caps)
                .map_err(|m| backend_error(format!("value {slot} {shape:?}: {m}")))?;
        }
        let device = ctx.device();
        let mut pipelines: Vec<wgpu::ComputePipeline> = Vec::new();
        let mut by_kernel: HashMap<Kernel, usize> = HashMap::new();
        let mut steps = Vec::new();
        let mut scratch: Vec<Option<wgpu::Buffer>> = vec![None; plan.slot_shapes.len()];
        for step in &plan.steps {
            let out_shape = &plan.slot_shapes[step.output];
            let out_elements = elements(out_shape);
            let shape_of = |i: usize| plan.slot_shapes[step.inputs[i]].as_slice();
            // (kernel, source, parameter words, dispatch groups)
            let planned: Option<Planned> = match &step.op {
                Op::Reshape { .. } => None,
                Op::Transpose { perm } if perm.is_empty() => None,
                Op::Relu | Op::Sigmoid => {
                    let kind = if step.op == Op::Relu {
                        Unary::Relu
                    } else {
                        Unary::Sigmoid
                    };
                    let groups = fold(out_elements, &cfg, &caps)?;
                    Some(Planned {
                        kernel: Kernel::Unary(kind),
                        source: shaders::unary(kind, &cfg),
                        words: vec![out_elements as u32, groups.0],
                        groups,
                    })
                }
                Op::Add | Op::Mul => {
                    let kind = if step.op == Op::Add {
                        Binary::Add
                    } else {
                        Binary::Mul
                    };
                    let (lhs, rhs) = scalar_flags(shape_of(0), shape_of(1));
                    let groups = fold(out_elements, &cfg, &caps)?;
                    Some(Planned {
                        kernel: Kernel::Binary(kind, lhs, rhs),
                        source: shaders::binary(kind, lhs, rhs, &cfg),
                        words: vec![out_elements as u32, groups.0],
                        groups,
                    })
                }
                Op::Transpose { perm } if perm.len() == 2 && perm == &[1, 0] => {
                    let (rows, cols) = (shape_of(0)[0] as u64, shape_of(0)[1] as u64);
                    let grid =
                        tile_grid(rows, cols, cfg.transpose_tile, caps.max_workgroups_per_dim)
                            .ok_or_else(|| {
                                backend_error(format!("{rows}x{cols} cannot be dispatched"))
                            })?;
                    Some(Planned {
                        kernel: Kernel::Transpose2d,
                        source: shaders::transpose2d(&cfg),
                        words: vec![
                            rows as u32,
                            cols as u32,
                            grid.tiles_x,
                            grid.x_count,
                            grid.total_groups() as u32,
                        ],
                        groups: (grid.x_count, grid.y_count),
                    })
                }
                Op::Transpose { perm } => {
                    let rank = perm.len();
                    let (dims, strides) = permute_layout(shape_of(0), perm);
                    let groups = fold(out_elements, &cfg, &caps)?;
                    let mut words = vec![out_elements as u32, groups.0, 0, 0];
                    words.extend(dims.iter().chain(&[1; 8]).take(8));
                    words.extend(strides.iter().chain(&[0; 8]).take(8));
                    Some(Planned {
                        kernel: Kernel::Permute(rank),
                        source: shaders::permute(rank, &cfg),
                        words,
                        groups,
                    })
                }
                Op::MatMul => {
                    let (m, k) = (shape_of(0)[0] as u64, shape_of(0)[1] as u64);
                    let n = shape_of(1)[1] as u64;
                    let grid = tile_grid(m, n, cfg.matmul_block(), caps.max_workgroups_per_dim)
                        .ok_or_else(|| backend_error(format!("{m}x{n} cannot be dispatched")))?;
                    Some(Planned {
                        kernel: Kernel::Matmul,
                        source: shaders::matmul(&cfg),
                        words: vec![
                            m as u32,
                            k as u32,
                            n as u32,
                            grid.tiles_x,
                            grid.x_count,
                            grid.total_groups() as u32,
                        ],
                        groups: (grid.x_count, grid.y_count),
                    })
                }
            };
            let dispatch = match planned {
                None => None,
                Some(Planned {
                    kernel,
                    source,
                    words,
                    groups,
                }) => {
                    let pipeline = match by_kernel.get(&kernel) {
                        Some(&index) => index,
                        None => {
                            let module =
                                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                    label: None,
                                    source: wgpu::ShaderSource::Wgsl(source.into()),
                                });
                            pipelines.push(device.create_compute_pipeline(
                                &wgpu::ComputePipelineDescriptor {
                                    label: None,
                                    layout: None,
                                    module: &module,
                                    entry_point: Some("main"),
                                    compilation_options: Default::default(),
                                    cache: None,
                                },
                            ));
                            by_kernel.insert(kernel, pipelines.len() - 1);
                            pipelines.len() - 1
                        }
                    };
                    scratch[step.output] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: out_elements * 4,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }));
                    Some(Dispatch {
                        pipeline,
                        params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: &bytes_of_words(&words),
                            usage: wgpu::BufferUsages::UNIFORM,
                        }),
                        groups,
                    })
                }
            };
            steps.push(CompiledStep {
                inputs: step.inputs.clone(),
                output: step.output,
                dispatch,
            });
        }
        let constants = plan
            .constants
            .iter()
            .map(|(slot, data)| {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
                (*slot, buffer)
            })
            .collect();
        let executable = WgpuExecutable {
            ctx,
            plan,
            pipelines,
            steps,
            constants,
            scratch: Mutex::new(scratch),
        };
        executable.check_device()?;
        Ok(executable)
    }

    fn check_device(&self) -> Result<(), Error> {
        match self.ctx.take_errors() {
            Some(message) => Err(backend_error(message)),
            None => Ok(()),
        }
    }

    pub fn upload(&self, tensor: &Tensor) -> Result<DeviceTensor, Error> {
        check_tensor(elements(tensor.shape()), &caps_of(self.ctx.limits()))
            .map_err(backend_error)?;
        let bytes: Vec<u8> = tensor.data().iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.check_device()?;
        Ok(DeviceTensor {
            buffer,
            shape: tensor.shape().to_vec(),
        })
    }

    pub fn download(&self, tensor: &DeviceTensor) -> Result<Tensor, Error> {
        let device = self.ctx.device();
        let size = elements(&tensor.shape) * 4;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, size);
        let submission = self.ctx.queue().submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        // Wait for this submission, not the newest one: another thread may have
        // submitted since, and our map callback only fires once we have polled
        // past our own work. Poll again until it has fired.
        let deadline = Instant::now() + POLL_TIMEOUT;
        let mapped = loop {
            if let Ok(result) = receiver.try_recv() {
                break result;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(backend_error("timed out waiting for the device"));
            }
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission.clone()),
                    timeout: Some(remaining),
                })
                .map_err(|e| backend_error(format!("waiting for the device failed: {e}")))?;
            std::thread::yield_now();
        };
        self.check_device()?;
        mapped.map_err(|e| backend_error(format!("mapping the result failed: {e}")))?;
        let view = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| backend_error(format!("reading the result failed: {e}")))?;
        let data: Vec<f32> = view
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        drop(view);
        staging.unmap();
        Tensor::new(tensor.shape.clone(), data)
    }

    pub fn run_device(&self, inputs: &[DeviceTensor]) -> Result<Vec<DeviceTensor>, Error> {
        let plan = &self.plan;
        if inputs.len() != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found: inputs.len(),
            });
        }
        for (binding, tensor) in plan.inputs.iter().zip(inputs) {
            if tensor.shape != binding.shape {
                return Err(Error::InputShape {
                    name: binding.name.clone(),
                    expected: binding.shape.clone(),
                    found: tensor.shape.clone(),
                });
            }
        }
        let scratch = self.scratch.lock().unwrap();
        let mut slots: Vec<Option<wgpu::Buffer>> = scratch.clone();
        for (binding, tensor) in plan.inputs.iter().zip(inputs) {
            slots[binding.slot] = Some(tensor.buffer.clone());
        }
        for (slot, buffer) in &self.constants {
            slots[*slot] = Some(buffer.clone());
        }
        let device = self.ctx.device();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for step in &self.steps {
                let Some(dispatch) = &step.dispatch else {
                    slots[step.output] = slots[step.inputs[0]].clone();
                    continue;
                };
                let pipeline = &self.pipelines[dispatch.pipeline];
                let mut entries = Vec::new();
                let buffers: Vec<&wgpu::Buffer> = step
                    .inputs
                    .iter()
                    .chain([&step.output])
                    .map(|&s| slots[s].as_ref().expect("slot is bound"))
                    .collect();
                for (binding, buffer) in buffers.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    });
                }
                entries.push(wgpu::BindGroupEntry {
                    binding: buffers.len() as u32,
                    resource: dispatch.params.as_entire_binding(),
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(dispatch.groups.0, dispatch.groups.1, 1);
            }
        }
        let mut results = Vec::with_capacity(plan.outputs.len());
        for &slot in &plan.outputs {
            let shape = plan.slot_shapes[slot].clone();
            let size = elements(&shape) * 4;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                slots[slot].as_ref().expect("output is bound"),
                0,
                &buffer,
                0,
                size,
            );
            results.push(DeviceTensor { buffer, shape });
        }
        self.ctx.queue().submit([encoder.finish()]);
        drop(scratch);
        self.check_device()?;
        Ok(results)
    }
}

fn fold(elements: u64, cfg: &KernelConfig, caps: &DeviceCaps) -> Result<(u32, u32), Error> {
    grid_fold(elements, cfg.elementwise_wg, caps.max_workgroups_per_dim)
        .ok_or_else(|| backend_error(format!("{elements} elements cannot be dispatched")))
}

impl Executable for WgpuExecutable {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let plan = &self.plan;
        if inputs.len() != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found: inputs.len(),
            });
        }
        let device_inputs = inputs
            .iter()
            .map(|t| self.upload(t))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = self.run_device(&device_inputs)?;
        outputs.iter().map(|t| self.download(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_flags_mark_only_the_broadcast_operand() {
        assert_eq!(scalar_flags(&[], &[3]), (true, false));
        assert_eq!(scalar_flags(&[3], &[]), (false, true));
        assert_eq!(scalar_flags(&[], &[]), (false, false));
        assert_eq!(scalar_flags(&[3], &[3]), (false, false));
    }

    #[test]
    fn a_device_error_recorded_since_the_last_call_fails_the_next_one() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let mut b = oxfuse_core::GraphBuilder::new(oxfuse_core::Limits::default());
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let exe = backend.compile_device(&b.finish(&[y]).unwrap()).unwrap();
        let tensor = Tensor::new(vec![2], vec![1.0, -1.0]).unwrap();
        backend.context().record_error("injected failure");
        match exe.upload(&tensor) {
            Err(Error::Backend(m)) => assert!(m.contains("injected failure"), "{m}"),
            other => panic!("the recorded error must surface, got {other:?}"),
        }
        let device = exe
            .upload(&tensor)
            .expect("errors are drained once reported");
        backend.context().record_error("second failure");
        assert!(
            matches!(exe.run_device(&[device]), Err(Error::Backend(m)) if m.contains("second failure"))
        );
    }
}
