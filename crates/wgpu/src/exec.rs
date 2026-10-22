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
use std::time::Instant;

/// Highest rank the permute kernel walks (its dimensions fit two `vec4<u32>`).
const MAX_PERMUTE_RANK: usize = 8;

fn backend_error(message: impl Into<String>) -> Error {
    Error::Backend(message.into())
}

/// The kernel configuration an adapter gets by default. Software adapters get a
/// 1 by 1 register block: on the Windows software adapter the 4 by 4 kernel took
/// more than 30 seconds to run its first dispatch, while 1 by 1 compiles in about
/// half a second and runs a 128 by 128 matmul in about a second. Register blocking
/// gains nothing on a CPU-backed device anyway.
fn default_config(device_type: wgpu::DeviceType) -> KernelConfig {
    match device_type {
        wgpu::DeviceType::Cpu => KernelConfig {
            reg_block: 1,
            ..KernelConfig::DEFAULT
        },
        _ => KernelConfig::DEFAULT,
    }
}

pub struct WgpuBackend {
    ctx: Arc<Context>,
    cfg: KernelConfig,
}

impl WgpuBackend {
    /// Fails with `Error::Backend` when there is no adapter.
    pub fn new() -> Result<Self, Error> {
        let ctx = Context::new(LimitsProfile::Adapter)?;
        Ok(Self::for_context(Arc::new(ctx)))
    }

    /// A backend on `ctx` with the default configuration for that adapter.
    pub fn for_context(ctx: Arc<Context>) -> Self {
        let cfg = default_config(ctx.info().device_type);
        Self::with(ctx, cfg)
    }

    pub fn with(ctx: Arc<Context>, cfg: KernelConfig) -> Self {
        WgpuBackend { ctx, cfg }
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    pub fn config(&self) -> KernelConfig {
        self.cfg
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
    /// One buffer per computed slot that is not a direct output; the mutex
    /// serializes runs that reuse them.
    scratch: Mutex<Vec<Option<wgpu::Buffer>>>,
    /// Slots a kernel writes straight into the buffer that is handed back: the
    /// computed outputs. Their buffer is created per run, so a result never
    /// shares memory with a later run. Results are read-only, so two results
    /// (an output listed twice, or one a Reshape aliases) may share a buffer.
    direct: Vec<bool>,
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
        let scope = Arc::clone(&ctx);
        scope.scoped(|| Self::build_unscoped(ctx, cfg, plan))
    }

    fn build_unscoped(ctx: Arc<Context>, cfg: KernelConfig, plan: Plan) -> Result<Self, Error> {
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
        let mut direct = vec![false; plan.slot_shapes.len()];
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
                Op::Transpose { perm } if perm.len() > MAX_PERMUTE_RANK => {
                    return Err(backend_error(format!(
                        "a transpose of rank {} is not supported, the kernels handle rank {MAX_PERMUTE_RANK} at most",
                        perm.len()
                    )));
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
                            pipelines.push(ctx.pipeline(source));
                            by_kernel.insert(kernel, pipelines.len() - 1);
                            pipelines.len() - 1
                        }
                    };
                    if plan.outputs.contains(&step.output) {
                        direct[step.output] = true;
                    } else {
                        scratch[step.output] =
                            Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: None,
                                size: out_elements * 4,
                                usage:
                                    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC),
                                mapped_at_creation: false,
                            }));
                    }
                    Some(Dispatch {
                        pipeline,
                        params: {
                            let bytes = bytes_of_words(&words);
                            ctx.buffer_init(
                                bytes.len() as u64,
                                wgpu::BufferUsages::UNIFORM,
                                |view| view.copy_from_slice(&bytes),
                            )?
                        },
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
                let usage = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
                Ok((*slot, ctx.buffer_from_f32(data, usage)?))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let executable = WgpuExecutable {
            ctx,
            plan,
            pipelines,
            steps,
            constants,
            scratch: Mutex::new(scratch),
            direct,
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
        self.ctx.scoped(|| {
            check_tensor(elements(tensor.shape()), &caps_of(self.ctx.limits()))
                .map_err(backend_error)?;
            let usage = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
            let buffer = self.ctx.buffer_from_f32(tensor.data(), usage)?;
            self.check_device()?;
            Ok(DeviceTensor {
                buffer,
                shape: tensor.shape().to_vec(),
            })
        })
    }

    pub fn download(&self, tensor: &DeviceTensor) -> Result<Tensor, Error> {
        self.ctx.scoped(|| self.download_unscoped(tensor))
    }

    fn download_unscoped(&self, tensor: &DeviceTensor) -> Result<Tensor, Error> {
        let device = self.ctx.device();
        let size = elements(&tensor.shape) * 4;
        let staging = self.staging_buffer(size);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, size);
        let submission = self.ctx.queue().submit([encoder.finish()]);
        let mut tensors = self.read_staged(submission, vec![(staging, tensor.shape.clone())])?;
        Ok(tensors.remove(0))
    }

    fn staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST),
            mapped_at_creation: false,
        })
    }

    /// Waits for `submission`, then reads every staging buffer back as a tensor
    /// of the paired shape. All buffers are mapped before the single wait.
    fn read_staged(
        &self,
        submission: wgpu::SubmissionIndex,
        staged: Vec<(wgpu::Buffer, Vec<usize>)>,
    ) -> Result<Vec<Tensor>, Error> {
        let device = self.ctx.device();
        let (sender, receiver) = std::sync::mpsc::channel();
        for (index, (staging, _)) in staged.iter().enumerate() {
            let sender = sender.clone();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send((index, result));
                });
        }
        drop(sender);
        // Wait for this submission, not the newest one: another thread may have
        // submitted since, and our map callbacks only fire once we have polled
        // past our own work. Poll again until every one has fired.
        let deadline = Instant::now() + self.ctx.timeout();
        let mut mapped: Vec<Option<Result<(), wgpu::BufferAsyncError>>> =
            (0..staged.len()).map(|_| None).collect();
        let mut pending = staged.len();
        loop {
            while let Ok((index, result)) = receiver.try_recv() {
                mapped[index] = Some(result);
                pending -= 1;
            }
            if pending == 0 {
                break;
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
        }
        self.check_device()?;
        staged
            .into_iter()
            .zip(mapped)
            .map(|((staging, shape), result)| {
                result
                    .expect("every map callback fired")
                    .map_err(|e| backend_error(format!("mapping the result failed: {e}")))?;
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
                Tensor::new(shape, data)
            })
            .collect()
    }

    pub fn run_device(&self, inputs: &[DeviceTensor]) -> Result<Vec<DeviceTensor>, Error> {
        self.ctx.scoped(|| self.run_device_unscoped(inputs))
    }

    fn check_inputs(&self, shapes: &[&[usize]]) -> Result<(), Error> {
        let plan = &self.plan;
        if shapes.len() != plan.inputs.len() {
            return Err(Error::InputCount {
                expected: plan.inputs.len(),
                found: shapes.len(),
            });
        }
        for (binding, shape) in plan.inputs.iter().zip(shapes) {
            if *shape != binding.shape.as_slice() {
                return Err(Error::InputShape {
                    name: binding.name.clone(),
                    expected: binding.shape.clone(),
                    found: shape.to_vec(),
                });
            }
        }
        Ok(())
    }

    /// The buffer of every slot for one run: scratch, the given inputs,
    /// constants, and a fresh buffer for each output a kernel writes directly.
    fn bind_slots(
        &self,
        scratch: &[Option<wgpu::Buffer>],
        inputs: &[&wgpu::Buffer],
    ) -> Vec<Option<wgpu::Buffer>> {
        let plan = &self.plan;
        let mut slots: Vec<Option<wgpu::Buffer>> = scratch.to_vec();
        for (binding, buffer) in plan.inputs.iter().zip(inputs) {
            slots[binding.slot] = Some((*buffer).clone());
        }
        for (slot, buffer) in &self.constants {
            slots[*slot] = Some(buffer.clone());
        }
        for &slot in &plan.outputs {
            if self.direct[slot] {
                slots[slot] = Some(
                    self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: elements(&plan.slot_shapes[slot]) * 4,
                        usage: wgpu::BufferUsages::STORAGE
                            .union(wgpu::BufferUsages::COPY_SRC)
                            .union(wgpu::BufferUsages::COPY_DST),
                        mapped_at_creation: false,
                    }),
                );
            }
        }
        slots
    }

    /// Records one compute pass with a dispatch per step.
    fn encode_pass(&self, encoder: &mut wgpu::CommandEncoder, slots: &mut [Option<wgpu::Buffer>]) {
        let device = self.ctx.device();
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

    fn run_device_unscoped(&self, inputs: &[DeviceTensor]) -> Result<Vec<DeviceTensor>, Error> {
        let plan = &self.plan;
        let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape.as_slice()).collect();
        self.check_inputs(&shapes)?;
        let scratch = self.scratch.lock().unwrap();
        let buffers: Vec<&wgpu::Buffer> = inputs.iter().map(|t| &t.buffer).collect();
        let mut slots = self.bind_slots(&scratch, &buffers);
        let device = self.ctx.device();
        let mut encoder = device.create_command_encoder(&Default::default());
        self.encode_pass(&mut encoder, &mut slots);
        let mut results = Vec::with_capacity(plan.outputs.len());
        for &slot in &plan.outputs {
            let shape = plan.slot_shapes[slot].clone();
            if self.direct[slot] {
                let buffer = slots[slot].clone().expect("output is bound");
                results.push(DeviceTensor { buffer, shape });
                continue;
            }
            let size = elements(&shape) * 4;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    .union(wgpu::BufferUsages::COPY_SRC)
                    .union(wgpu::BufferUsages::COPY_DST),
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

    /// `run_many` in one submission: every run's kernels and output copies share
    /// one command buffer and one wait. The runs share the scratch buffers; the
    /// passes execute in order, so each sees its own intermediates.
    fn run_many_unscoped(&self, batches: &[Vec<Tensor>]) -> Result<Vec<Vec<Tensor>>, Error> {
        let plan = &self.plan;
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        for inputs in batches {
            let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
            self.check_inputs(&shapes)?;
        }
        let caps = caps_of(self.ctx.limits());
        let usage = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
        let mut uploads = Vec::with_capacity(batches.len());
        for inputs in batches {
            let mut uploaded = Vec::with_capacity(inputs.len());
            for tensor in inputs {
                check_tensor(elements(tensor.shape()), &caps).map_err(backend_error)?;
                uploaded.push(self.ctx.buffer_from_f32(tensor.data(), usage)?);
            }
            uploads.push(uploaded);
        }
        let scratch = self.scratch.lock().unwrap();
        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        // Per run: which staged tensor holds each output slot.
        let mut staging_of: Vec<HashMap<usize, usize>> = Vec::with_capacity(batches.len());
        let mut staged = Vec::new();
        for uploaded in &uploads {
            let buffers: Vec<&wgpu::Buffer> = uploaded.iter().collect();
            let mut slots = self.bind_slots(&scratch, &buffers);
            self.encode_pass(&mut encoder, &mut slots);
            staging_of.push(self.stage_outputs(&mut encoder, &slots, &mut staged));
        }
        let submission = self.ctx.queue().submit([encoder.finish()]);
        drop(scratch);
        let tensors = self.read_staged(submission, staged)?;
        Ok(staging_of
            .iter()
            .map(|of| {
                plan.outputs
                    .iter()
                    .map(|slot| tensors[of[slot]].clone())
                    .collect()
            })
            .collect())
    }

    /// Copies each distinct output slot into a new staging buffer appended to
    /// `staged`; returns the position of each slot's staging buffer.
    fn stage_outputs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slots: &[Option<wgpu::Buffer>],
        staged: &mut Vec<(wgpu::Buffer, Vec<usize>)>,
    ) -> HashMap<usize, usize> {
        let plan = &self.plan;
        let mut staging_of: HashMap<usize, usize> = HashMap::new();
        for &slot in &plan.outputs {
            if staging_of.contains_key(&slot) {
                continue;
            }
            let size = elements(&plan.slot_shapes[slot]) * 4;
            let staging = self.staging_buffer(size);
            encoder.copy_buffer_to_buffer(
                slots[slot].as_ref().expect("output is bound"),
                0,
                &staging,
                0,
                size,
            );
            staging_of.insert(slot, staged.len());
            staged.push((staging, plan.slot_shapes[slot].clone()));
        }
        staging_of
    }

    /// `run` in one submission: the kernels and the copies of the outputs into
    /// staging buffers share a command buffer, and one wait covers every output.
    fn run_to_host_unscoped(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let plan = &self.plan;
        let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape()).collect();
        self.check_inputs(&shapes)?;
        let caps = caps_of(self.ctx.limits());
        let usage = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC);
        let mut uploaded = Vec::with_capacity(inputs.len());
        for tensor in inputs {
            check_tensor(elements(tensor.shape()), &caps).map_err(backend_error)?;
            uploaded.push(self.ctx.buffer_from_f32(tensor.data(), usage)?);
        }
        let scratch = self.scratch.lock().unwrap();
        let buffers: Vec<&wgpu::Buffer> = uploaded.iter().collect();
        let mut slots = self.bind_slots(&scratch, &buffers);
        let mut encoder = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        self.encode_pass(&mut encoder, &mut slots);
        // One staging buffer per distinct output slot; a slot listed twice is read once.
        let mut staged = Vec::new();
        let staging_of = self.stage_outputs(&mut encoder, &slots, &mut staged);
        let submission = self.ctx.queue().submit([encoder.finish()]);
        drop(scratch);
        let tensors = self.read_staged(submission, staged)?;
        Ok(plan
            .outputs
            .iter()
            .map(|slot| tensors[staging_of[slot]].clone())
            .collect())
    }
}

fn fold(elements: u64, cfg: &KernelConfig, caps: &DeviceCaps) -> Result<(u32, u32), Error> {
    grid_fold(elements, cfg.elementwise_wg, caps.max_workgroups_per_dim)
        .ok_or_else(|| backend_error(format!("{elements} elements cannot be dispatched")))
}

impl Executable for WgpuExecutable {
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        self.ctx.scoped(|| self.run_to_host_unscoped(inputs))
    }

    fn run_many(&self, batches: &[Vec<Tensor>]) -> Result<Vec<Vec<Tensor>>, Error> {
        self.ctx.scoped(|| self.run_many_unscoped(batches))
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
    fn executables_on_one_device_share_compiled_pipelines() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let relu = |shape: Vec<usize>| {
            let mut b = oxfuse_core::GraphBuilder::new(oxfuse_core::Limits::default());
            let x = b.input("x", shape).unwrap();
            let y = b.node("y", Op::Relu, &[x]).unwrap();
            b.finish(&[y]).unwrap()
        };
        let first = backend.compile_device(&relu(vec![2])).unwrap();
        assert_eq!(backend.context().compiled_count(), 1);
        // Another shape is the same shader: nothing new is compiled.
        let second = backend.compile_device(&relu(vec![5, 3])).unwrap();
        assert_eq!(backend.context().compiled_count(), 1);
        // And the shared pipeline still computes each shape correctly.
        let run = |exe: &WgpuExecutable, shape: Vec<usize>, data: Vec<f32>| {
            let input = exe.upload(&Tensor::new(shape, data).unwrap()).unwrap();
            let out = exe.run_device(&[input]).unwrap();
            exe.download(&out[0]).unwrap().data().to_vec()
        };
        assert_eq!(run(&first, vec![2], vec![-1.0, 2.0]), [0.0, 2.0]);
        let data: Vec<f32> = (0..15).map(|i| i as f32 - 7.0).collect();
        let want: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
        assert_eq!(run(&second, vec![5, 3], data), want);
        // A different kernel is a new pipeline.
        let mut b = oxfuse_core::GraphBuilder::new(oxfuse_core::Limits::default());
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Sigmoid, &[x]).unwrap();
        backend.compile_device(&b.finish(&[y]).unwrap()).unwrap();
        assert_eq!(backend.context().compiled_count(), 2);
    }

    /// Outputs the kernel writes in place must still be separate results: one
    /// that a later step reads, one listed twice, one that a Reshape aliases,
    /// and results that survive the next run.
    #[test]
    fn outputs_written_in_place_stay_distinct_and_correct() {
        use oxfuse_core::{GraphBuilder, Limits};
        use oxfuse_runtime::{Backend, ReferenceBackend};
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        type Build = fn(&mut GraphBuilder) -> Vec<oxfuse_core::ValueId>;
        let cases: [(&str, Build); 4] = [
            ("read later", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let c = b.constant("c", vec![6], vec![1.0; 6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                let z = b.node("z", Op::Add, &[y, c]).unwrap();
                vec![y, z]
            }),
            ("listed twice", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                vec![y, y]
            }),
            ("reshape alias", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                let r = b
                    .node("r", Op::Reshape { shape: vec![2, 3] }, &[y])
                    .unwrap();
                vec![y, r]
            }),
            ("single output", |b| {
                let x = b.input("x", vec![6]).unwrap();
                vec![b.node("y", Op::Relu, &[x]).unwrap()]
            }),
        ];
        for (name, build) in cases {
            let mut b = GraphBuilder::new(Limits::default());
            let outputs = build(&mut b);
            let graph = b.finish(&outputs).unwrap();
            let gpu = backend.compile_device(&graph).unwrap();
            let reference = ReferenceBackend.compile(&graph).unwrap();
            let first_in = Tensor::new(vec![6], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
            let second_in = Tensor::new(vec![6], vec![7.0, -8.0, 9.0, -1.0, 2.0, -3.0]).unwrap();
            let first = gpu.run_device(&[gpu.upload(&first_in).unwrap()]).unwrap();
            // A second run must not touch the first run's results.
            gpu.run_device(&[gpu.upload(&second_in).unwrap()]).unwrap();
            let want = reference.run(&[first_in]).unwrap();
            assert_eq!(first.len(), want.len(), "{name}");
            for (got, want) in first.iter().zip(&want) {
                let got = gpu.download(got).unwrap();
                assert_eq!(got.shape(), want.shape(), "{name}");
                assert_eq!(got.data(), want.data(), "{name}");
            }
        }
    }

    fn relu_graph(len: usize) -> Graph {
        let mut b = oxfuse_core::GraphBuilder::new(oxfuse_core::Limits::default());
        let x = b.input("x", vec![len]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        b.finish(&[y]).unwrap()
    }

    fn message(result: Result<impl std::fmt::Debug, Error>) -> String {
        match result {
            Err(Error::Backend(m)) => m,
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    /// `run` records everything in one submission; its results must equal the
    /// device path (upload, `run_device`, download) bit for bit on graphs that
    /// share outputs, alias buffers or return inputs and constants.
    #[test]
    fn run_in_one_submission_matches_the_device_path() {
        use oxfuse_core::{GraphBuilder, Limits, ValueId};
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        type Build = fn(&mut GraphBuilder) -> Vec<ValueId>;
        let cases: [(&str, Build); 7] = [
            ("two outputs", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let c = b.constant("c", vec![6], vec![0.5; 6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                vec![y, b.node("z", Op::Add, &[y, c]).unwrap()]
            }),
            ("listed twice", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                vec![y, y]
            }),
            ("input and constant as outputs", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let c = b.constant("c", vec![6], vec![1.5; 6]).unwrap();
                let y = b.node("y", Op::Mul, &[x, c]).unwrap();
                vec![x, c, y]
            }),
            ("reshape alias", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let y = b.node("y", Op::Relu, &[x]).unwrap();
                let r = b
                    .node("r", Op::Reshape { shape: vec![2, 3] }, &[y])
                    .unwrap();
                vec![r, y]
            }),
            ("scalar broadcast", |b| {
                let x = b.input("x", vec![6]).unwrap();
                let s = b.constant("s", vec![], vec![2.0]).unwrap();
                vec![b.node("y", Op::Mul, &[s, x]).unwrap()]
            }),
            ("matmul and transpose", |b| {
                let x = b.input("x", vec![2, 3]).unwrap();
                let w = b
                    .constant("w", vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap();
                let m = b.node("m", Op::MatMul, &[x, w]).unwrap();
                vec![
                    b.node("t", Op::Transpose { perm: vec![1, 0] }, &[m])
                        .unwrap(),
                ]
            }),
            ("one element", |b| {
                let x = b.input("x", vec![1]).unwrap();
                vec![b.node("y", Op::Sigmoid, &[x]).unwrap()]
            }),
        ];
        for (name, build) in cases {
            let mut b = GraphBuilder::new(Limits::default());
            let outputs = build(&mut b);
            let graph = b.finish(&outputs).unwrap();
            let exe = backend.compile_device(&graph).unwrap();
            let shape = match name {
                "matmul and transpose" => vec![2, 3],
                "one element" => vec![1],
                _ => vec![6],
            };
            let len: usize = shape.iter().product();
            for seed in 0..3 {
                let data: Vec<f32> = (0..len).map(|i| ((i + seed * 2) as f32) - 2.5).collect();
                let input = Tensor::new(shape.clone(), data).unwrap();
                let fused = exe.run(std::slice::from_ref(&input)).unwrap();
                let uploaded = exe.upload(&input).unwrap();
                let device = exe.run_device(std::slice::from_ref(&uploaded)).unwrap();
                assert_eq!(fused.len(), device.len(), "{name}");
                for (got, want) in fused.iter().zip(&device) {
                    let want = exe.download(want).unwrap();
                    assert_eq!(got.shape(), want.shape(), "{name}");
                    let bits =
                        |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(got), bits(&want), "{name} seed {seed}");
                }
            }
        }
    }

    /// Runs of one batch share the scratch buffers inside one command buffer;
    /// each must still see only its own intermediates.
    #[test]
    fn run_many_equals_separate_runs_on_a_graph_with_intermediates() {
        use oxfuse_core::{GraphBuilder, Limits};
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![8]).unwrap();
        let c = b.constant("c", vec![8], vec![0.5; 8]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let z = b.node("z", Op::Add, &[y, c]).unwrap();
        let w = b.node("w", Op::Mul, &[z, z]).unwrap();
        let graph = b.finish(&[w, y, w]).unwrap();
        let exe = backend.compile_device(&graph).unwrap();
        let batches: Vec<Vec<Tensor>> = (0..5)
            .map(|k| {
                let data: Vec<f32> = (0..8)
                    .map(|i| (i as f32 - 3.0) * (k as f32 + 1.0))
                    .collect();
                vec![Tensor::new(vec![8], data).unwrap()]
            })
            .collect();
        let many = exe.run_many(&batches).unwrap();
        assert_eq!(many.len(), batches.len());
        let bits = |t: &Tensor| t.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (batch, got) in batches.iter().zip(&many) {
            let want = exe.run(batch).unwrap();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.shape(), w.shape());
                assert_eq!(bits(g), bits(w));
            }
        }
        // An empty batch does nothing, a batch of one equals run.
        assert!(exe.run_many(&[]).unwrap().is_empty());
        let one = exe.run_many(&batches[..1]).unwrap();
        assert_eq!(bits(&one[0][0]), bits(&exe.run(&batches[0]).unwrap()[0]));
    }

    #[test]
    fn run_many_validates_every_batch_before_running_any() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let exe = backend.compile_device(&relu_graph(4)).unwrap();
        let good = vec![Tensor::new(vec![4], vec![1.0; 4]).unwrap()];
        let wrong = vec![Tensor::new(vec![5], vec![1.0; 5]).unwrap()];
        let missing: Vec<Tensor> = Vec::new();
        assert!(matches!(
            exe.run_many(&[good.clone(), wrong]),
            Err(Error::InputShape { .. })
        ));
        assert!(matches!(
            exe.run_many(&[good.clone(), missing]),
            Err(Error::InputCount {
                expected: 1,
                found: 0
            })
        ));
        assert_eq!(exe.run_many(&[good]).unwrap().len(), 1);
    }

    #[test]
    fn run_rejects_wrong_inputs_before_touching_the_device() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let exe = backend.compile_device(&relu_graph(4)).unwrap();
        let good = Tensor::new(vec![4], vec![1.0; 4]).unwrap();
        let wrong = Tensor::new(vec![5], vec![1.0; 5]).unwrap();
        assert!(matches!(
            exe.run(&[]),
            Err(Error::InputCount {
                expected: 1,
                found: 0
            })
        ));
        assert!(matches!(
            exe.run(&[good.clone(), good.clone()]),
            Err(Error::InputCount {
                expected: 1,
                found: 2
            })
        ));
        assert!(matches!(exe.run(&[wrong]), Err(Error::InputShape { .. })));
        // And it still works afterwards.
        assert_eq!(exe.run(&[good]).unwrap()[0].data(), &[1.0; 4]);
    }

    #[test]
    fn two_threads_sharing_one_executable_get_their_own_results() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let exe = backend.compile_device(&relu_graph(64)).unwrap();
        std::thread::scope(|scope| {
            for thread in 0..2u32 {
                let exe = &exe;
                scope.spawn(move || {
                    for round in 0..20u32 {
                        let sign = if thread == 0 { 1.0 } else { -1.0 };
                        let data: Vec<f32> = (0..64).map(|i| sign * (i + round) as f32).collect();
                        let want: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
                        let input = Tensor::new(vec![64], data).unwrap();
                        let got = exe.run(&[input]).unwrap();
                        assert_eq!(got[0].data(), want, "thread {thread} round {round}");
                    }
                });
            }
        });
    }

    /// Upload writes 4 bytes per element in chunks; lengths around the chunk
    /// size and odd lengths must round trip with the exact buffer size.
    #[test]
    fn upload_sizes_the_buffer_per_element_and_keeps_every_value() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        for len in [1usize, 2, 3, 5, 16383, 16384, 16385, 40001] {
            let exe = backend.compile_device(&relu_graph(len)).unwrap();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 + 1.0).collect();
            let tensor = Tensor::new(vec![len], data.clone()).unwrap();
            let uploaded = exe.upload(&tensor).unwrap();
            assert_eq!(uploaded.buffer.size(), len as u64 * 4, "len {len}");
            assert_eq!(exe.download(&uploaded).unwrap().data(), data, "len {len}");
        }
    }

    #[test]
    fn a_lost_device_fails_every_entry_with_the_same_error_and_never_panics() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let exe = backend.compile_device(&relu_graph(4)).unwrap();
        let tensor = Tensor::new(vec![4], vec![1.0, -1.0, 2.0, -2.0]).unwrap();
        let input = exe.upload(&tensor).unwrap();
        let output = exe.run_device(std::slice::from_ref(&input)).unwrap();
        backend.context().device().destroy();
        // The device-lost callback runs while the device is polled.
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        while backend.context().ensure_alive().is_ok() {
            assert!(Instant::now() < deadline, "the loss was never reported");
            let _ = backend.context().device().poll(wgpu::PollType::Poll);
            std::thread::yield_now();
        }
        let first = message(exe.upload(&tensor));
        assert!(first.contains("device lost"), "{first}");
        assert_eq!(message(exe.run_device(std::slice::from_ref(&input))), first);
        assert_eq!(message(exe.download(&output[0])), first);
        assert_eq!(message(exe.run(std::slice::from_ref(&tensor))), first);
        assert_eq!(
            message(backend.compile_device(&relu_graph(4)).map(|_| ())),
            first
        );
        assert_eq!(message(backend.context().wait_idle().map(|()| 0)), first);
        // Sticky: asking again gives the same answer, not a fresh panic.
        assert_eq!(message(exe.upload(&tensor)), first);
    }

    #[test]
    fn a_failure_in_one_thread_is_not_reported_to_another() {
        use std::sync::Barrier;
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        let ctx = Arc::clone(backend.context());
        let exe = backend.compile_device(&relu_graph(4)).unwrap();
        let tensor = Tensor::new(vec![4], vec![1.0, -1.0, 2.0, -2.0]).unwrap();
        let barrier = Barrier::new(2);
        std::thread::scope(|scope| {
            let failing = scope.spawn(|| {
                ctx.scoped(|| {
                    barrier.wait();
                    // A buffer with no usage is a validation error.
                    let _invalid = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: 16,
                        usage: wgpu::BufferUsages::empty(),
                        mapped_at_creation: false,
                    });
                    barrier.wait();
                    Ok(())
                })
            });
            barrier.wait();
            barrier.wait();
            // The other thread's error has happened; this call must not see it.
            let upload = exe.upload(&tensor);
            assert!(upload.is_ok(), "{upload:?}");
            let error = failing
                .join()
                .unwrap()
                .expect_err("the failing call reports it");
            assert!(matches!(error, Error::Backend(_)), "{error:?}");
        });
    }

    #[test]
    fn a_zero_timeout_fails_a_download_and_the_default_does_not() {
        let tensor = Tensor::new(vec![4], vec![1.0, -1.0, 2.0, -2.0]).unwrap();
        for (timeout, ok) in [(None, true), (Some(std::time::Duration::ZERO), false)] {
            let Ok(mut ctx) = Context::new(LimitsProfile::Adapter) else {
                return; // no adapter on this machine
            };
            if let Some(timeout) = timeout {
                ctx = ctx.with_timeout(timeout);
            }
            let backend = WgpuBackend::for_context(Arc::new(ctx));
            let exe = backend.compile_device(&relu_graph(4)).unwrap();
            let input = exe.upload(&tensor).unwrap();
            let output = exe.run_device(std::slice::from_ref(&input)).unwrap();
            let downloaded = exe.download(&output[0]);
            if ok {
                assert_eq!(downloaded.unwrap().data(), &[1.0, 0.0, 2.0, 0.0]);
            } else {
                assert!(message(downloaded).contains("timed out"));
            }
        }
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

    #[test]
    fn software_adapters_get_the_small_matmul_block() {
        let default_block = KernelConfig::DEFAULT.reg_block;
        assert_eq!(default_config(wgpu::DeviceType::Cpu).reg_block, 1);
        for hardware in [
            wgpu::DeviceType::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::VirtualGpu,
            wgpu::DeviceType::Other,
        ] {
            assert_eq!(
                default_config(hardware),
                KernelConfig::DEFAULT,
                "{hardware:?}"
            );
        }
        assert_eq!(default_block, 4);
    }

    #[test]
    fn every_default_config_fits_the_smallest_guaranteed_limits() {
        for device_type in [wgpu::DeviceType::Cpu, wgpu::DeviceType::DiscreteGpu] {
            assert_eq!(
                default_config(device_type).validate(&DeviceCaps::DOWNLEVEL),
                Ok(())
            );
        }
    }

    #[test]
    fn a_backend_picks_its_config_from_the_adapter() {
        let Ok(backend) = WgpuBackend::new() else {
            return; // no adapter on this machine
        };
        assert_eq!(
            backend.config(),
            default_config(backend.context().info().device_type)
        );
    }
}
