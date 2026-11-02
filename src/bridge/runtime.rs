//! A worker owns the provider and every native handle; providers need not implement Send.
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use virtio_accel_core::{
    Accelerator, AccessMode, BackendError, BindingRef, BufferDesc, BufferRange, BufferUsage,
    ContextDesc, EventState, MemoryDomain, QueueDesc, ReleaseFailure, SubmitFailure, Timeout,
    TransportByteSink, TransportByteSource,
};
use virtio_accel_tosa::{CapabilityDescriptor, DType, TosaCapabilityProvider, ValueRoles, parse};

#[derive(Clone)]
pub enum Provider {
    #[cfg(feature = "vulkan")]
    Vulkan(String),
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    CoreMl,
}

pub struct Discovered {
    pub provider: Provider,
    pub name: &'static str,
    pub description: String,
    pub capability: CapabilityDescriptor,
}

fn capability<A: TosaCapabilityProvider>(a: &A) -> Option<CapabilityDescriptor> {
    a.tosa_capabilities()
        .iter()
        .copied()
        .find(|c| c.supports_dtype(DType::FP32, ValueRoles::ALL))
}

pub fn discover() -> Vec<Discovered> {
    let mut found = Vec::new();
    #[cfg(feature = "vulkan")]
    {
        let selected = std::env::var("GGML_NPU_VULKAN_DEVICE").ok();
        let result = match selected {
            Some(ref name) => virtio_accel_vulkan::VulkanAccelerator::with_device(name),
            None => virtio_accel_vulkan::VulkanAccelerator::new(),
        };
        match result {
            Ok(a) => {
                if let Some(capability) = capability(&a) {
                    found.push(Discovered {
                        provider: Provider::Vulkan(a.device_name().to_owned()),
                        name: "VIRTIO_VULKAN",
                        description: format!("virtio-accel Vulkan: {}", a.device_name()),
                        capability,
                    });
                }
            }
            Err(e) => eprintln!("ggml-npu: Vulkan unavailable: {e}"),
        }
    }
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    match virtio_accel_coreml::CoreMlAccelerator::new_tosa() {
        Ok(a) => {
            if let Some(capability) = capability(&a) {
                found.push(Discovered {
                    provider: Provider::CoreMl,
                    name: "VIRTIO_COREML",
                    description: "virtio-accel Core ML (CPU and Neural Engine)".into(),
                    capability,
                });
            }
        }
        Err(e) => eprintln!("ggml-npu: Core ML unavailable: {e}"),
    }
    found
}

pub struct Job {
    pub artifact: Vec<u8>,
    pub inputs: Vec<Vec<u8>>,
    pub output_sizes: Vec<usize>,
}
type ResultBytes = Result<Vec<Vec<u8>>, String>;
struct Command {
    job: Job,
    reply: Sender<ResultBytes>,
}
pub struct Worker {
    sender: Option<Sender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub fn new(provider: Provider, cap: CapabilityDescriptor) -> Result<Self, String> {
        let (sender, receiver) = mpsc::channel::<Command>();
        let (ready, initialized) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("ggml-virtio-accel".into())
            .spawn(move || {
                fn run<A: Accelerator>(
                    a: Result<A, String>,
                    cap: CapabilityDescriptor,
                    ready: Sender<Result<(), String>>,
                    receiver: mpsc::Receiver<Command>,
                ) {
                    let mut session = match a.and_then(Session::new) {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = ready.send(Err(e));
                            return;
                        }
                    };
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    for command in receiver {
                        let result = session.execute(command.job, cap);
                        let _ = command.reply.send(result);
                    }
                }
                match provider {
                    #[cfg(feature = "vulkan")]
                    Provider::Vulkan(name) => run(
                        virtio_accel_vulkan::VulkanAccelerator::with_device(&name)
                            .map_err(|e| e.to_string()),
                        cap,
                        ready,
                        receiver,
                    ),
                    #[cfg(all(feature = "coreml", target_os = "macos"))]
                    Provider::CoreMl => run(
                        virtio_accel_coreml::CoreMlAccelerator::new_tosa()
                            .map_err(|e| e.to_string()),
                        cap,
                        ready,
                        receiver,
                    ),
                }
            })
            .map_err(|e| e.to_string())?;
        match initialized.recv().map_err(|e| e.to_string())? {
            Ok(()) => Ok(Self {
                sender: Some(sender),
                thread: Some(thread),
            }),
            Err(e) => {
                let _ = thread.join();
                Err(e)
            }
        }
    }
    pub fn execute(&self, job: Job) -> ResultBytes {
        let (reply, result) = mpsc::channel();
        self.sender
            .as_ref()
            .ok_or("worker closed")?
            .send(Command { job, reply })
            .map_err(|e| e.to_string())?;
        result.recv().map_err(|e| e.to_string())?
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Cached<A: Accelerator> {
    artifact: Vec<u8>,
    program: A::Program,
    buffers: Vec<A::Buffer>,
    sizes: Vec<usize>,
    input_count: usize,
}
struct Session<A: Accelerator> {
    info: virtio_accel_core::DeviceInfo,
    accelerator: Option<A>,
    context: Option<A::Context>,
    queue: Option<A::Queue>,
    cached: Option<Cached<A>>,
    pending: Option<A::Event>,
    poisoned: bool,
}
fn error(e: impl std::fmt::Debug) -> String {
    format!("virtio-accel: {e:?}")
}

// Failed releases retain ownership. Never let a generic handle's destructor guess its status.
fn release<T>(r: Result<(), ReleaseFailure<T>>) -> bool {
    match r {
        Ok(()) => true,
        Err(ReleaseFailure::Rejected { resource, .. }) => {
            std::mem::forget(resource);
            false
        }
        Err(ReleaseFailure::Indeterminate { .. }) => false,
    }
}
impl<A: Accelerator> Session<A> {
    fn new(accelerator: A) -> Result<Self, String> {
        let info = accelerator.device_info().map_err(error)?;
        info.validate().map_err(error)?;
        let mut s = Self {
            info,
            accelerator: Some(accelerator),
            context: None,
            queue: None,
            cached: None,
            pending: None,
            poisoned: false,
        };
        let a = s.accelerator.as_ref().ok_or("missing provider")?;
        s.context = Some(a.create_context(ContextDesc::default()).map_err(error)?);
        s.queue = Some(
            a.create_queue(
                s.context.as_ref().ok_or("missing context")?,
                QueueDesc::default(),
            )
            .map_err(error)?,
        );
        Ok(s)
    }
    fn clear_cache(&mut self) -> Result<(), String> {
        if let Some(c) = self.cached.take() {
            let a = self.accelerator.as_ref().ok_or("missing provider")?;
            let mut ok = release(a.unload_program(c.program));
            for b in c.buffers {
                ok &= release(a.free_buffer(b));
            }
            if !ok {
                self.poisoned = true;
                return Err("provider failed to release cached resources".into());
            }
        }
        Ok(())
    }
    fn prepare(&mut self, job: &Job, cap: CapabilityDescriptor) -> Result<(), String> {
        let sizes: Vec<_> = job
            .inputs
            .iter()
            .map(Vec::len)
            .chain(job.output_sizes.iter().copied())
            .collect();
        if self.cached.as_ref().is_some_and(|c| {
            c.artifact == job.artifact && c.sizes == sizes && c.input_count == job.inputs.len()
        }) {
            return Ok(());
        }
        self.clear_cache()?;
        let a = self.accelerator.as_ref().ok_or("missing provider")?;
        let context = self.context.as_ref().ok_or("missing context")?;
        if sizes.len() > self.info.limits.max_buffers_per_context as usize
            || sizes.len() > self.info.limits.max_bindings_per_submission as usize
            || job.artifact.len() as u64 > self.info.limits.max_artifact_bytes
        {
            return Err("program exceeds provider limits".into());
        }
        let model = parse(&job.artifact).map_err(error)?;
        let artifact = model.artifact_ref(cap.target, u64::MAX).map_err(error)?;
        let program = a.load_program(context, artifact).map_err(error)?;
        self.cached = Some(Cached {
            artifact: job.artifact.clone(),
            program,
            buffers: Vec::new(),
            sizes: sizes.clone(),
            input_count: job.inputs.len(),
        });
        for (i, size) in sizes.into_iter().enumerate() {
            let usage = if i < job.inputs.len() {
                BufferUsage::TRANSFER_DESTINATION | BufferUsage::PROGRAM_INPUT
            } else {
                BufferUsage::TRANSFER_SOURCE | BufferUsage::PROGRAM_OUTPUT
            };
            let desc =
                BufferDesc::new(size as u64, 64, MemoryDomain::Host, usage).map_err(error)?;
            self.info.validate_buffer_desc(desc).map_err(error)?;
            let (buffer, actual) = a
                .allocate_buffer(context, desc)
                .map_err(error)?
                .into_parts();
            self.cached
                .as_mut()
                .ok_or("missing cache")?
                .buffers
                .push(buffer);
            self.info
                .validate_buffer_info(desc, actual)
                .map_err(error)?;
        }
        Ok(())
    }
    fn execute(&mut self, job: Job, cap: CapabilityDescriptor) -> ResultBytes {
        if self.poisoned || self.pending.is_some() {
            return Err("provider requires recovery; recreate backend".into());
        }
        if let Err(e) = self.prepare(&job, cap) {
            let _ = self.clear_cache();
            return Err(e);
        }
        let a = self.accelerator.as_ref().ok_or("missing provider")?;
        let cache = self.cached.as_mut().ok_or("missing cache")?;
        for (b, bytes) in cache.buffers.iter_mut().zip(&job.inputs) {
            a.write_buffer(b, 0, &TransportByteSource::new(bytes.as_slice()))
                .map_err(error)?;
        }
        let bindings: Vec<_> = cache
            .buffers
            .iter()
            .zip(&cache.sizes)
            .enumerate()
            .map(|(i, (b, &size))| {
                Ok(BindingRef {
                    slot: i as u32,
                    buffer: b,
                    range: BufferRange::new(0, size as u64)?,
                    access: if i < cache.input_count {
                        AccessMode::Read
                    } else {
                        AccessMode::Write
                    },
                })
            })
            .collect::<Result<_, BackendError>>()
            .map_err(error)?;
        let mut admission_error = None;
        self.pending = Some(
            match a.submit(
                self.queue.as_ref().ok_or("missing queue")?,
                &cache.program,
                &bindings,
                Timeout::Infinite,
            ) {
                Ok(event) => event,
                Err(SubmitFailure::Rejected(e)) => return Err(error(e)),
                Err(SubmitFailure::Indeterminate { error: e, event }) => {
                    admission_error = Some(error(e));
                    event
                }
            },
        );
        // An indeterminate submission is drained, never retried or allowed to release live bindings.
        let result = loop {
            match a
                .poll_event(self.pending.as_ref().ok_or("missing event")?)
                .map_err(error)?
            {
                EventState::Pending => thread::sleep(std::time::Duration::from_micros(50)),
                EventState::Complete => break admission_error.map_or(Ok(()), Err),
                EventState::Failed(e) => break Err(error(e)),
                EventState::Cancelled => break Err("submission cancelled".into()),
            }
        };
        let event = self.pending.take().ok_or("missing event")?;
        if !release(a.destroy_event(event)) {
            self.poisoned = true;
            return Err("event release failed".into());
        }
        result?;
        let mut outputs = Vec::new();
        for (b, &size) in cache
            .buffers
            .iter()
            .zip(&cache.sizes)
            .skip(cache.input_count)
        {
            let mut bytes = vec![0; size];
            a.read_buffer(b, 0, &mut TransportByteSink::new(bytes.as_mut_slice()))
                .map_err(error)?;
            outputs.push(bytes);
        }
        Ok(outputs)
    }
}
impl<A: Accelerator> Drop for Session<A> {
    fn drop(&mut self) {
        // A failed poll leaves device ownership unresolved. Retain all backing until process exit.
        if self.pending.is_some() || self.poisoned {
            std::mem::forget(self.pending.take());
            std::mem::forget(self.cached.take());
            std::mem::forget(self.queue.take());
            std::mem::forget(self.context.take());
            std::mem::forget(self.accelerator.take());
            return;
        }
        let _ = self.clear_cache();
        if let Some(a) = self.accelerator.as_ref() {
            if let Some(q) = self.queue.take() {
                self.poisoned |= !release(a.destroy_queue(q));
            }
            if !self.poisoned
                && let Some(c) = self.context.take()
            {
                self.poisoned |= !release(a.destroy_context(c));
            }
        }
        if self.poisoned {
            std::mem::forget(self.context.take());
            std::mem::forget(self.accelerator.take());
        }
    }
}

#[cfg(test)]
mod tests;
