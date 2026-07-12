//! Default direct-rustix IVPU execution session with byte-budgeted resident programs.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};

use super::direct::budget::RuntimeBudget;
use super::direct::elf::parse_intel_elf_with_resident_limit;
use super::direct::hpi::inspect_hpi_requirements;
use super::direct::link::{visit_binding_requirements, DeviceMemoryClass};
use super::direct::prepare::{prepare_inference, PreparedInference};
use super::direct::transport::release_host_writes;
use super::direct::{CommandQueue, IvpuDevice};
use super::profile;
use crate::identity::{ExactKeyBuilder, ExactSemanticKey};
use crate::metrics::MetricEvent;
use crate::ops::Operand;
use crate::os::linux::LinuxOs;
use crate::os::OsServices;
use crate::runtime::{Deadline, RuntimeError};

struct DirectContext {
    device: &'static IvpuDevice,
    queue: CommandQueue<'static>,
}

static CONTEXT: AtomicPtr<DirectContext> = AtomicPtr::new(ptr::null_mut());

fn context() -> Result<&'static DirectContext, RuntimeError> {
    let current = CONTEXT.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: published contexts are leaked and immutable after initialization.
        return Ok(unsafe { &*current });
    }
    let device = Box::leak(Box::new(IvpuDevice::open_first()?));
    let queue = device.create_queue(0, 0)?;
    let candidate = Box::into_raw(Box::new(DirectContext { device, queue }));
    match CONTEXT.compare_exchange(
        ptr::null_mut(),
        candidate,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {
            // SAFETY: the winning allocation is process-lifetime.
            Ok(unsafe { &*candidate })
        }
        Err(winner) => {
            // SAFETY: this candidate was never published; dropping destroys its queue.
            drop(unsafe { Box::from_raw(candidate) });
            // SAFETY: the winner is a published process-lifetime context.
            Ok(unsafe { &*winner })
        }
    }
}

struct ResidentProgram {
    generation: u64,
    key: ExactSemanticKey,
    prepared: PreparedInference<'static>,
    resident_bytes: u64,
    accounted_bytes: u64,
    last_used: u64,
    pins: u32,
}

struct CacheState {
    entries: Vec<ResidentProgram>,
    resident_bytes: u64,
    resident_high_water: u64,
    accounted_high_water: u64,
    clock: u64,
    generation: u64,
}

impl CacheState {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            resident_bytes: 0,
            resident_high_water: 0,
            accounted_high_water: 0,
            clock: 0,
            generation: 0,
        }
    }

    fn evict_one(&mut self) -> bool {
        let candidate = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.pins == 0)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(index, _)| index);
        let Some(index) = candidate else {
            return false;
        };
        let entry = self.entries.swap_remove(index);
        self.resident_bytes = self.resident_bytes.saturating_sub(entry.resident_bytes);
        profile::note(&alloc::format!(
            "[npu direct] evict resident={} accounted={} remaining={}\n",
            entry.resident_bytes,
            entry.accounted_bytes,
            self.resident_bytes
        ));
        profile::event(MetricEvent::CacheEviction {
            cache: "direct-programs",
            bytes: entry.resident_bytes,
        });
        drop(entry);
        true
    }

    fn make_room(&mut self, bytes: u64, limit: u64) -> bool {
        while self
            .resident_bytes
            .checked_add(bytes)
            .is_none_or(|total| total > limit)
        {
            if !self.evict_one() {
                return false;
            }
        }
        true
    }
}

struct ResidentCache {
    locked: AtomicBool,
    state: UnsafeCell<CacheState>,
}

impl ResidentCache {
    const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
            state: UnsafeCell::new(CacheState::new()),
        }
    }

    fn lock(&self) -> CacheGuard<'_> {
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        CacheGuard { cache: self }
    }
}

// SAFETY: all access to the UnsafeCell is serialized by the atomic lock.
unsafe impl Sync for ResidentCache {}

struct CacheGuard<'a> {
    cache: &'a ResidentCache,
}

impl CacheGuard<'_> {
    fn state(&mut self) -> &mut CacheState {
        // SAFETY: this guard uniquely owns the cache lock.
        unsafe { &mut *self.cache.state.get() }
    }
}

impl Drop for CacheGuard<'_> {
    fn drop(&mut self) {
        self.cache.locked.store(false, Ordering::Release);
    }
}

static PROGRAMS: ResidentCache = ResidentCache::new();
static DEVICE_FAILED: AtomicBool = AtomicBool::new(false);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static RESIDENT_HIGH_WATER: AtomicU64 = AtomicU64::new(0);
static ACCOUNTED_HIGH_WATER: AtomicU64 = AtomicU64::new(0);

pub(super) fn async_enabled() -> bool {
    false
}

pub(super) fn synchronize() -> bool {
    true
}

pub(super) fn runtime_available() -> bool {
    let Ok(device) = IvpuDevice::open_first() else {
        return false;
    };
    let Ok(parameters) = device.parameters() else {
        return false;
    };
    super::direct::validate::validate_device(&parameters).is_ok()
        && RuntimeBudget::query(&parameters).is_ok()
}

fn operand_storage_bytes(operand: &Operand) -> Option<usize> {
    // SAFETY: dtype comes from a live ggml tensor and the returned type traits are static.
    let traits = unsafe { crate::ffi::ggml_get_type_traits(operand.dtype) };
    if traits.is_null() {
        return None;
    }
    // SAFETY: checked non-null static type-traits pointer.
    let traits = unsafe { &*traits };
    let block = usize::try_from(traits.blck_size).ok()?;
    let type_size = traits.type_size;
    if block == 0 || operand.elems % block != 0 {
        return None;
    }
    operand.elems.checked_div(block)?.checked_mul(type_size)
}

fn operand_bytes<'a>(operand: &'a Operand, bytes: usize) -> Option<&'a [u8]> {
    if operand.ptr.is_null() || operand_storage_bytes(operand)? < bytes {
        return None;
    }
    // SAFETY: storage size was checked and the operand remains live through dispatch.
    Some(unsafe { core::slice::from_raw_parts(operand.ptr.cast::<u8>(), bytes) })
}

fn copy_output(program: &PreparedInference<'_>, index: usize, output: &Operand) -> bool {
    let Some(bytes) = program.output_bytes(index) else {
        profile::note(&alloc::format!(
            "[npu direct] output {index} missing from artifact bindings\n"
        ));
        return false;
    };
    let available = operand_storage_bytes(output);
    if output.ptr.is_null() || available.is_none_or(|size| size < bytes) {
        profile::note(&alloc::format!(
            "[npu direct] output {index} storage mismatch: required={bytes} available={available:?} dtype={} elems={}\n",
            output.dtype, output.elems
        ));
        return false;
    }
    // SAFETY: output storage is live and at least `bytes` long.
    let target = unsafe { core::slice::from_raw_parts_mut(output.ptr.cast::<u8>(), bytes) };
    match program.copy_output(index, target) {
        Ok(()) => true,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] output {index} readback failed: {error:?}\n"
            ));
            false
        }
    }
}

fn dispatch(
    context: &DirectContext,
    program: &PreparedInference<'_>,
    inputs: &[&Operand],
    outputs: &[&Operand],
) -> bool {
    let mut input_bytes = Vec::new();
    if input_bytes.try_reserve_exact(inputs.len()).is_err() {
        return false;
    }
    for (index, input) in inputs.iter().enumerate() {
        let Some(bytes) = program.input_bytes(index) else {
            profile::note(&alloc::format!(
                "[npu direct] dispatch input index {index} missing from artifact\n"
            ));
            return false;
        };
        let Some(data) = operand_bytes(input, bytes) else {
            profile::note(&alloc::format!(
                "[npu direct] dispatch input {index} storage mismatch: required={bytes} available={:?} dtype={} elems={}\n",
                operand_storage_bytes(input),
                input.dtype,
                input.elems
            ));
            return false;
        };
        input_bytes.push(data);
    }
    if outputs.len()
        != program
            .bindings
            .iter()
            .filter(|binding| binding.class == super::direct::link::DeviceMemoryClass::Output)
            .count()
    {
        profile::note(&alloc::format!(
            "[npu direct] dispatch output count mismatch: runtime={} artifact={}\n",
            outputs.len(),
            program
                .bindings
                .iter()
                .filter(|binding| binding.class == DeviceMemoryClass::Output)
                .count()
        ));
        return false;
    }

    let prof = profile::enabled();
    let t0 = if prof { profile::now_ns() } else { 0 };
    if let Err(error) = program.write_inputs(&input_bytes) {
        profile::note(&alloc::format!(
            "[npu direct] input staging failed: {error:?}\n"
        ));
        return false;
    }
    if let Err(error) = program.zero_outputs() {
        profile::note(&alloc::format!(
            "[npu direct] output initialization failed: {error:?}\n"
        ));
        return false;
    }
    release_host_writes();
    let t1 = if prof { profile::now_ns() } else { 0 };
    let now = match LinuxOs.monotonic_now_ns() {
        Ok(now) => now,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] monotonic clock failed: {error:?}\n"
            ));
            return false;
        }
    };
    if context
        .queue
        .submit(&program.handles, program.commands_offset(), 0)
        .is_err()
    {
        DEVICE_FAILED.store(true, Ordering::Release);
        profile::note("[npu direct] submit failed\n");
        return false;
    }
    let tmid = if prof { profile::now_ns() } else { 0 };
    let status = program
        .command
        .wait_idle(Deadline::MonotonicNs(now.saturating_add(2_000_000_000)));
    if !matches!(status, Ok(0)) {
        DEVICE_FAILED.store(true, Ordering::Release);
        profile::note(&alloc::format!("[npu direct] wait failed: {status:?}\n"));
        return false;
    }
    let t2 = if prof { profile::now_ns() } else { 0 };
    for (index, output) in outputs.iter().enumerate() {
        if !copy_output(program, index, output) {
            return false;
        }
    }
    let t3 = if prof { profile::now_ns() } else { 0 };
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    if prof {
        let bytes_in = input_bytes.iter().map(|bytes| bytes.len() as u64).sum();
        let bytes_out = outputs
            .iter()
            .filter_map(|output| operand_storage_bytes(output))
            .map(|bytes| bytes as u64)
            .sum();
        profile::record(
            t0,
            t1,
            tmid,
            t2,
            t3,
            0,
            (inputs.len() + outputs.len()) as u64,
            bytes_in,
            bytes_out,
        );
    }
    true
}

fn run_program(
    generation: u64,
    key: ExactSemanticKey,
    inputs: &[&Operand],
    outputs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    profile::note(&alloc::format!(
        "[npu direct] run generation={generation} inputs={} outputs={} key={}\n",
        inputs.len(),
        outputs.len(),
        key.digest().hex()
    ));
    if DEVICE_FAILED.load(Ordering::Acquire) {
        profile::note("[npu direct] device fault latched; refusing automatic retry\n");
        return false;
    }
    let context = match context() {
        Ok(context) => context,
        Err(error) => {
            profile::note(&alloc::format!("[npu direct] context failed: {error:?}\n"));
            return false;
        }
    };
    let parameters = match context.device.parameters() {
        Ok(parameters) => parameters,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] parameter query failed: {error:?}\n"
            ));
            return false;
        }
    };
    let budget = match RuntimeBudget::query(&parameters) {
        Ok(budget) => budget,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] budget rejected load: {error:?}\n"
            ));
            return false;
        }
    };
    let mut guard = PROGRAMS.lock();
    let state = guard.state();
    if state.generation != generation {
        let evicted = state.resident_bytes;
        state.entries.clear();
        state.resident_bytes = 0;
        state.generation = generation;
        if evicted != 0 {
            profile::event(MetricEvent::CacheEviction {
                cache: "direct-programs",
                bytes: evicted,
            });
        }
    }
    state.clock = state.clock.wrapping_add(1);
    let clock = state.clock;
    let index = state
        .entries
        .iter()
        .position(|entry| entry.generation == generation && entry.key == key);
    let index = if let Some(index) = index {
        profile::note("[npu direct] resident program hit\n");
        index
    } else {
        let Some(blob) = compile() else {
            profile::note("[npu direct] compiler returned no artifact\n");
            return false;
        };
        profile::note(&alloc::format!(
            "[npu direct] artifact bytes={} compiled/cache-loaded\n",
            blob.len()
        ));
        let elf = match parse_intel_elf_with_resident_limit(&blob, budget.resident_program_bytes) {
            Ok(elf) => elf,
            Err(error) => {
                profile::note(&alloc::format!(
                    "[npu direct] unsupported artifact before load: {error:?}\n"
                ));
                return false;
            }
        };
        if !state.make_room(
            elf.inspection().resident_bytes,
            budget.resident_program_bytes,
        ) {
            return false;
        }
        let hpi = match inspect_hpi_requirements(&elf, &parameters) {
            Ok(hpi) => hpi,
            Err(error) => {
                profile::note(&alloc::format!(
                    "[npu direct] unsupported HPI metadata: {error:?}\n"
                ));
                return false;
            }
        };
        let mut initial_inputs = Vec::new();
        if initial_inputs.try_reserve_exact(inputs.len()).is_err() {
            return false;
        }
        let mut input_requirements = Vec::new();
        visit_binding_requirements(&elf, |binding| {
            if binding.class == DeviceMemoryClass::Input {
                input_requirements.push(binding);
            }
        });
        if input_requirements.len() != inputs.len() {
            profile::note(&alloc::format!(
                "[npu direct] load input count mismatch: runtime={} artifact={}\n",
                inputs.len(),
                input_requirements.len()
            ));
            return false;
        }
        for (index, input) in inputs.iter().enumerate() {
            let expected = input_requirements
                .get(index)
                .and_then(|binding| usize::try_from(binding.bytes).ok());
            let Some(data) = expected.and_then(|bytes| operand_bytes(input, bytes)) else {
                profile::note(&alloc::format!(
                    "[npu direct] load input {index} storage mismatch: required={expected:?} available={:?} dtype={} elems={}\n",
                    operand_storage_bytes(input),
                    input.dtype,
                    input.elems
                ));
                return false;
            };
            initial_inputs.push(data);
        }
        let prepared = match prepare_inference(
            context.device,
            &parameters,
            &elf,
            &hpi,
            &initial_inputs,
            budget.preparation(),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                profile::note(&alloc::format!(
                    "[npu direct] artifact load failed: {error:?}\n"
                ));
                return false;
            }
        };
        let resident_bytes = prepared.total_user_bo_bytes;
        if !state.make_room(resident_bytes, budget.resident_program_bytes) {
            return false;
        }
        let accounted_bytes = prepared.total_accounted_bytes;
        profile::event(MetricEvent::DirectMemoryBreakdown {
            executable_bytes: prepared.executable_bo_bytes,
            binding_bytes: prepared.binding_bo_bytes,
            hpi_bytes: prepared.hpi_bo_bytes,
            command_bytes: prepared.command_bo_bytes,
            preemption_bytes: prepared.preemption_bytes,
        });
        state.resident_bytes = state.resident_bytes.saturating_add(resident_bytes);
        state.resident_high_water = state.resident_high_water.max(state.resident_bytes);
        state.accounted_high_water = state
            .accounted_high_water
            .max(state.resident_bytes.saturating_add(budget.preemption_bytes));
        RESIDENT_HIGH_WATER.fetch_max(state.resident_high_water, Ordering::Relaxed);
        ACCOUNTED_HIGH_WATER.fetch_max(state.accounted_high_water, Ordering::Relaxed);
        profile::event(MetricEvent::MemoryState {
            resident_bytes: state.resident_bytes,
            resident_high_water: state.resident_high_water,
            accounted_high_water: state.accounted_high_water,
            resident_limit: budget.resident_program_bytes,
        });
        if state.entries.try_reserve(1).is_err() {
            return false;
        }
        state.entries.push(ResidentProgram {
            generation,
            key,
            prepared,
            resident_bytes,
            accounted_bytes,
            last_used: clock,
            pins: 0,
        });
        profile::note(&alloc::format!(
            "[npu direct] loaded resident={} accounted={} cache_resident={}\n",
            resident_bytes,
            accounted_bytes,
            state.resident_bytes
        ));
        state.entries.len() - 1
    };
    let entry = &mut state.entries[index];
    entry.last_used = clock;
    entry.pins = entry.pins.saturating_add(1);
    let result = dispatch(context, &entry.prepared, inputs, outputs);
    profile::note(&alloc::format!("[npu direct] dispatch result={result}\n"));
    entry.pins = entry.pins.saturating_sub(1);
    result
}

pub(super) fn run_const(
    weight_ptr: usize,
    m: i64,
    inputs: &[&Operand],
    outputs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    let generation = crate::backend::buffer_generation();
    let mut key = ExactKeyBuilder::new(b"direct-runtime/constant-program");
    key.u64(b"generation", generation)
        .u64(b"weight-address", weight_ptr as u64)
        .i64(b"m", m);
    run_program(generation, key.finish(), inputs, outputs, compile)
}

pub(super) fn run_subgraph(
    key: ExactSemanticKey,
    inputs: &[&Operand],
    outputs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    run_program(
        crate::backend::buffer_generation(),
        key,
        inputs,
        outputs,
        compile,
    )
}

pub(super) fn memory_high_water() -> (u64, u64) {
    (
        RESIDENT_HIGH_WATER.load(Ordering::Relaxed),
        ACCOUNTED_HIGH_WATER.load(Ordering::Relaxed),
    )
}

pub(super) fn dispatch_count() -> u64 {
    DISPATCHES.load(Ordering::Relaxed)
}

pub(super) fn run_blob_selftest_multi(
    blob: &[u8],
    inputs: &[&[f32]],
    output: &mut [f32],
) -> Result<usize, RuntimeError> {
    let context = context()?;
    let parameters = context.device.parameters()?;
    let budget = RuntimeBudget::query(&parameters)?;
    let elf = parse_intel_elf_with_resident_limit(blob, budget.resident_program_bytes)
        .map_err(|_| RuntimeError::Incompatible)?;
    let hpi =
        inspect_hpi_requirements(&elf, &parameters).map_err(|_| RuntimeError::Incompatible)?;
    let input_bytes: Vec<&[u8]> = inputs
        .iter()
        .map(|input| {
            // SAFETY: f32 has no invalid bit patterns and this is an exact immutable byte view.
            unsafe {
                core::slice::from_raw_parts(
                    input.as_ptr().cast::<u8>(),
                    input.len() * core::mem::size_of::<f32>(),
                )
            }
        })
        .collect();
    let prepared = prepare_inference(
        context.device,
        &parameters,
        &elf,
        &hpi,
        &input_bytes,
        budget.preparation(),
    )
    .map_err(|_| RuntimeError::Incompatible)?;
    release_host_writes();
    context
        .queue
        .submit(&prepared.handles, prepared.commands_offset(), 0)?;
    let now = LinuxOs
        .monotonic_now_ns()
        .map_err(|error| RuntimeError::Os(error.0))?;
    if prepared
        .command
        .wait_idle(Deadline::MonotonicNs(now.saturating_add(2_000_000_000)))?
        != 0
    {
        return Err(RuntimeError::DeviceLost);
    }
    let bytes = prepared.output_bytes(0).ok_or(RuntimeError::Incompatible)?;
    if bytes % size_of::<f32>() != 0 || output.len() < bytes / size_of::<f32>() {
        return Err(RuntimeError::InvalidArgument);
    }
    // SAFETY: output has capacity for `bytes` and is writable for the duration of the copy.
    let target =
        unsafe { core::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), bytes) };
    prepared
        .copy_output(0, target)
        .map_err(|_| RuntimeError::Incompatible)?;
    Ok(bytes / size_of::<f32>())
}

#[cfg(feature = "bench-hardware")]
pub(crate) struct DirectBenchmarkFixture {
    prepared: PreparedInference<'static>,
}

#[cfg(feature = "bench-hardware")]
impl DirectBenchmarkFixture {
    pub(super) fn new(blob: &[u8], inputs: &[&[f32]]) -> Result<Self, RuntimeError> {
        let context = context()?;
        let parameters = context.device.parameters()?;
        let budget = RuntimeBudget::query(&parameters)?;
        let elf = parse_intel_elf_with_resident_limit(blob, budget.resident_program_bytes)
            .map_err(|_| RuntimeError::Incompatible)?;
        let hpi =
            inspect_hpi_requirements(&elf, &parameters).map_err(|_| RuntimeError::Incompatible)?;
        let input_bytes: Vec<&[u8]> = inputs
            .iter()
            .map(|input| {
                // SAFETY: exact immutable byte view of a live f32 slice.
                unsafe {
                    core::slice::from_raw_parts(
                        input.as_ptr().cast::<u8>(),
                        input.len() * size_of::<f32>(),
                    )
                }
            })
            .collect();
        let prepared = prepare_inference(
            context.device,
            &parameters,
            &elf,
            &hpi,
            &input_bytes,
            budget.preparation(),
        )
        .map_err(|_| RuntimeError::Incompatible)?;
        Ok(Self { prepared })
    }

    pub(super) fn execute(&self) -> Result<(), RuntimeError> {
        let context = context()?;
        self.prepared
            .zero_outputs()
            .map_err(|_| RuntimeError::Incompatible)?;
        release_host_writes();
        context
            .queue
            .submit(&self.prepared.handles, self.prepared.commands_offset(), 0)?;
        let now = LinuxOs
            .monotonic_now_ns()
            .map_err(|error| RuntimeError::Os(error.0))?;
        if self
            .prepared
            .command
            .wait_idle(Deadline::MonotonicNs(now.saturating_add(2_000_000_000)))?
            != 0
        {
            return Err(RuntimeError::DeviceLost);
        }
        Ok(())
    }

    pub(super) fn copy_output(&self, output: &mut [f32]) -> Result<usize, RuntimeError> {
        let bytes = self
            .prepared
            .output_bytes(0)
            .ok_or(RuntimeError::Incompatible)?;
        if bytes % size_of::<f32>() != 0 || output.len() < bytes / size_of::<f32>() {
            return Err(RuntimeError::InvalidArgument);
        }
        // SAFETY: output capacity was checked.
        let target =
            unsafe { core::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), bytes) };
        self.prepared
            .copy_output(0, target)
            .map_err(|_| RuntimeError::Incompatible)?;
        Ok(bytes / size_of::<f32>())
    }
}
