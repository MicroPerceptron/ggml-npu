//! Direct IVPU execution session (default runtime) with byte-budgeted resident programs.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};

use super::direct::budget::RuntimeBudget;
use super::direct::elf::parse_intel_elf_with_resident_limit;
use super::direct::hpi::inspect_hpi_requirements;
use super::direct::link::{visit_binding_requirements, DeviceMemoryClass};
use super::direct::prepare::{
    prepare_inference, prepare_inference_shared, rebind_inference_shared, PreparedInference,
};
use super::direct::transport::release_host_writes;
use super::direct::{CommandQueue, IvpuDevice};
use super::profile;
use crate::artifact_cache::WaitProvider;
use crate::host::{ClockService, ConfigService, HostEnvironment};
use crate::identity::{ExactKeyBuilder, ExactSemanticKey};
use crate::metrics::MetricEvent;
use crate::ops::Operand;
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
    /// Resident-cache identity (program key for warm-reusable artifacts, else full binding key).
    key: ExactSemanticKey,
    program_key: ExactSemanticKey,
    binding_fp: UnsafeCell<ExactSemanticKey>,
    prepared: UnsafeCell<PreparedInference<'static>>,
    /// Retained only for small artifacts so binding sets can re-JIT without recompile (#5).
    warm_artifact: Option<alloc::sync::Arc<[u8]>>,
    resident_bytes: u64,
    accounted_bytes: u64,
    last_used: AtomicU64,
    pins: AtomicU32,
    dispatch_locked: AtomicBool,
}

struct LoadingProgram {
    generation: u64,
    key: ExactSemanticKey,
    state: Arc<AtomicU32>,
}

struct CacheState {
    entries: Vec<Arc<ResidentProgram>>,
    loading: Vec<LoadingProgram>,
    resident_bytes: u64,
    reserved_bytes: u64,
    resident_high_water: u64,
    accounted_high_water: u64,
    clock: u64,
    generation: u64,
}

fn oldest_unpinned(entries: impl Iterator<Item = (usize, u32, u64)>) -> Option<usize> {
    entries
        .filter(|(_, pins, _)| *pins == 0)
        .min_by_key(|(_, _, last_used)| *last_used)
        .map(|(index, _, _)| index)
}

impl CacheState {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            loading: Vec::new(),
            resident_bytes: 0,
            reserved_bytes: 0,
            resident_high_water: 0,
            accounted_high_water: 0,
            clock: 0,
            generation: 0,
        }
    }

    fn evict_one(&mut self) -> Option<Arc<ResidentProgram>> {
        let candidate = oldest_unpinned(self.entries.iter().enumerate().map(|(index, entry)| {
            (
                index,
                entry.pins.load(Ordering::Acquire),
                entry.last_used.load(Ordering::Relaxed),
            )
        }));
        let index = candidate?;
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
            reason: "capacity",
            bytes: entry.resident_bytes,
        });
        Some(entry)
    }

    fn make_room(
        &mut self,
        bytes: u64,
        high_water: u64,
        low_water: u64,
    ) -> Option<Vec<Arc<ResidentProgram>>> {
        let mut evicted = Vec::new();
        if bytes > high_water {
            return None;
        }
        let projected = self
            .resident_bytes
            .checked_add(self.reserved_bytes)
            .and_then(|total| total.checked_add(bytes));
        if projected.is_some_and(|total| total <= high_water) {
            return Some(evicted);
        }
        let target_before_insert = low_water.saturating_sub(bytes);
        while self.resident_bytes.saturating_add(self.reserved_bytes) > target_before_insert {
            evicted.push(self.evict_one()?);
        }
        Some(evicted)
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
const LOAD_RUNNING: u32 = 1;
const LOAD_COMPLETE: u32 = 2;

struct PinnedProgram {
    program: Arc<ResidentProgram>,
}

impl Drop for PinnedProgram {
    fn drop(&mut self) {
        self.program.pins.fetch_sub(1, Ordering::AcqRel);
    }
}

struct DispatchGuard<'a> {
    program: &'a ResidentProgram,
}

impl ResidentProgram {
    fn prepared(&self) -> &PreparedInference<'static> {
        // SAFETY: exclusive dispatch lock or single-threaded construction before publish.
        unsafe { &*self.prepared.get() }
    }

    fn prepared_mut(&self) -> &mut PreparedInference<'static> {
        // SAFETY: caller holds `dispatch_locked` exclusively for this program.
        unsafe { &mut *self.prepared.get() }
    }

    fn binding_fp(&self) -> &ExactSemanticKey {
        // SAFETY: exclusive dispatch lock or cache lock during lookup publication.
        unsafe { &*self.binding_fp.get() }
    }

    fn set_binding_fp(&self, fingerprint: ExactSemanticKey) {
        // SAFETY: caller holds `dispatch_locked` exclusively for this program.
        unsafe {
            *self.binding_fp.get() = fingerprint;
        }
    }

    fn lock_dispatch(&self) -> DispatchGuard<'_> {
        while self
            .dispatch_locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        DispatchGuard { program: self }
    }
}

impl Drop for DispatchGuard<'_> {
    fn drop(&mut self) {
        self.program.dispatch_locked.store(false, Ordering::Release);
    }
}

struct LoadingGuard {
    state: Arc<AtomicU32>,
}

impl Drop for LoadingGuard {
    fn drop(&mut self) {
        self.state.store(LOAD_COMPLETE, Ordering::Release);
        {
            let mut guard = PROGRAMS.lock();
            let state = guard.state();
            if let Some(index) = state
                .loading
                .iter()
                .position(|loading| Arc::ptr_eq(&loading.state, &self.state))
            {
                state.loading.swap_remove(index);
            }
        }
        crate::host::builtin().wait_provider().wake_all(&self.state);
    }
}

struct ReservationGuard {
    bytes: u64,
    active: bool,
}

impl ReservationGuard {
    fn release(&mut self, state: &mut CacheState) {
        if self.active {
            state.reserved_bytes = state.reserved_bytes.saturating_sub(self.bytes);
            self.active = false;
        }
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if self.active {
            let mut guard = PROGRAMS.lock();
            let state = guard.state();
            state.reserved_bytes = state.reserved_bytes.saturating_sub(self.bytes);
            self.active = false;
        }
    }
}

enum ResidentAction {
    Hit(PinnedProgram),
    Wait(Arc<AtomicU32>),
    Load(LoadingGuard),
}

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
    if block == 0 || !operand.elems.is_multiple_of(block) {
        return None;
    }
    operand.elems.checked_div(block)?.checked_mul(type_size)
}

fn operand_bytes(operand: &Operand, bytes: usize) -> Option<&[u8]> {
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
    if program.output_is_direct(index) {
        return true;
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

    let prof = profile::enabled() || crate::trace::perf_enabled();
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
    let now = match crate::host::builtin().clock().monotonic_now_ns() {
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
        let bytes_in: u64 = input_bytes
            .iter()
            .enumerate()
            .filter(|(index, _)| !program.input_is_direct(*index))
            .map(|(_, bytes)| bytes.len() as u64)
            .sum();
        let bytes_out: u64 = outputs
            .iter()
            .enumerate()
            .filter(|(index, _)| !program.output_is_direct(*index))
            .filter_map(|(_, output)| operand_storage_bytes(output))
            .map(|bytes| bytes as u64)
            .sum();
        let imported = program.direct_binding_count() as u64;
        let staged = (inputs.len() + outputs.len()) as u64 - imported;
        profile::record(
            t0,
            t1,
            tmid,
            t2,
            t3,
            0,
            0,
            imported,
            staged,
            bytes_in.saturating_add(bytes_out),
            bytes_in,
            bytes_out,
        );
        crate::trace::perf(
            "dispatch",
            &alloc::format!(
                ",\"stage_in_ns\":{},\"submit_ns\":{},\"wait_ns\":{},\"copy_out_ns\":{},\"direct_bindings\":{},\"staged_bindings\":{},\"bytes_in\":{bytes_in},\"bytes_out\":{bytes_out}",
                t1.saturating_sub(t0),
                tmid.saturating_sub(t1),
                t2.saturating_sub(tmid),
                t3.saturating_sub(t2),
                imported,
                staged,
            ),
        );
    }
    true
}

struct PreparedCandidate {
    prepared: PreparedInference<'static>,
    resident_bytes: u64,
    accounted_bytes: u64,
    reservation: ReservationGuard,
    artifact: alloc::sync::Arc<[u8]>,
}

/// Small weightless/glue artifacts can keep a host copy for warm IO rebind across similar programs
/// without double-keeping baked-weight blobs that dominate resident FFN programs (#5).
const WARM_REUSE_MAX_ARTIFACT_BYTES: usize = 512 * 1024;

fn binding_fingerprint(
    key: &ExactSemanticKey,
    inputs: &[&Operand],
    outputs: &[&Operand],
) -> ExactSemanticKey {
    let mut builder = ExactKeyBuilder::new(b"direct-runtime/shared-bindings-v1");
    builder.field(b"program", &key.digest().0);
    for (direction, operands) in [(0u8, inputs), (1u8, outputs)] {
        for (index, operand) in operands.iter().enumerate() {
            builder
                .u8(b"direction", direction)
                .u64(b"index", index as u64);
            if let Some(view) = operand.shared {
                builder
                    .u64(b"provider", view.identity.provider)
                    .u64(b"object", view.identity.object)
                    .u64(b"generation", view.identity.generation)
                    .u64(b"offset", view.offset as u64)
                    .u64(b"bytes", view.bytes as u64);
            } else {
                builder.u8(b"staged", 1);
            }
        }
    }
    builder.finish()
}

/// Cache identity for a prepared program: warm-reusable artifacts share one resident entry across
/// binding sets (L0 SG_GRAPHS analogue); large baked programs keep the binding-set fold.
fn resident_cache_key(
    program_key: ExactSemanticKey,
    binding_fp: ExactSemanticKey,
    artifact_len: usize,
) -> ExactSemanticKey {
    if artifact_len <= WARM_REUSE_MAX_ARTIFACT_BYTES {
        program_key
    } else {
        binding_fp
    }
}

fn warm_reuse_eligible(artifact_len: usize, prepared: &PreparedInference<'_>) -> bool {
    artifact_len <= WARM_REUSE_MAX_ARTIFACT_BYTES && prepared.warm_rebind.is_some()
}

fn prepare_candidate(
    context: &DirectContext,
    parameters: &super::direct::DeviceParameters,
    budget: RuntimeBudget,
    inputs: &[&Operand],
    outputs: &[&Operand],
    blob: Vec<u8>,
) -> Option<PreparedCandidate> {
    profile::note(&alloc::format!(
        "[npu direct] artifact bytes={} compiled/cache-loaded\n",
        blob.len()
    ));
    let artifact: alloc::sync::Arc<[u8]> = alloc::sync::Arc::from(blob.into_boxed_slice());
    let elf = match parse_intel_elf_with_resident_limit(&artifact, budget.resident_program_bytes) {
        Ok(elf) => elf,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] unsupported artifact before load: {error:?}\n"
            ));
            return None;
        }
    };
    let reservation_bytes = elf.inspection().resident_bytes;
    let evicted = {
        let mut guard = PROGRAMS.lock();
        let state = guard.state();
        let evicted = state.make_room(
            reservation_bytes,
            budget.resident_program_bytes,
            budget.resident_low_water_bytes,
        )?;
        state.reserved_bytes = state.reserved_bytes.saturating_add(reservation_bytes);
        evicted
    };
    // Resource destruction is intentionally outside the global cache lock.
    drop(evicted);
    let reservation = ReservationGuard {
        bytes: reservation_bytes,
        active: true,
    };
    let hpi = match inspect_hpi_requirements(&elf, parameters) {
        Ok(hpi) => hpi,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] unsupported HPI metadata: {error:?}\n"
            ));
            return None;
        }
    };
    let mut initial_inputs = Vec::new();
    if initial_inputs.try_reserve_exact(inputs.len()).is_err() {
        return None;
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
        return None;
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
            return None;
        };
        initial_inputs.push(data);
    }
    let shared_bindings = !crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_DISABLE_SHARED_BINDINGS");
    let input_views: alloc::vec::Vec<_> = inputs
        .iter()
        .map(|operand| shared_bindings.then_some(operand.shared).flatten())
        .collect();
    let output_views: alloc::vec::Vec<_> = outputs
        .iter()
        .map(|operand| shared_bindings.then_some(operand.shared).flatten())
        .collect();
    let prepared = match prepare_inference_shared(
        context.device,
        parameters,
        &elf,
        &hpi,
        &initial_inputs,
        &input_views,
        &output_views,
        budget.preparation(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] artifact load failed: {error:?}\n"
            ));
            return None;
        }
    };
    let resident_bytes = prepared.total_user_bo_bytes;
    let accounted_bytes = prepared.total_accounted_bytes;
    for binding in &prepared.bindings {
        let allocation = binding
            .bo
            .shared_identity()
            .map_or(binding.bo.vpu_address(), |identity| identity.object);
        if !binding.direct {
            profile::event(MetricEvent::StagingAllocation {
                allocation,
                bytes: binding.logical_bytes,
            });
        }
    }
    profile::record_direct_memory_breakdown(
        prepared.executable_bo_bytes,
        prepared.binding_bo_bytes,
        prepared.hpi_bo_bytes,
        prepared.command_bo_bytes,
        prepared.preemption_bytes,
    );
    Some(PreparedCandidate {
        prepared,
        resident_bytes,
        accounted_bytes,
        reservation,
        artifact,
    })
}

fn try_warm_rebind(
    context: &DirectContext,
    budget: RuntimeBudget,
    program: &ResidentProgram,
    binding_fp: &ExactSemanticKey,
    inputs: &[&Operand],
    outputs: &[&Operand],
) -> bool {
    if program.binding_fp() == binding_fp {
        return true;
    }
    let Some(artifact) = program.warm_artifact.as_ref() else {
        profile::note("[npu direct] binding set changed without warm artifact; miss\n");
        return false;
    };
    if program.prepared().warm_rebind.is_none() {
        return false;
    }
    let elf = match parse_intel_elf_with_resident_limit(artifact, budget.resident_program_bytes) {
        Ok(elf) => elf,
        Err(_) => return false,
    };
    let mut initial_inputs = Vec::new();
    if initial_inputs.try_reserve_exact(inputs.len()).is_err() {
        return false;
    }
    for (index, input) in inputs.iter().enumerate() {
        let Some(bytes) = program.prepared().input_bytes(index) else {
            return false;
        };
        let Some(data) = operand_bytes(input, bytes) else {
            return false;
        };
        initial_inputs.push(data);
    }
    let shared_bindings = !crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_DISABLE_SHARED_BINDINGS");
    let input_views: alloc::vec::Vec<_> = inputs
        .iter()
        .map(|operand| shared_bindings.then_some(operand.shared).flatten())
        .collect();
    let output_views: alloc::vec::Vec<_> = outputs
        .iter()
        .map(|operand| shared_bindings.then_some(operand.shared).flatten())
        .collect();
    match rebind_inference_shared(
        context.device,
        program.prepared_mut(),
        &elf,
        &initial_inputs,
        &input_views,
        &output_views,
    ) {
        Ok(()) => {
            program.set_binding_fp(binding_fp.clone());
            profile::note("[npu direct] warm binding rebind\n");
            true
        }
        Err(error) => {
            profile::note(&alloc::format!(
                "[npu direct] warm binding rebind failed: {error:?}\n"
            ));
            false
        }
    }
}

fn evict_unpinned(program: &Arc<ResidentProgram>) {
    let mut guard = PROGRAMS.lock();
    let state = guard.state();
    if let Some(index) = state
        .entries
        .iter()
        .position(|entry| Arc::ptr_eq(entry, program) && entry.pins.load(Ordering::Acquire) == 0)
    {
        let entry = state.entries.swap_remove(index);
        state.resident_bytes = state.resident_bytes.saturating_sub(entry.resident_bytes);
        profile::event(MetricEvent::CacheEviction {
            cache: "direct-programs",
            reason: "rebind-miss",
            bytes: entry.resident_bytes,
        });
        drop(entry);
    }
}

fn run_program(
    generation: u64,
    program_key: ExactSemanticKey,
    inputs: &[&Operand],
    outputs: &[&Operand],
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    let binding_fp = binding_fingerprint(&program_key, inputs, outputs);
    profile::note(&alloc::format!(
        "[npu direct] run generation={generation} inputs={} outputs={} program={} bindings={}\n",
        inputs.len(),
        outputs.len(),
        program_key.digest().hex(),
        binding_fp.digest().hex()
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
    let mut compile = Some(compile);
    // First probe: warm program-key entry, then exact binding-key entry (large programs).
    let lookup_keys = [program_key.clone(), binding_fp.clone()];
    loop {
        let (action, stale) = {
            let mut guard = PROGRAMS.lock();
            let state = guard.state();
            let mut stale = Vec::new();
            if state.generation != generation {
                let mut index = 0;
                while index < state.entries.len() {
                    if state.entries[index].generation != generation
                        && state.entries[index].pins.load(Ordering::Acquire) == 0
                    {
                        let entry = state.entries.swap_remove(index);
                        state.resident_bytes =
                            state.resident_bytes.saturating_sub(entry.resident_bytes);
                        profile::event(MetricEvent::CacheEviction {
                            cache: "direct-programs",
                            reason: "generation",
                            bytes: entry.resident_bytes,
                        });
                        stale.push(entry);
                    } else {
                        index += 1;
                    }
                }
                state.generation = generation;
            }
            state.clock = state.clock.wrapping_add(1);
            let clock = state.clock;
            let hit = state
                .entries
                .iter()
                .find(|entry| {
                    entry.generation == generation
                        && lookup_keys.iter().any(|key| entry.key == *key)
                })
                .map(|entry| {
                    entry.pins.fetch_add(1, Ordering::AcqRel);
                    entry.last_used.store(clock, Ordering::Relaxed);
                    PinnedProgram {
                        program: Arc::clone(entry),
                    }
                });
            let action = if let Some(hit) = hit {
                ResidentAction::Hit(hit)
            } else if let Some(loading) = state.loading.iter().find(|loading| {
                loading.generation == generation
                    && lookup_keys.iter().any(|key| loading.key == *key)
            }) {
                ResidentAction::Wait(Arc::clone(&loading.state))
            } else {
                // Reserve under the program key; final resident key is chosen after compile.
                if state.loading.try_reserve(1).is_err() {
                    return false;
                }
                let loading_state = Arc::new(AtomicU32::new(LOAD_RUNNING));
                state.loading.push(LoadingProgram {
                    generation,
                    key: program_key.clone(),
                    state: Arc::clone(&loading_state),
                });
                ResidentAction::Load(LoadingGuard {
                    state: loading_state,
                })
            };
            (action, stale)
        };
        drop(stale);
        match action {
            ResidentAction::Hit(pinned) => {
                profile::event(MetricEvent::CacheHit {
                    cache: "direct-programs",
                    bytes: pinned.program.resident_bytes,
                });
                profile::note("[npu direct] resident program hit\n");
                let program = Arc::clone(&pinned.program);
                let _dispatch_guard = program.lock_dispatch();
                if !try_warm_rebind(
                    context,
                    budget,
                    &program,
                    &binding_fp,
                    inputs,
                    outputs,
                ) {
                    profile::note("[npu direct] resident hit cannot rebind; reloading\n");
                    drop(_dispatch_guard);
                    drop(pinned);
                    evict_unpinned(&program);
                    continue;
                }
                let result = dispatch(context, program.prepared(), inputs, outputs);
                profile::note(&alloc::format!("[npu direct] dispatch result={result}\n"));
                return result;
            }
            ResidentAction::Wait(loading) => {
                profile::event(MetricEvent::CacheMiss {
                    cache: "direct-programs",
                });
                crate::host::builtin()
                    .wait_provider()
                    .wait_while(&loading, LOAD_RUNNING);
            }
            ResidentAction::Load(loading) => {
                profile::event(MetricEvent::CacheMiss {
                    cache: "direct-programs",
                });
                let Some(blob) = compile.take().and_then(|compile| compile()) else {
                    profile::note("[npu direct] compiler returned no artifact\n");
                    return false;
                };
                let artifact_len = blob.len();
                let Some(mut candidate) =
                    prepare_candidate(context, &parameters, budget, inputs, outputs, blob)
                else {
                    return false;
                };
                let cache_key =
                    resident_cache_key(program_key.clone(), binding_fp.clone(), artifact_len);
                let warm_artifact = warm_reuse_eligible(artifact_len, &candidate.prepared)
                    .then(|| alloc::sync::Arc::clone(&candidate.artifact));
                let program = Arc::new(ResidentProgram {
                    generation,
                    key: cache_key,
                    program_key: program_key.clone(),
                    binding_fp: UnsafeCell::new(binding_fp.clone()),
                    prepared: UnsafeCell::new(candidate.prepared),
                    warm_artifact,
                    resident_bytes: candidate.resident_bytes,
                    accounted_bytes: candidate.accounted_bytes,
                    last_used: AtomicU64::new(0),
                    pins: AtomicU32::new(1),
                    dispatch_locked: AtomicBool::new(false),
                });
                let insertion = {
                    let mut guard = PROGRAMS.lock();
                    let state = guard.state();
                    candidate.reservation.release(state);
                    if state.generation != generation || state.entries.try_reserve(1).is_err() {
                        None
                    } else {
                        let evicted = state.make_room(
                            program.resident_bytes,
                            budget.resident_program_bytes,
                            budget.resident_low_water_bytes,
                        );
                        evicted.inspect(|_| {
                            state.clock = state.clock.wrapping_add(1);
                            program.last_used.store(state.clock, Ordering::Relaxed);
                            state.resident_bytes =
                                state.resident_bytes.saturating_add(program.resident_bytes);
                            state.resident_high_water =
                                state.resident_high_water.max(state.resident_bytes);
                            state.accounted_high_water = state
                                .accounted_high_water
                                .max(state.resident_bytes.saturating_add(budget.preemption_bytes));
                            RESIDENT_HIGH_WATER
                                .fetch_max(state.resident_high_water, Ordering::Relaxed);
                            ACCOUNTED_HIGH_WATER
                                .fetch_max(state.accounted_high_water, Ordering::Relaxed);
                            profile::event(MetricEvent::MemoryState {
                                resident_bytes: state.resident_bytes,
                                resident_high_water: state.resident_high_water,
                                accounted_high_water: state.accounted_high_water,
                                resident_limit: budget.resident_program_bytes,
                            });
                            state.entries.push(Arc::clone(&program));
                        })
                    }
                };
                let Some(evicted) = insertion else {
                    return false;
                };
                drop(evicted);
                drop(loading);
                profile::note(&alloc::format!(
                    "[npu direct] loaded resident={} accounted={} cache_resident warm={}\n",
                    program.resident_bytes,
                    program.accounted_bytes,
                    program.warm_artifact.is_some()
                ));
                let pinned = PinnedProgram { program };
                let _dispatch_guard = pinned.program.lock_dispatch();
                let result = dispatch(context, pinned.program.prepared(), inputs, outputs);
                profile::note(&alloc::format!("[npu direct] dispatch result={result}\n"));
                return result;
            }
        }
    }
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
                    core::mem::size_of_val(*input),
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
    let now = crate::host::builtin()
        .clock()
        .monotonic_now_ns()
        .map_err(|error| RuntimeError::External {
            domain: crate::host::ERROR_DOMAIN,
            code: error.0,
        })?;
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
                        core::mem::size_of_val(*input),
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
        let now = crate::host::builtin()
            .clock()
            .monotonic_now_ns()
            .map_err(|error| RuntimeError::External {
                domain: crate::host::ERROR_DOMAIN,
                code: error.0,
            })?;
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

#[cfg(test)]
mod cache_policy_tests {
    use super::{
        oldest_unpinned, resident_cache_key, WARM_REUSE_MAX_ARTIFACT_BYTES,
    };
    use crate::identity::ExactKeyBuilder;

    #[test]
    fn resident_eviction_never_selects_pinned_programs() {
        let entries = [(0, 1, 1), (1, 0, 3), (2, 0, 2)];
        assert_eq!(oldest_unpinned(entries.into_iter()), Some(2));
        assert_eq!(oldest_unpinned([(0, 1, 1), (1, 2, 2)].into_iter()), None);
    }

    #[test]
    fn warm_reusable_artifacts_cache_by_program_key_only() {
        let mut program = ExactKeyBuilder::new(b"test/program");
        program.u64(b"id", 1);
        let program = program.finish();
        let mut bindings = ExactKeyBuilder::new(b"test/bindings");
        bindings.u64(b"id", 2);
        let bindings = bindings.finish();
        let key = resident_cache_key(program.clone(), bindings.clone(), 1024);
        assert_eq!(key, program);
        assert_ne!(key, bindings);
    }

    #[test]
    fn large_artifacts_keep_binding_set_cache_key() {
        let mut program = ExactKeyBuilder::new(b"test/program");
        program.u64(b"id", 1);
        let program = program.finish();
        let mut bindings = ExactKeyBuilder::new(b"test/bindings");
        bindings.u64(b"id", 2);
        let bindings = bindings.finish();
        let key = resident_cache_key(
            program.clone(),
            bindings.clone(),
            WARM_REUSE_MAX_ARTIFACT_BYTES + 1,
        );
        assert_eq!(key, bindings);
        assert_ne!(key, program);
    }
}
