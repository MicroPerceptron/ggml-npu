use std::ffi::c_void;
use std::future::{pending, Future};
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Wake, Waker};

use ggml_npu::runtime::async_runtime::mock::{
    deterministic_wait_context, DeterministicAtomicWait, MockReactor, ScriptedOutcome, VirtualClock,
};
use ggml_npu::runtime::async_runtime::*;
use ggml_npu::runtime::{
    artifact::ArtifactEnvelope, AccessMode, Binding, Deadline, DeviceId, MemoryId, MemoryRequest,
    RuntimeError, RuntimeHal, SubmissionBindings,
};

fn ids(value: u64) -> (OperationId, DeviceEpoch) {
    (
        OperationId::new(value).unwrap(),
        DeviceEpoch::new(1).unwrap(),
    )
}

fn wait_context() -> (
    Arc<VirtualClock>,
    Arc<DeterministicAtomicWait>,
    Arc<MockReactor>,
    WaitContext,
) {
    let clock = Arc::new(VirtualClock::new(0));
    let atomic = Arc::new(DeterministicAtomicWait::new(Arc::clone(&clock)));
    let reactor = Arc::new(MockReactor::new(Arc::clone(&clock), 1));
    let context = deterministic_wait_context(
        Arc::clone(&clock),
        Arc::clone(&atomic),
        Arc::clone(&reactor),
    );
    (clock, atomic, reactor, context)
}

fn success(id: OperationId, epoch: DeviceEpoch, sequence: u64) -> CompletionInfo {
    CompletionInfo {
        operation_id: id,
        device_epoch: epoch,
        queue_sequence: sequence,
        progress: Progress::default(),
    }
}

#[derive(Debug)]
struct Lease;

type TestOperation = OwnedOperation<Lease, Lease, Lease, Lease>;

struct ImmediateQueue;

impl Submitter for ImmediateQueue {
    type Operation = TestOperation;

    fn submit(
        &self,
        operation: Self::Operation,
    ) -> Result<Completion, SubmitFailure<Self::Operation>> {
        let id = operation.id();
        let epoch = operation.epoch();
        let (completion, control) =
            CompletionControl::pending(id, epoch, 7, operation, false, None);
        control.succeed(true);
        Ok(completion)
    }
}

#[derive(Debug)]
enum RunError {
    Submit,
    Complete(CompletionError),
}

impl From<CompletionError> for RunError {
    fn from(error: CompletionError) -> Self {
        Self::Complete(error)
    }
}

async fn idiomatic_submit(
    queue: &ImmediateQueue,
    operation: TestOperation,
) -> Result<CompletionInfo, RunError> {
    let info = queue
        .submit(operation)
        .map_err(|_| RunError::Submit)?
        .await?;
    Ok(info)
}

fn operation(id: u64) -> TestOperation {
    let (id, epoch) = ids(id);
    OwnedOperation::new(
        id,
        epoch,
        Arc::new(Lease),
        Arc::new(Lease),
        Box::new([]),
        Lease,
        Deadline::Infinite,
    )
}

#[test]
fn native_async_await_is_the_typed_api() {
    let (_, _, _, wait) = wait_context();
    let executor = CallerExecutor::new(4);
    let info = executor
        .block_on(
            idiomatic_submit(&ImmediateQueue, operation(1)),
            &wait,
            Deadline::Infinite,
        )
        .unwrap()
        .unwrap();
    assert_eq!(info.operation_id.get(), 1);
    let info = executor
        .block_on(
            submit_and_wait(&ImmediateQueue, operation(20)),
            &wait,
            Deadline::Infinite,
        )
        .unwrap()
        .unwrap();
    assert_eq!(info.operation_id.get(), 20);

    let error = RunError::Complete(CompletionError::ExecutionFailed { domain: 1, code: 2 });
    assert!(matches!(
        error,
        RunError::Complete(CompletionError::ExecutionFailed { code: 2, .. })
    ));
}

struct DropLease(Arc<AtomicUsize>);

impl Drop for DropLease {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn detached_completion_retains_resources_until_reclamation_is_safe() {
    let drops = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(InFlightRegistry::new());
    let (id, epoch) = ids(2);
    let (completion, control) = CompletionControl::pending(
        id,
        epoch,
        1,
        DropLease(Arc::clone(&drops)),
        true,
        Some(Arc::clone(&registry)),
    );
    drop(completion);
    assert_eq!(registry.len(), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);

    assert!(control.succeed(false));
    assert_eq!(registry.len(), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    control.mark_reclaim_safe();
    assert!(registry.is_empty());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

struct IndeterminateQueue {
    registry: Arc<InFlightRegistry>,
    control: Mutex<Option<CompletionControl>>,
}

impl Submitter for IndeterminateQueue {
    type Operation = DropLease;

    fn submit(
        &self,
        operation: Self::Operation,
    ) -> Result<Completion, SubmitFailure<Self::Operation>> {
        let (id, epoch) = ids(21);
        let (completion, control) = CompletionControl::pending(
            id,
            epoch,
            1,
            operation,
            false,
            Some(Arc::clone(&self.registry)),
        );
        *self.control.lock().unwrap() = Some(control);
        Err(SubmitFailure::Indeterminate {
            error: SubmitError::External {
                domain: 1,
                code: -1,
            },
            completion,
        })
    }
}

#[test]
fn indeterminate_acceptance_never_returns_resources_to_the_caller() {
    let drops = Arc::new(AtomicUsize::new(0));
    let queue = IndeterminateQueue {
        registry: Arc::new(InFlightRegistry::new()),
        control: Mutex::new(None),
    };
    let completion = match queue.submit(DropLease(Arc::clone(&drops))) {
        Err(SubmitFailure::Indeterminate { completion, .. }) => completion,
        _ => panic!("scripted submit must be indeterminate"),
    };
    drop(completion);
    assert_eq!(queue.registry.len(), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    let control = queue.control.lock().unwrap().take().unwrap();
    control.mark_device_lost(1, false);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    control.mark_reclaim_safe();
    assert!(queue.registry.is_empty());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn timeout_is_not_terminal_and_spurious_wakes_are_tolerated() {
    let (_, atomic, reactor, wait) = wait_context();
    atomic.inject_spurious(2);
    let (id, epoch) = ids(3);
    let (completion, control) = CompletionControl::pending(id, epoch, 1, (), false, None);
    assert_eq!(
        completion.wait(&wait, Deadline::MonotonicNs(5)).unwrap(),
        WaitStatus::TimedOut
    );
    assert!(completion.try_complete().is_none());

    reactor.schedule(7, control, ScriptedOutcome::Success { reclaim_safe: true });
    assert!(matches!(
        completion.wait(&wait, Deadline::MonotonicNs(10)).unwrap(),
        WaitStatus::Ready(Ok(_))
    ));
}

#[derive(Default)]
struct CountingWake(AtomicUsize);

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn multiple_observers_are_all_woken() {
    let (id, epoch) = ids(4);
    let (mut first, control) = CompletionControl::pending(id, epoch, 1, (), false, None);
    let mut second = first.clone();
    let first_wake = Arc::new(CountingWake::default());
    let second_wake = Arc::new(CountingWake::default());
    let first_waker = Waker::from(Arc::clone(&first_wake));
    let second_waker = Waker::from(Arc::clone(&second_wake));
    assert!(Pin::new(&mut first)
        .poll(&mut Context::from_waker(&first_waker))
        .is_pending());
    assert!(Pin::new(&mut second)
        .poll(&mut Context::from_waker(&second_waker))
        .is_pending());
    control.succeed(true);
    assert_eq!(first_wake.0.load(Ordering::Relaxed), 1);
    assert_eq!(second_wake.0.load(Ordering::Relaxed), 1);
}

struct RegisterRaceSource {
    id: OperationId,
    epoch: DeviceEpoch,
    ready: AtomicBool,
}

impl CompletionSource for RegisterRaceSource {
    fn operation_id(&self) -> OperationId {
        self.id
    }

    fn device_epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    fn try_complete(&self) -> Option<CompletionResult> {
        self.ready
            .load(Ordering::Acquire)
            .then_some(Ok(success(self.id, self.epoch, 1)))
    }

    fn register_waker(&self, _observer: u64, waker: &Waker) {
        self.ready.store(true, Ordering::Release);
        waker.wake_by_ref();
    }

    fn unregister_waker(&self, _observer: u64) {}

    fn request_cancel(&self) -> CancellationRequest {
        CancellationRequest::Unsupported
    }
}

#[test]
fn register_then_recheck_closes_completion_race() {
    let (id, epoch) = ids(5);
    let source = Arc::new(RegisterRaceSource {
        id,
        epoch,
        ready: AtomicBool::new(false),
    });
    let mut completion = Completion::from_source(source);
    let waker = Waker::noop();
    assert!(Pin::new(&mut completion)
        .poll(&mut Context::from_waker(waker))
        .is_ready());
}

#[test]
fn wait_any_and_all_use_exact_snapshots() {
    let (_, _, reactor, wait) = wait_context();
    let (id1, epoch) = ids(6);
    let (id2, _) = ids(7);
    let (first, first_control) = CompletionControl::pending(id1, epoch, 1, (), false, None);
    let (second, second_control) = CompletionControl::pending(id2, epoch, 2, (), false, None);
    reactor.schedule(
        1,
        first_control,
        ScriptedOutcome::Success { reclaim_safe: true },
    );
    reactor.schedule(
        3,
        second_control,
        ScriptedOutcome::Success { reclaim_safe: true },
    );
    let completions = [first, second];
    assert_eq!(
        wait_many(
            &completions,
            WaitMode::Any,
            &wait,
            Deadline::MonotonicNs(10)
        )
        .unwrap(),
        WaitManyOutcome::Ready(vec![0])
    );
    assert_eq!(
        wait_many(
            &completions,
            WaitMode::All,
            &wait,
            Deadline::MonotonicNs(10)
        )
        .unwrap(),
        WaitManyOutcome::Ready(vec![0, 1])
    );
}

#[test]
fn cancellation_is_request_and_acknowledgement_with_success_races() {
    let (id, epoch) = ids(8);
    let (completion, control) = CompletionControl::pending(id, epoch, 1, (), true, None);
    assert_eq!(completion.request_cancel(), CancellationRequest::Requested);
    assert_eq!(control.cancellation_state(), CancellationState::Requested);
    assert!(control.succeed(true));
    assert!(!control.acknowledge_cancel(true, true));
    assert!(completion.try_complete().unwrap().is_ok());

    let (id, _) = ids(9);
    let (cancelled, control) = CompletionControl::pending(id, epoch, 2, (), true, None);
    assert_eq!(cancelled.request_cancel(), CancellationRequest::Requested);
    assert!(control.acknowledge_cancel(true, true));
    assert!(matches!(
        cancelled.try_complete(),
        Some(Err(CompletionError::Cancelled {
            may_have_executed: true
        }))
    ));

    let (id, _) = ids(10);
    let (dropped, control) = CompletionControl::pending(id, epoch, 3, (), true, None);
    drop(dropped);
    assert_eq!(control.cancellation_state(), CancellationState::Cancellable);
}

#[test]
fn device_loss_fans_out_but_reclamation_waits_for_epoch_teardown() {
    let drops = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(InFlightRegistry::new());
    let fanout = DeviceEpochFanout::default();
    let epoch = DeviceEpoch::new(2).unwrap();
    let mut completions = Vec::new();
    for value in 11..=12 {
        let (completion, control) = CompletionControl::pending(
            OperationId::new(value).unwrap(),
            epoch,
            value,
            DropLease(Arc::clone(&drops)),
            false,
            Some(Arc::clone(&registry)),
        );
        fanout.track(&control);
        completions.push(completion);
    }
    assert_eq!(fanout.mark_lost(epoch, 99), 2);
    assert_eq!(registry.len(), 2);
    assert!(completions.iter().all(|completion| matches!(
        completion.try_complete(),
        Some(Err(CompletionError::DeviceLost { reason: 99, .. }))
    )));
    fanout.mark_epoch_reclaim_safe(epoch);
    assert!(registry.is_empty());
    assert_eq!(drops.load(Ordering::Relaxed), 2);
}

#[test]
fn caller_executor_is_bounded_and_shutdown_cancels_detached_tasks() {
    let executor = CallerExecutor::new(1);
    let first = match executor.spawn(pending::<()>()) {
        Ok(handle) => handle,
        Err(_) => panic!("first task must fit"),
    };
    assert!(matches!(
        executor.spawn(async {}),
        Err(SpawnError::AtCapacity(_))
    ));
    executor.run_until_stalled(1);
    assert_eq!(first.state(), TaskState::Waiting);
    assert_eq!(executor.shutdown(), 1);
    assert_eq!(first.state(), TaskState::Cancelled);
    assert!(matches!(
        executor.spawn(async {}),
        Err(SpawnError::ShuttingDown(_))
    ));
}

fn blocking_request(value: u64) -> BlockingRequest {
    let (id, epoch) = ids(value);
    BlockingRequest {
        operation_id: id,
        device_epoch: epoch,
        queue_sequence: value,
        job: Box::new(move || Ok(success(id, epoch, value))),
    }
}

#[test]
fn caller_driven_blocking_port_saturates_and_shuts_down_explicitly() {
    let port = CallerDrivenBlockingExecutor::new(1);
    let first = match port.try_spawn(blocking_request(13)) {
        Ok(completion) => completion,
        Err(_) => panic!("first blocking job must fit"),
    };
    assert!(matches!(
        port.try_spawn(blocking_request(14)),
        Err((SubmitError::ResourceLimit, _))
    ));
    assert!(first.try_complete().is_none());
    assert!(port.run_one());
    assert!(first.try_complete().unwrap().is_ok());

    let queued = match port.try_spawn(blocking_request(15)) {
        Ok(completion) => completion,
        Err(_) => panic!("blocking queue must accept after drain"),
    };
    port.shutdown();
    assert!(matches!(
        queued.try_complete(),
        Some(Err(CompletionError::ExecutionFailed { .. }))
    ));
    assert!(matches!(
        port.try_spawn(blocking_request(16)),
        Err((SubmitError::RuntimeShuttingDown, _))
    ));
}

struct PartialQueue;

impl Submitter for PartialQueue {
    type Operation = u32;

    fn submit(
        &self,
        operation: Self::Operation,
    ) -> Result<Completion, SubmitFailure<Self::Operation>> {
        if operation % 2 == 0 {
            return Err(SubmitFailure::Rejected {
                error: SubmitError::ResourceLimit,
                operation,
            });
        }
        let (id, epoch) = ids(u64::from(operation));
        Ok(CompletionControl::ready(Ok(success(
            id,
            epoch,
            u64::from(operation),
        ))))
    }
}

#[test]
fn batch_partial_acceptance_is_explicit_per_operation() {
    let result = BatchSubmitResult::submit_all(&PartialQueue, [1, 2, 3, 4]);
    assert_eq!(
        result
            .accepted
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        result
            .rejected
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        [1, 3]
    );
}

#[test]
fn failed_visibility_transition_quarantines_the_lease() {
    let (_, _, _, wait) = wait_context();
    let executor = CallerExecutor::new(1);
    let (id, epoch) = ids(17);
    let (completion, control) = CompletionControl::pending(id, epoch, 1, (), false, None);
    let transition = VisibilityTransition::new(41u32, VisibilityDomain::Host, completion);
    control.fail(
        CompletionError::ExecutionFailed {
            domain: 7,
            code: -1,
        },
        true,
    );
    let transition_result = executor
        .block_on(transition, &wait, Deadline::Infinite)
        .unwrap();
    let quarantined = match transition_result {
        Ok(_) => panic!("failed visibility work must quarantine"),
        Err(quarantined) => quarantined,
    };
    let (lease, cause) = quarantined.into_recovery_parts();
    assert_eq!(lease, 41);
    assert!(matches!(
        cause,
        CompletionError::ExecutionFailed { code: -1, .. }
    ));
}

#[test]
fn ownership_transition_changes_only_the_typed_owner_marker() {
    let (_, _, _, wait) = wait_context();
    let executor = CallerExecutor::new(1);
    let (id, epoch) = ids(22);
    let completion = CompletionControl::ready(Ok(success(id, epoch, 1)));
    let transition = OwnershipTransition::new(7u32, "adapter-owner", completion);
    let owned = executor
        .block_on(transition, &wait, Deadline::Infinite)
        .unwrap()
        .ok()
        .unwrap();
    let (lease, owner) = owned.into_parts();
    assert_eq!(lease, 7);
    assert_eq!(owner, "adapter-owner");
}

struct AbiContext {
    retains: AtomicUsize,
    releases: AtomicUsize,
    registrations: AtomicUsize,
    unregistrations: AtomicUsize,
    ready: AtomicBool,
    callback: Mutex<Option<(ForeignWakeCallback, usize)>>,
}

unsafe extern "C" fn abi_retain(context: *mut c_void) {
    let context = unsafe { &*context.cast::<AbiContext>() };
    context.retains.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn abi_release(context: *mut c_void) {
    let context = unsafe { &*context.cast::<AbiContext>() };
    context.releases.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn abi_poll(context: *mut c_void, output: *mut CompletionAbiResult) -> i32 {
    let context = unsafe { &*context.cast::<AbiContext>() };
    if !context.ready.load(Ordering::Acquire) {
        return PollAbiStatus::Pending as i32;
    }
    unsafe {
        output.write(CompletionAbiResult::success(
            DeviceEpoch::new(1).unwrap(),
            1,
        ));
    }
    PollAbiStatus::Ready as i32
}

unsafe extern "C" fn abi_register(
    context: *mut c_void,
    callback: ForeignWakeCallback,
    callback_context: *mut c_void,
    registration: *mut u64,
) -> i32 {
    let context = unsafe { &*context.cast::<AbiContext>() };
    context.registrations.fetch_add(1, Ordering::Relaxed);
    *context.callback.lock().unwrap() = Some((callback, callback_context as usize));
    unsafe { registration.write(7) };
    0
}

unsafe extern "C" fn abi_unregister(context: *mut c_void, registration: u64) -> i32 {
    assert_eq!(registration, 7);
    let context = unsafe { &*context.cast::<AbiContext>() };
    context.unregistrations.fetch_add(1, Ordering::Relaxed);
    context.callback.lock().unwrap().take();
    0
}

unsafe extern "C" fn abi_cancel(_context: *mut c_void) -> i32 {
    3
}

static ABI_VTABLE: CompletionAbiVTable = CompletionAbiVTable {
    abi_version: 1,
    retain: abi_retain,
    release: abi_release,
    try_poll: abi_poll,
    register_wake: abi_register,
    unregister_wake: abi_unregister,
    request_cancel: abi_cancel,
};

#[test]
fn ffi_completion_pairs_retain_release_and_callback_unregister() {
    let context = Box::new(AbiContext {
        retains: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        registrations: AtomicUsize::new(0),
        unregistrations: AtomicUsize::new(0),
        ready: AtomicBool::new(false),
        callback: Mutex::new(None),
    });
    let raw = NonNull::new(Box::into_raw(context).cast::<c_void>()).unwrap();
    let (id, epoch) = ids(18);
    let mut completion = unsafe { ErasedCompletion::from_raw(raw, &ABI_VTABLE, id, epoch) };
    assert!(Pin::new(&mut completion)
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending());
    let context = unsafe { &*raw.as_ptr().cast::<AbiContext>() };
    assert_eq!(context.retains.load(Ordering::Relaxed), 1);
    assert_eq!(context.registrations.load(Ordering::Relaxed), 1);
    drop(completion);
    assert_eq!(context.unregistrations.load(Ordering::Relaxed), 1);
    assert_eq!(context.releases.load(Ordering::Relaxed), 1);
    drop(unsafe { Box::from_raw(raw.as_ptr().cast::<AbiContext>()) });
}

struct FakeHal {
    destroyed: AtomicUsize,
}

impl RuntimeHal for FakeHal {
    type Device = ();
    type Memory = ();
    type Program = Lease;
    type Queue = Lease;
    type Event = u32;

    fn open_device(&self, _id: DeviceId) -> Result<Self::Device, RuntimeError> {
        Ok(())
    }
    fn allocate(
        &self,
        _device: &Self::Device,
        _request: MemoryRequest,
    ) -> Result<Self::Memory, RuntimeError> {
        Ok(())
    }
    fn load_program(
        &self,
        _device: &Self::Device,
        _artifact: &ArtifactEnvelope<'_>,
    ) -> Result<Self::Program, RuntimeError> {
        Ok(Lease)
    }
    fn create_queue(&self, _device: &Self::Device) -> Result<Self::Queue, RuntimeError> {
        Ok(Lease)
    }
    fn submit(
        &self,
        _queue: &Self::Queue,
        _program: &Self::Program,
        bindings: &SubmissionBindings<'_>,
        _deadline: Deadline,
    ) -> Result<Self::Event, RuntimeError> {
        assert_eq!(bindings.entries().len(), 1);
        Ok(1)
    }
    fn wait(&self, _event: &Self::Event, _deadline: Deadline) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn map<'a>(&self, _memory: &'a mut Self::Memory) -> Result<&'a mut [u8], RuntimeError> {
        Ok(&mut [])
    }
    fn unmap(&self, _memory: &mut Self::Memory) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn destroy_event(&self, _event: Self::Event) -> Result<(), RuntimeError> {
        self.destroyed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn destroy_queue(&self, _queue: Self::Queue) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn unload_program(&self, _program: Self::Program) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn free_memory(&self, _memory: Self::Memory) -> Result<(), RuntimeError> {
        Ok(())
    }
    fn close_device(&self, _device: Self::Device) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[test]
fn synchronous_runtime_hal_adapter_preserves_behavior_and_returns_future() {
    let hal = Arc::new(FakeHal {
        destroyed: AtomicUsize::new(0),
    });
    let adapter = RuntimeHalAdapter::<FakeHal, Lease, Lease>::new(Arc::clone(&hal));
    let (id, epoch) = ids(19);
    let allocation = Arc::new(Lease);
    let binding = BindingLease::new(
        Binding {
            slot: 0,
            memory: MemoryId::new(1).unwrap(),
            offset: 0,
            bytes: 4,
            access: AccessMode::Read,
        },
        allocation,
    );
    let operation = OwnedOperation::new(
        id,
        epoch,
        Arc::new(Lease),
        Arc::new(Lease),
        vec![binding].into_boxed_slice(),
        Lease,
        Deadline::Infinite,
    );
    let completion = adapter.submit(operation).unwrap();
    assert!(completion.try_complete().unwrap().is_ok());
    assert_eq!(hal.destroyed.load(Ordering::Relaxed), 1);
}
