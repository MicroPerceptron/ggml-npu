use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use alloc::task::Wake;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::fmt;
use core::future::Future;
use core::hint::spin_loop;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use core::task::{Context, Poll, Waker};

use super::operation::{DeviceEpoch, OperationId, RetainedResources};
use super::{AtomicWait, MonotonicClock, ProgressBudget, ProgressDriver, WaitChange, WakeCount};
use crate::runtime::Deadline;

const WAIT_DOMAIN: u32 = u32::from_be_bytes(*b"WAIT");

pub(crate) struct SpinMutex<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

impl<T> SpinMutex<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    pub(crate) fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.locked.load(Ordering::Relaxed) {
                spin_loop();
            }
        }
        SpinGuard { mutex: self }
    }
}

unsafe impl<T: Send> Send for SpinMutex<T> {}
unsafe impl<T: Send> Sync for SpinMutex<T> {}

pub(crate) struct SpinGuard<'a, T> {
    mutex: &'a SpinMutex<T>,
}

impl<T> core::ops::Deref for SpinGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: the guard owns the mutex until drop.
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> core::ops::DerefMut for SpinGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the guard owns the mutex until drop.
        unsafe { &mut *self.mutex.value.get() }
    }
}

impl<T> Drop for SpinGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletionInfo {
    pub operation_id: OperationId,
    pub device_epoch: DeviceEpoch,
    pub queue_sequence: u64,
    pub progress: Progress,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionError {
    ExecutionFailed { domain: u32, code: i64 },
    Cancelled { may_have_executed: bool },
    DeviceLost { epoch: DeviceEpoch, reason: u32 },
}

pub type CompletionResult = Result<CompletionInfo, CompletionError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitError {
    HostFailure { domain: u32, code: i64 },
    RuntimeShuttingDown,
}

impl fmt::Display for WaitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitStatus<T> {
    Ready(T),
    TimedOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitMode {
    Any,
    All,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WaitManyOutcome {
    Ready(Vec<usize>),
    TimedOut,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationState {
    Unsupported,
    Cancellable,
    Requested,
    Acknowledged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationRequest {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
    Unsupported,
}

/// Provider-facing object-safe source beneath the typed future.
pub trait CompletionSource: Send + Sync {
    fn operation_id(&self) -> OperationId;
    fn device_epoch(&self) -> DeviceEpoch;
    fn try_complete(&self) -> Option<CompletionResult>;
    fn register_waker(&self, observer: u64, waker: &Waker);
    fn unregister_waker(&self, observer: u64);
    fn request_cancel(&self) -> CancellationRequest;

    fn poll_complete(&self, observer: u64, context: &mut Context<'_>) -> Poll<CompletionResult> {
        if let Some(result) = self.try_complete() {
            return Poll::Ready(result);
        }
        self.register_waker(observer, context.waker());
        match self.try_complete() {
            Some(result) => {
                self.unregister_waker(observer);
                Poll::Ready(result)
            }
            None => Poll::Pending,
        }
    }
}

static NEXT_OBSERVER: AtomicU64 = AtomicU64::new(1);

fn next_observer() -> u64 {
    let id = NEXT_OBSERVER.fetch_add(1, Ordering::Relaxed);
    if id == 0 {
        NEXT_OBSERVER.fetch_add(1, Ordering::Relaxed)
    } else {
        id
    }
}

/// Cloneable native Rust completion future. Every clone owns a distinct observer registration.
pub struct Completion {
    source: Arc<dyn CompletionSource>,
    observer: u64,
}

impl Completion {
    pub fn from_source(source: Arc<dyn CompletionSource>) -> Self {
        Self {
            source,
            observer: next_observer(),
        }
    }

    pub fn try_complete(&self) -> Option<CompletionResult> {
        self.source.try_complete()
    }

    pub fn request_cancel(&self) -> CancellationRequest {
        self.source.request_cancel()
    }

    pub fn operation_id(&self) -> OperationId {
        self.source.operation_id()
    }

    pub fn device_epoch(&self) -> DeviceEpoch {
        self.source.device_epoch()
    }

    pub fn wait(
        &self,
        context: &WaitContext,
        deadline: Deadline,
    ) -> Result<WaitStatus<CompletionResult>, WaitError> {
        match wait_many(
            core::slice::from_ref(self),
            WaitMode::All,
            context,
            deadline,
        )? {
            WaitManyOutcome::Ready(_) => Ok(WaitStatus::Ready(self.try_complete().ok_or(
                WaitError::HostFailure {
                    domain: WAIT_DOMAIN,
                    code: 1,
                },
            )?)),
            WaitManyOutcome::TimedOut => Ok(WaitStatus::TimedOut),
        }
    }
}

impl Clone for Completion {
    fn clone(&self) -> Self {
        Self {
            source: Arc::clone(&self.source),
            observer: next_observer(),
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        self.source.unregister_waker(self.observer);
    }
}

impl Future for Completion {
    type Output = CompletionResult;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.source.poll_complete(self.observer, context)
    }
}

pub struct WaitContext {
    pub atomic_wait: Arc<dyn AtomicWait>,
    pub clock: Arc<dyn MonotonicClock>,
    pub progress: Arc<dyn ProgressDriver>,
}

impl WaitContext {
    pub fn new(
        atomic_wait: Arc<dyn AtomicWait>,
        clock: Arc<dyn MonotonicClock>,
        progress: Arc<dyn ProgressDriver>,
    ) -> Self {
        Self {
            atomic_wait,
            clock,
            progress,
        }
    }
}

struct WaitSignal {
    generation: AtomicU32,
    atomic_wait: Arc<dyn AtomicWait>,
}

impl Wake for WaitSignal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.generation.fetch_add(1, Ordering::Release);
        let _ = self.atomic_wait.wake_u32(&self.generation, WakeCount::All);
    }
}

fn deadline_expired(context: &WaitContext, deadline: Deadline) -> Result<bool, WaitError> {
    match deadline {
        Deadline::Infinite => Ok(false),
        Deadline::MonotonicNs(value) => Ok(context.clock.now_ns()? >= value),
    }
}

/// Blocking wait over exact completion snapshots. Spurious wakes and timeout races are rechecked.
pub fn wait_many(
    completions: &[Completion],
    mode: WaitMode,
    context: &WaitContext,
    deadline: Deadline,
) -> Result<WaitManyOutcome, WaitError> {
    if completions.is_empty() {
        return Ok(WaitManyOutcome::Ready(Vec::new()));
    }

    let signal = Arc::new(WaitSignal {
        generation: AtomicU32::new(0),
        atomic_wait: Arc::clone(&context.atomic_wait),
    });
    let waker = Waker::from(Arc::clone(&signal));
    let mut task_context = Context::from_waker(&waker);
    let mut observers: Vec<_> = completions.to_vec();

    loop {
        let expected = signal.generation.load(Ordering::Acquire);
        let mut ready = Vec::new();
        for (index, completion) in observers.iter_mut().enumerate() {
            if Pin::new(completion).poll(&mut task_context).is_ready() {
                ready.push(index);
            }
        }
        let satisfied = match mode {
            WaitMode::Any => !ready.is_empty(),
            WaitMode::All => ready.len() == observers.len(),
        };
        if satisfied {
            return Ok(WaitManyOutcome::Ready(ready));
        }
        if deadline_expired(context, deadline)? {
            return Ok(WaitManyOutcome::TimedOut);
        }

        let report = context.progress.progress(ProgressBudget::DEFAULT)?;
        if report.work_done != 0 {
            continue;
        }

        match context
            .atomic_wait
            .wait_u32(&signal.generation, expected, deadline)?
        {
            WaitChange::Changed | WaitChange::Spurious => {}
            WaitChange::TimedOut => {
                // Completion may have won the timeout race; one final poll decides.
                let ready = observers
                    .iter()
                    .enumerate()
                    .filter_map(|(index, completion)| {
                        completion.try_complete().is_some().then_some(index)
                    })
                    .collect::<Vec<_>>();
                let satisfied = match mode {
                    WaitMode::Any => !ready.is_empty(),
                    WaitMode::All => ready.len() == observers.len(),
                };
                return Ok(if satisfied {
                    WaitManyOutcome::Ready(ready)
                } else {
                    WaitManyOutcome::TimedOut
                });
            }
        }
    }
}

struct Waiter {
    observer: u64,
    waker: Waker,
}

struct CoreState {
    terminal: Option<CompletionResult>,
    reclaim_safe: bool,
    cancellation: CancellationState,
    progress: Progress,
    resources: Option<Box<dyn RetainedResources>>,
    waiters: Vec<Waiter>,
}

struct CompletionCore {
    operation_id: OperationId,
    epoch: DeviceEpoch,
    queue_sequence: u64,
    state: SpinMutex<CoreState>,
    registry: Option<Weak<InFlightRegistry>>,
}

impl CompletionCore {
    fn finish(&self, result: CompletionResult, reclaim_safe: bool) -> bool {
        let (wakers, resources, reclaim_safe) = {
            let mut state = self.state.lock();
            if state.terminal.is_some() {
                return false;
            }
            if matches!(result, Err(CompletionError::Cancelled { .. })) {
                state.cancellation = CancellationState::Acknowledged;
            }
            state.terminal = Some(result);
            state.reclaim_safe |= reclaim_safe;
            let wakers = state
                .waiters
                .drain(..)
                .map(|waiter| waiter.waker)
                .collect::<Vec<Waker>>();
            let reclaim_safe = state.reclaim_safe;
            let resources = reclaim_safe.then(|| state.resources.take()).flatten();
            (wakers, resources, reclaim_safe)
        };
        for waker in wakers {
            waker.wake();
        }
        drop(resources);
        if reclaim_safe {
            self.remove_from_registry();
        }
        true
    }

    fn remove_from_registry(&self) {
        if let Some(registry) = self.registry.as_ref().and_then(Weak::upgrade) {
            registry.remove(self.operation_id);
        }
    }

    fn mark_reclaim_safe(&self) {
        let resources = {
            let mut state = self.state.lock();
            state.reclaim_safe = true;
            if state.terminal.is_some() {
                state.resources.take()
            } else {
                None
            }
        };
        drop(resources);
        if self.state.lock().terminal.is_some() {
            self.remove_from_registry();
        }
    }
}

impl CompletionSource for CompletionCore {
    fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    fn device_epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    fn try_complete(&self) -> Option<CompletionResult> {
        self.state.lock().terminal
    }

    fn register_waker(&self, observer: u64, waker: &Waker) {
        let mut state = self.state.lock();
        if state.terminal.is_some() {
            return;
        }
        if let Some(waiter) = state
            .waiters
            .iter_mut()
            .find(|waiter| waiter.observer == observer)
        {
            if !waiter.waker.will_wake(waker) {
                waiter.waker = waker.clone();
            }
        } else {
            state.waiters.push(Waiter {
                observer,
                waker: waker.clone(),
            });
        }
    }

    fn unregister_waker(&self, observer: u64) {
        let mut state = self.state.lock();
        if let Some(index) = state
            .waiters
            .iter()
            .position(|waiter| waiter.observer == observer)
        {
            state.waiters.swap_remove(index);
        }
    }

    fn request_cancel(&self) -> CancellationRequest {
        let mut state = self.state.lock();
        if state.terminal.is_some() {
            return CancellationRequest::AlreadyTerminal;
        }
        match state.cancellation {
            CancellationState::Unsupported => CancellationRequest::Unsupported,
            CancellationState::Cancellable => {
                state.cancellation = CancellationState::Requested;
                CancellationRequest::Requested
            }
            CancellationState::Requested | CancellationState::Acknowledged => {
                CancellationRequest::AlreadyRequested
            }
        }
    }
}

/// Provider handle for state transitions. No provider callback runs inside `Future::poll`.
#[derive(Clone)]
pub struct CompletionControl {
    core: Arc<CompletionCore>,
}

impl CompletionControl {
    #[allow(clippy::too_many_arguments)]
    pub fn pending<R>(
        operation_id: OperationId,
        epoch: DeviceEpoch,
        queue_sequence: u64,
        resources: R,
        cancellation_supported: bool,
        registry: Option<Arc<InFlightRegistry>>,
    ) -> (Completion, Self)
    where
        R: Send + Sync + 'static,
    {
        let core = Arc::new(CompletionCore {
            operation_id,
            epoch,
            queue_sequence,
            state: SpinMutex::new(CoreState {
                terminal: None,
                reclaim_safe: false,
                cancellation: if cancellation_supported {
                    CancellationState::Cancellable
                } else {
                    CancellationState::Unsupported
                },
                progress: Progress::default(),
                resources: Some(Box::new(resources)),
                waiters: Vec::new(),
            }),
            registry: registry.as_ref().map(Arc::downgrade),
        });
        if let Some(registry) = registry {
            registry.insert(Arc::clone(&core));
        }
        let completion = Completion::from_source(core.clone());
        (completion, Self { core })
    }

    pub fn ready(result: CompletionResult) -> Completion {
        let (operation_id, epoch, sequence, progress) = match result {
            Ok(info) => (
                info.operation_id,
                info.device_epoch,
                info.queue_sequence,
                info.progress,
            ),
            Err(CompletionError::DeviceLost { epoch, .. }) => (
                OperationId::new(1).expect("constant is nonzero"),
                epoch,
                0,
                Progress::default(),
            ),
            Err(_) => (
                OperationId::new(1).expect("constant is nonzero"),
                DeviceEpoch::new(1).expect("constant is nonzero"),
                0,
                Progress::default(),
            ),
        };
        let (completion, control) = Self::pending(operation_id, epoch, sequence, (), false, None);
        control.set_progress(progress);
        control.core.finish(result, true);
        completion
    }

    pub fn succeed(&self, reclaim_safe: bool) -> bool {
        let progress = self.core.state.lock().progress;
        self.core.finish(
            Ok(CompletionInfo {
                operation_id: self.core.operation_id,
                device_epoch: self.core.epoch,
                queue_sequence: self.core.queue_sequence,
                progress,
            }),
            reclaim_safe,
        )
    }

    pub fn fail(&self, error: CompletionError, reclaim_safe: bool) -> bool {
        self.core.finish(Err(error), reclaim_safe)
    }

    pub fn acknowledge_cancel(&self, may_have_executed: bool, reclaim_safe: bool) -> bool {
        self.fail(
            CompletionError::Cancelled { may_have_executed },
            reclaim_safe,
        )
    }

    pub fn mark_device_lost(&self, reason: u32, reclaim_safe: bool) -> bool {
        self.fail(
            CompletionError::DeviceLost {
                epoch: self.core.epoch,
                reason,
            },
            reclaim_safe,
        )
    }

    pub fn mark_reclaim_safe(&self) {
        self.core.mark_reclaim_safe();
    }

    pub fn set_progress(&self, progress: Progress) {
        let mut state = self.core.state.lock();
        if state.terminal.is_none() {
            state.progress = progress;
        }
    }

    pub fn cancellation_state(&self) -> CancellationState {
        self.core.state.lock().cancellation
    }

    pub fn operation_id(&self) -> OperationId {
        self.core.operation_id
    }

    pub fn device_epoch(&self) -> DeviceEpoch {
        self.core.epoch
    }
}

#[derive(Default)]
pub struct InFlightRegistry {
    entries: SpinMutex<Vec<(OperationId, Arc<CompletionCore>)>>,
}

impl InFlightRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    fn insert(&self, core: Arc<CompletionCore>) {
        self.entries.lock().push((core.operation_id, core));
    }

    fn remove(&self, operation_id: OperationId) {
        self.entries
            .lock()
            .retain(|(candidate, _)| *candidate != operation_id);
    }

    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SpinMutex<Vec<(OperationId, Arc<CompletionCore>)>> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[derive(Default)]
pub struct DeviceEpochFanout {
    completions: SpinMutex<Vec<(DeviceEpoch, Weak<CompletionCore>)>>,
}

impl DeviceEpochFanout {
    pub fn track(&self, control: &CompletionControl) {
        self.completions
            .lock()
            .push((control.core.epoch, Arc::downgrade(&control.core)));
    }

    pub fn mark_lost(&self, epoch: DeviceEpoch, reason: u32) -> usize {
        let controls = {
            let mut completions = self.completions.lock();
            let mut controls = Vec::new();
            completions.retain(|(candidate, weak)| {
                if let Some(core) = weak.upgrade() {
                    if *candidate == epoch {
                        controls.push(core);
                    }
                    true
                } else {
                    false
                }
            });
            controls
        };
        controls
            .into_iter()
            .filter(|core| core.finish(Err(CompletionError::DeviceLost { epoch, reason }), false))
            .count()
    }

    pub fn mark_epoch_reclaim_safe(&self, epoch: DeviceEpoch) {
        let completions = self.completions.lock();
        for (candidate, weak) in completions.iter() {
            if *candidate == epoch {
                if let Some(core) = weak.upgrade() {
                    core.mark_reclaim_safe();
                }
            }
        }
    }
}

impl Default for SpinMutex<Vec<(DeviceEpoch, Weak<CompletionCore>)>> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}
