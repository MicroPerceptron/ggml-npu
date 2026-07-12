use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::{Arc, Weak};
use alloc::task::Wake;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};

use super::completion::SpinMutex;
use super::{
    BlockingJob, Completion, CompletionControl, CompletionError, DeviceEpoch, OperationId,
    ProgressBudget, SubmitError, WaitChange, WaitContext, WakeCount,
};
use crate::runtime::Deadline;

const BLOCKING_DOMAIN: u32 = u32::from_be_bytes(*b"BLCK");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunReport {
    pub polled: usize,
    pub completed: usize,
    pub may_have_more: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    Scheduled = 0,
    Waiting = 1,
    Complete = 2,
    Cancelled = 3,
}

struct Task {
    future: SpinMutex<Option<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>>,
    executor: Weak<ExecutorInner>,
    scheduled: AtomicBool,
    state: AtomicU8,
}

impl Task {
    fn schedule(self: &Arc<Self>) {
        if self.state.load(Ordering::Acquire) >= TaskState::Complete as u8 {
            return;
        }
        if self
            .scheduled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            if let Some(executor) = self.executor.upgrade() {
                self.state
                    .store(TaskState::Scheduled as u8, Ordering::Release);
                executor.queue.lock().push_back(Arc::clone(self));
            }
        }
    }

    fn cancel(&self) -> bool {
        let previous = self
            .state
            .swap(TaskState::Cancelled as u8, Ordering::AcqRel);
        if previous < TaskState::Complete as u8 {
            self.future.lock().take();
            if let Some(executor) = self.executor.upgrade() {
                executor.active.fetch_sub(1, Ordering::AcqRel);
            }
            true
        } else {
            false
        }
    }
}

impl Wake for Task {
    fn wake(self: Arc<Self>) {
        self.schedule();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.schedule();
    }
}

struct ExecutorInner {
    capacity: usize,
    active: AtomicUsize,
    shutdown: AtomicBool,
    queue: SpinMutex<VecDeque<Arc<Task>>>,
    tasks: SpinMutex<Vec<Weak<Task>>>,
}

/// Bounded caller-driven task executor. It creates no worker threads.
#[derive(Clone)]
pub struct CallerExecutor {
    inner: Arc<ExecutorInner>,
}

impl CallerExecutor {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(ExecutorInner {
                capacity,
                active: AtomicUsize::new(0),
                shutdown: AtomicBool::new(false),
                queue: SpinMutex::new(VecDeque::new()),
                tasks: SpinMutex::new(Vec::new()),
            }),
        }
    }

    pub fn spawn<F>(&self, future: F) -> Result<TaskHandle, SpawnError<F>>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if self.inner.shutdown.load(Ordering::Acquire) {
            return Err(SpawnError::ShuttingDown(future));
        }
        let mut active = self.inner.active.load(Ordering::Acquire);
        let admitted = loop {
            if active >= self.inner.capacity {
                break false;
            }
            match self.inner.active.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break true,
                Err(observed) => active = observed,
            }
        };
        if !admitted {
            return Err(SpawnError::AtCapacity(future));
        }
        if self.inner.shutdown.load(Ordering::Acquire) {
            self.inner.active.fetch_sub(1, Ordering::AcqRel);
            return Err(SpawnError::ShuttingDown(future));
        }

        let task = Arc::new(Task {
            future: SpinMutex::new(Some(Box::pin(future))),
            executor: Arc::downgrade(&self.inner),
            scheduled: AtomicBool::new(true),
            state: AtomicU8::new(TaskState::Scheduled as u8),
        });
        self.inner.tasks.lock().push(Arc::downgrade(&task));
        self.inner.queue.lock().push_back(Arc::clone(&task));
        Ok(TaskHandle { task })
    }

    pub fn run_until_stalled(&self, max_polls: usize) -> RunReport {
        let mut report = RunReport::default();
        while report.polled < max_polls {
            let Some(task) = self.inner.queue.lock().pop_front() else {
                break;
            };
            task.scheduled.store(false, Ordering::Release);
            if task.state.load(Ordering::Acquire) >= TaskState::Complete as u8 {
                continue;
            }
            task.state
                .store(TaskState::Waiting as u8, Ordering::Release);
            let waker = Waker::from(Arc::clone(&task));
            let mut context = Context::from_waker(&waker);
            report.polled += 1;
            let ready = {
                let mut future = task.future.lock();
                future
                    .as_mut()
                    .is_some_and(|future| future.as_mut().poll(&mut context).is_ready())
            };
            if ready {
                task.future.lock().take();
                let previous = task.state.swap(TaskState::Complete as u8, Ordering::AcqRel);
                if previous < TaskState::Complete as u8 {
                    self.inner.active.fetch_sub(1, Ordering::AcqRel);
                    report.completed += 1;
                }
            }
        }
        report.may_have_more = !self.inner.queue.lock().is_empty();
        report
    }

    pub fn active_tasks(&self) -> usize {
        self.inner.active.load(Ordering::Acquire)
    }

    pub fn shutdown(&self) -> usize {
        self.inner.shutdown.store(true, Ordering::Release);
        let tasks = {
            let mut tracked = self.inner.tasks.lock();
            let tasks = tracked.iter().filter_map(Weak::upgrade).collect::<Vec<_>>();
            tracked.clear();
            tasks
        };
        let cancelled = tasks.into_iter().filter(|task| task.cancel()).count();
        self.inner.queue.lock().clear();
        cancelled
    }

    /// Drive a borrowed future, spawned tasks, and injected reactor progress to a result.
    pub fn block_on<F>(
        &self,
        future: F,
        wait: &WaitContext,
        deadline: Deadline,
    ) -> Result<F::Output, BlockOnError>
    where
        F: Future,
    {
        let signal = Arc::new(BlockSignal {
            generation: core::sync::atomic::AtomicU32::new(0),
            wait: Arc::clone(&wait.atomic_wait),
        });
        let waker = Waker::from(Arc::clone(&signal));
        let mut context = Context::from_waker(&waker);
        let mut future = core::pin::pin!(future);

        loop {
            let expected = signal.generation.load(Ordering::Acquire);
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return Ok(output);
            }
            let tasks = self.run_until_stalled(1024);
            let progress = wait
                .progress
                .progress(ProgressBudget::DEFAULT)
                .map_err(BlockOnError::Wait)?;
            if tasks.polled != 0 || progress.work_done != 0 {
                continue;
            }
            if let Deadline::MonotonicNs(value) = deadline {
                if wait.clock.now_ns().map_err(BlockOnError::Wait)? >= value {
                    return Err(BlockOnError::TimedOut);
                }
            }
            if matches!(
                wait.atomic_wait
                    .wait_u32(&signal.generation, expected, deadline)
                    .map_err(BlockOnError::Wait)?,
                WaitChange::TimedOut
            ) {
                return Err(BlockOnError::TimedOut);
            }
        }
    }
}

impl Drop for CallerExecutor {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.shutdown();
        }
    }
}

pub enum SpawnError<F> {
    AtCapacity(F),
    ShuttingDown(F),
}

pub struct TaskHandle {
    task: Arc<Task>,
}

impl TaskHandle {
    pub fn state(&self) -> TaskState {
        match self.task.state.load(Ordering::Acquire) {
            0 => TaskState::Scheduled,
            1 => TaskState::Waiting,
            2 => TaskState::Complete,
            _ => TaskState::Cancelled,
        }
    }

    pub fn cancel(&self) -> bool {
        self.task.cancel()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockOnError {
    TimedOut,
    Wait(super::WaitError),
}

struct BlockSignal {
    generation: core::sync::atomic::AtomicU32,
    wait: Arc<dyn super::AtomicWait>,
}

impl Wake for BlockSignal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.generation.fetch_add(1, Ordering::Release);
        let _ = self.wait.wake_u32(&self.generation, WakeCount::All);
    }
}

pub struct BlockingRequest {
    pub operation_id: OperationId,
    pub device_epoch: DeviceEpoch,
    pub queue_sequence: u64,
    pub job: BlockingJob,
}

pub trait BlockingExecutorPort: Send + Sync {
    fn try_spawn(
        &self,
        request: BlockingRequest,
    ) -> Result<Completion, (SubmitError, BlockingRequest)>;
    fn shutdown(&self);
}

struct QueuedBlocking {
    job: BlockingJob,
    control: CompletionControl,
}

/// Correct bounded fallback when the host does not inject autonomous workers.
///
/// `run_one` must be called outside all `Future::poll` paths.
pub struct CallerDrivenBlockingExecutor {
    capacity: usize,
    shutdown: AtomicBool,
    queue: SpinMutex<VecDeque<QueuedBlocking>>,
}

impl CallerDrivenBlockingExecutor {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            shutdown: AtomicBool::new(false),
            queue: SpinMutex::new(VecDeque::new()),
        }
    }

    pub fn run_one(&self) -> bool {
        let Some(queued) = self.queue.lock().pop_front() else {
            return false;
        };
        match (queued.job)() {
            Ok(info) => {
                queued.control.set_progress(info.progress);
                queued.control.succeed(true);
            }
            Err(error) => {
                queued.control.fail(error, true);
            }
        }
        true
    }

    pub fn queued(&self) -> usize {
        self.queue.lock().len()
    }
}

impl BlockingExecutorPort for CallerDrivenBlockingExecutor {
    fn try_spawn(
        &self,
        request: BlockingRequest,
    ) -> Result<Completion, (SubmitError, BlockingRequest)> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err((SubmitError::RuntimeShuttingDown, request));
        }
        let mut queue = self.queue.lock();
        if queue.len() >= self.capacity {
            return Err((SubmitError::ResourceLimit, request));
        }
        let (completion, control) = CompletionControl::pending(
            request.operation_id,
            request.device_epoch,
            request.queue_sequence,
            (),
            false,
            None,
        );
        queue.push_back(QueuedBlocking {
            job: request.job,
            control,
        });
        Ok(completion)
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let queued = self.queue.lock().drain(..).collect::<Vec<_>>();
        for queued in queued {
            queued.control.fail(
                CompletionError::ExecutionFailed {
                    domain: BLOCKING_DOMAIN,
                    code: 1,
                },
                true,
            );
        }
    }
}
