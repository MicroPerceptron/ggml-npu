//! Deterministic host-independent clock, wait, reactor, and wake sources.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use super::completion::SpinMutex;
use super::{
    AtomicWait, CompletionControl, CompletionError, DeviceEpoch, MonotonicClock, ProgressBudget,
    ProgressDriver, ProgressReport, WaitChange, WaitContext, WaitError, WakeCount, WakeSink,
    WakeSourceRegistrar,
};
use crate::runtime::Deadline;

#[derive(Default)]
pub struct VirtualClock {
    now_ns: AtomicU64,
}

impl VirtualClock {
    pub const fn new(now_ns: u64) -> Self {
        Self {
            now_ns: AtomicU64::new(now_ns),
        }
    }

    pub fn set(&self, now_ns: u64) {
        self.now_ns.store(now_ns, Ordering::Release);
    }

    pub fn advance(&self, nanoseconds: u64) -> u64 {
        self.now_ns.fetch_add(nanoseconds, Ordering::AcqRel) + nanoseconds
    }
}

impl MonotonicClock for VirtualClock {
    fn now_ns(&self) -> Result<u64, WaitError> {
        Ok(self.now_ns.load(Ordering::Acquire))
    }
}

pub struct DeterministicAtomicWait {
    clock: Arc<VirtualClock>,
    spurious: AtomicUsize,
    wakes: AtomicUsize,
}

impl DeterministicAtomicWait {
    pub fn new(clock: Arc<VirtualClock>) -> Self {
        Self {
            clock,
            spurious: AtomicUsize::new(0),
            wakes: AtomicUsize::new(0),
        }
    }

    pub fn inject_spurious(&self, count: usize) {
        self.spurious.fetch_add(count, Ordering::Relaxed);
    }

    pub fn wake_count(&self) -> usize {
        self.wakes.load(Ordering::Relaxed)
    }
}

impl AtomicWait for DeterministicAtomicWait {
    fn wait_u32(
        &self,
        state: &AtomicU32,
        expected: u32,
        deadline: Deadline,
    ) -> Result<WaitChange, WaitError> {
        if state.load(Ordering::Acquire) != expected {
            return Ok(WaitChange::Changed);
        }
        let mut spurious = self.spurious.load(Ordering::Acquire);
        while spurious != 0 {
            match self.spurious.compare_exchange_weak(
                spurious,
                spurious - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(WaitChange::Spurious),
                Err(observed) => spurious = observed,
            }
        }
        match deadline {
            Deadline::Infinite => Ok(WaitChange::Spurious),
            Deadline::MonotonicNs(deadline) => {
                if self.clock.now_ns()? >= deadline {
                    Ok(WaitChange::TimedOut)
                } else {
                    self.clock.advance(1);
                    Ok(WaitChange::Spurious)
                }
            }
        }
    }

    fn wake_u32(&self, _state: &AtomicU32, _count: WakeCount) -> Result<(), WaitError> {
        self.wakes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ScriptedOutcome {
    Success {
        reclaim_safe: bool,
    },
    Failure {
        error: CompletionError,
        reclaim_safe: bool,
    },
    Cancelled {
        may_have_executed: bool,
        reclaim_safe: bool,
    },
    DeviceLost {
        reason: u32,
        reclaim_safe: bool,
    },
}

struct Script {
    at_ns: u64,
    control: CompletionControl,
    outcome: ScriptedOutcome,
}

pub struct MockReactor {
    clock: Arc<VirtualClock>,
    quantum_ns: u64,
    scripts: SpinMutex<Vec<Script>>,
}

impl MockReactor {
    pub fn new(clock: Arc<VirtualClock>, quantum_ns: u64) -> Self {
        Self {
            clock,
            quantum_ns: quantum_ns.max(1),
            scripts: SpinMutex::new(Vec::new()),
        }
    }

    pub fn schedule(&self, at_ns: u64, control: CompletionControl, outcome: ScriptedOutcome) {
        self.scripts.lock().push(Script {
            at_ns,
            control,
            outcome,
        });
    }

    pub fn pending_scripts(&self) -> usize {
        self.scripts.lock().len()
    }
}

impl ProgressDriver for MockReactor {
    fn progress(&self, budget: ProgressBudget) -> Result<ProgressReport, WaitError> {
        self.clock.advance(self.quantum_ns);
        let now = self.clock.now_ns()?;
        let mut due = Vec::new();
        {
            let mut scripts = self.scripts.lock();
            let mut index = 0;
            while index < scripts.len() && due.len() < budget.max_work {
                if scripts[index].at_ns <= now {
                    due.push(scripts.swap_remove(index));
                } else {
                    index += 1;
                }
            }
        }
        for script in &due {
            match script.outcome {
                ScriptedOutcome::Success { reclaim_safe } => {
                    script.control.succeed(reclaim_safe);
                }
                ScriptedOutcome::Failure {
                    error,
                    reclaim_safe,
                } => {
                    script.control.fail(error, reclaim_safe);
                }
                ScriptedOutcome::Cancelled {
                    may_have_executed,
                    reclaim_safe,
                } => {
                    script
                        .control
                        .acknowledge_cancel(may_have_executed, reclaim_safe);
                }
                ScriptedOutcome::DeviceLost {
                    reason,
                    reclaim_safe,
                } => {
                    script.control.mark_device_lost(reason, reclaim_safe);
                }
            }
        }
        Ok(ProgressReport {
            work_done: due.len(),
            may_have_more: !self.scripts.lock().is_empty(),
        })
    }
}

pub fn deterministic_wait_context(
    clock: Arc<VirtualClock>,
    wait: Arc<DeterministicAtomicWait>,
    reactor: Arc<MockReactor>,
) -> WaitContext {
    WaitContext::new(wait, clock, reactor)
}

pub struct MockWakeSource {
    next: AtomicU64,
    registrations: SpinMutex<Vec<(u64, Arc<dyn WakeSink>)>>,
}

impl Default for MockWakeSource {
    fn default() -> Self {
        Self {
            next: AtomicU64::new(1),
            registrations: SpinMutex::new(Vec::new()),
        }
    }
}

impl MockWakeSource {
    pub fn signal(&self) {
        let sinks = self
            .registrations
            .lock()
            .iter()
            .map(|(_, sink)| Arc::clone(sink))
            .collect::<Vec<_>>();
        for sink in sinks {
            sink.wake();
        }
    }

    pub fn registrations(&self) -> usize {
        self.registrations.lock().len()
    }
}

impl WakeSourceRegistrar for MockWakeSource {
    type Source = ();
    type Registration = u64;

    fn register(
        &self,
        _source: &Self::Source,
        sink: Arc<dyn WakeSink>,
    ) -> Result<Self::Registration, WaitError> {
        let token = self.next.fetch_add(1, Ordering::Relaxed);
        self.registrations.lock().push((token, sink));
        Ok(token)
    }

    fn unregister(&self, registration: Self::Registration) -> Result<(), WaitError> {
        self.registrations
            .lock()
            .retain(|(token, _)| *token != registration);
        Ok(())
    }
}

pub fn device_lost(epoch: DeviceEpoch, reason: u32) -> CompletionError {
    CompletionError::DeviceLost { epoch, reason }
}
