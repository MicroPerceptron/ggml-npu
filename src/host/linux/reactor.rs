//! Instance-local epoll reactor for genuine pollable provider wake sources.
//!
//! A device descriptor is not assumed to represent job completion. Providers register only source
//! descriptors whose documented semantics are actual completion/progress notifications.

use alloc::sync::Arc;
use alloc::vec::Vec;

use rustix::event::{epoll, eventfd, EventfdFlags};
use rustix::fd::{AsFd, OwnedFd};
use rustix::io::{dup, read, write};
use rustix::time::Timespec;

use crate::host::async_services::ReactorFactory;
use crate::runtime::async_runtime::completion::SpinMutex;
use crate::runtime::async_runtime::{
    ProgressBudget, ProgressDriver, ProgressReport, WaitError, WakeSink, WakeSourceRegistrar,
};

use super::ERROR_DOMAIN;

fn reactor_error(error: rustix::io::Errno) -> WaitError {
    WaitError::HostFailure {
        domain: ERROR_DOMAIN,
        code: i64::from(error.raw_os_error()),
    }
}

pub(crate) struct LinuxWakeHandle {
    fd: OwnedFd,
}

impl LinuxWakeHandle {
    pub(crate) fn new() -> Result<Self, WaitError> {
        eventfd(
            0,
            EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK | EventfdFlags::SEMAPHORE,
        )
        .map(|fd| Self { fd })
        .map_err(reactor_error)
    }

    pub(crate) fn signal(&self) -> Result<(), WaitError> {
        write(&self.fd, &1u64.to_ne_bytes())
            .map(|_| ())
            .map_err(reactor_error)
    }

    pub(crate) fn drain(&self) -> Result<usize, WaitError> {
        let mut bytes = [0u8; 8];
        let mut count = 0;
        loop {
            match read(&self.fd, &mut bytes) {
                Ok(8) => count += 1,
                Ok(_) => {
                    return Err(WaitError::HostFailure {
                        domain: ERROR_DOMAIN,
                        code: 5,
                    });
                }
                Err(error) if error == rustix::io::Errno::AGAIN => return Ok(count),
                Err(error) => return Err(reactor_error(error)),
            }
        }
    }
}

struct Registration {
    token: u64,
    source: OwnedFd,
    sink: Arc<dyn WakeSink>,
}

pub(crate) struct LinuxReactor {
    epoll: OwnedFd,
    next_token: core::sync::atomic::AtomicU64,
    registrations: SpinMutex<Vec<Registration>>,
}

impl LinuxReactor {
    fn new() -> Result<Self, WaitError> {
        let epoll = epoll::create(epoll::CreateFlags::CLOEXEC).map_err(reactor_error)?;
        Ok(Self {
            epoll,
            next_token: core::sync::atomic::AtomicU64::new(1),
            registrations: SpinMutex::new(Vec::new()),
        })
    }
}

impl WakeSourceRegistrar for LinuxReactor {
    type Source = LinuxWakeHandle;
    type Registration = u64;

    fn register(
        &self,
        source: &Self::Source,
        sink: Arc<dyn WakeSink>,
    ) -> Result<Self::Registration, WaitError> {
        let owned = dup(source.fd.as_fd()).map_err(reactor_error)?;
        let token = self
            .next_token
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        epoll::add(
            &self.epoll,
            &owned,
            epoll::EventData::new_u64(token),
            epoll::EventFlags::IN | epoll::EventFlags::ERR | epoll::EventFlags::HUP,
        )
        .map_err(reactor_error)?;
        self.registrations.lock().push(Registration {
            token,
            source: owned,
            sink,
        });
        Ok(token)
    }

    fn unregister(&self, registration: Self::Registration) -> Result<(), WaitError> {
        let registration = {
            let mut registrations = self.registrations.lock();
            registrations
                .iter()
                .position(|candidate| candidate.token == registration)
                .map(|index| registrations.swap_remove(index))
        };
        if let Some(registration) = registration {
            epoll::delete(&self.epoll, &registration.source).map_err(reactor_error)?;
        }
        Ok(())
    }
}

impl ProgressDriver for LinuxReactor {
    fn progress(&self, budget: ProgressBudget) -> Result<ProgressReport, WaitError> {
        let capacity = budget.max_work.clamp(1, 256);
        let mut storage = [core::mem::MaybeUninit::<epoll::Event>::uninit(); 256];
        let timeout = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let (events, _) = epoll::wait(&self.epoll, &mut storage[..capacity], Some(&timeout))
            .map_err(reactor_error)?;
        let sinks = {
            let registrations = self.registrations.lock();
            events
                .iter()
                .filter_map(|event| {
                    let token = event.data.u64();
                    registrations
                        .iter()
                        .find(|registration| registration.token == token)
                        .map(|registration| Arc::clone(&registration.sink))
                })
                .collect::<Vec<_>>()
        };
        for sink in &sinks {
            sink.wake();
        }
        Ok(ProgressReport {
            work_done: sinks.len(),
            may_have_more: events.len() == capacity,
        })
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxReactorFactory;

impl ReactorFactory for LinuxReactorFactory {
    type Reactor = LinuxReactor;

    fn create(&self) -> Result<Self::Reactor, WaitError> {
        LinuxReactor::new()
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Default)]
    struct CountingSink(AtomicUsize);

    impl WakeSink for CountingSink {
        fn wake(&self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn epoll_reactor_registers_wakes_and_unregisters() {
        let reactor = LinuxReactor::new().unwrap();
        let source = LinuxWakeHandle::new().unwrap();
        let sink = Arc::new(CountingSink::default());
        let token = reactor
            .register(&source, sink.clone() as Arc<dyn WakeSink>)
            .unwrap();
        source.signal().unwrap();
        assert_eq!(
            reactor
                .progress(ProgressBudget { max_work: 8 })
                .unwrap()
                .work_done,
            1
        );
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
        assert_eq!(source.drain().unwrap(), 1);
        reactor.unregister(token).unwrap();
        source.signal().unwrap();
        assert_eq!(
            reactor
                .progress(ProgressBudget { max_work: 8 })
                .unwrap()
                .work_done,
            0
        );
    }
}
