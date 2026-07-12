//! Linux private futex implementation of the shared atomic wait/wake contract.

use core::sync::atomic::{AtomicU32, Ordering};

use rustix::thread::futex::{self, Flags, Timespec};
use rustix::time::{clock_gettime, ClockId};

use crate::runtime::async_runtime::{AtomicWait, WaitChange, WaitError, WakeCount};
use crate::runtime::Deadline;

use super::ERROR_DOMAIN;

#[derive(Clone, Copy, Default)]
pub(crate) struct LinuxAtomicWait;

fn wait_error(error: rustix::io::Errno) -> WaitError {
    WaitError::HostFailure {
        domain: ERROR_DOMAIN,
        code: i64::from(error.raw_os_error()),
    }
}

fn remaining(deadline: Deadline) -> Result<Option<Timespec>, WaitError> {
    let Deadline::MonotonicNs(deadline) = deadline else {
        return Ok(None);
    };
    let now = clock_gettime(ClockId::Monotonic);
    let seconds = u64::try_from(now.tv_sec).map_err(|_| WaitError::HostFailure {
        domain: ERROR_DOMAIN,
        code: 75,
    })?;
    let nanos = u64::try_from(now.tv_nsec).map_err(|_| WaitError::HostFailure {
        domain: ERROR_DOMAIN,
        code: 75,
    })?;
    let now = seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or(WaitError::HostFailure {
            domain: ERROR_DOMAIN,
            code: 75,
        })?;
    let remaining = deadline.saturating_sub(now);
    Ok(Some(Timespec {
        tv_sec: (remaining / 1_000_000_000) as i64,
        tv_nsec: (remaining % 1_000_000_000) as i64,
    }))
}

impl AtomicWait for LinuxAtomicWait {
    fn wait_u32(
        &self,
        state: &AtomicU32,
        expected: u32,
        deadline: Deadline,
    ) -> Result<WaitChange, WaitError> {
        if state.load(Ordering::Acquire) != expected {
            return Ok(WaitChange::Changed);
        }
        let timeout = remaining(deadline)?;
        match futex::wait(state, Flags::PRIVATE, expected, timeout.as_ref()) {
            Ok(()) => Ok(if state.load(Ordering::Acquire) == expected {
                WaitChange::Spurious
            } else {
                WaitChange::Changed
            }),
            Err(error) if error == rustix::io::Errno::AGAIN => Ok(WaitChange::Changed),
            Err(error) if error == rustix::io::Errno::INTR => Ok(WaitChange::Spurious),
            Err(error) if error == rustix::io::Errno::TIMEDOUT => Ok(WaitChange::TimedOut),
            Err(error) => Err(wait_error(error)),
        }
    }

    fn wake_u32(&self, state: &AtomicU32, count: WakeCount) -> Result<(), WaitError> {
        let count = match count {
            WakeCount::One => 1,
            WakeCount::All => u32::MAX,
        };
        futex::wake(state, Flags::PRIVATE, count)
            .map(|_| ())
            .map_err(wait_error)
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use core::sync::atomic::AtomicU32;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn futex_wait_wakes_without_hardware() {
        let word = Arc::new(AtomicU32::new(0));
        let waiter_word = Arc::clone(&word);
        let waiter = thread::spawn(move || {
            LinuxAtomicWait
                .wait_u32(&waiter_word, 0, Deadline::Infinite)
                .unwrap()
        });
        thread::sleep(Duration::from_millis(5));
        word.store(1, Ordering::Release);
        LinuxAtomicWait.wake_u32(&word, WakeCount::All).unwrap();
        assert_eq!(waiter.join().unwrap(), WaitChange::Changed);
    }

    #[test]
    fn futex_absolute_deadline_times_out() {
        let word = AtomicU32::new(0);
        let now = clock_gettime(ClockId::Monotonic);
        let now_ns = (now.tv_sec as u64) * 1_000_000_000 + now.tv_nsec as u64;
        assert_eq!(
            LinuxAtomicWait
                .wait_u32(&word, 0, Deadline::MonotonicNs(now_ns))
                .unwrap(),
            WaitChange::TimedOut
        );
    }
}
