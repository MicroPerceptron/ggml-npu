//! Built-in Linux host provider and explicit Linux extension modules.

pub(crate) mod dmabuf;
pub(crate) mod drm;
pub(crate) mod file;
pub(crate) mod ioctl;
pub(crate) mod mapping;
mod process;
mod reactor;
mod wait;

pub(crate) use process::LinuxLibrary as BuiltinLibrary;

use super::async_services::CallerDrivenBlockingFactory;
use super::HostEnvironment;
use file::LinuxFiles;
use process::{
    LinuxAllocator, LinuxArtifacts, LinuxClock, LinuxConfig, LinuxDiagnostics, LinuxLibraries,
    LinuxMath, LinuxMemoryStatus,
};
use reactor::LinuxReactorFactory;

pub(crate) const ERROR_DOMAIN: u32 = u32::from_be_bytes(*b"LNX ");

pub(crate) struct BuiltinHost {
    files: LinuxFiles,
    clock: LinuxClock,
    memory: LinuxMemoryStatus,
    math: LinuxMath,
    artifacts: LinuxArtifacts,
    config: LinuxConfig,
    diagnostics: LinuxDiagnostics,
    allocator: LinuxAllocator,
    libraries: LinuxLibraries,
    atomic_wait: wait::LinuxAtomicWait,
    reactor_factory: LinuxReactorFactory,
    blocking_executor_factory: CallerDrivenBlockingFactory,
}

impl HostEnvironment for BuiltinHost {
    type Files = LinuxFiles;
    type Clock = LinuxClock;
    type Memory = LinuxMemoryStatus;
    type Math = LinuxMath;
    type Artifacts = LinuxArtifacts;
    type Config = LinuxConfig;
    type Diagnostics = LinuxDiagnostics;
    type Allocator = LinuxAllocator;
    type Libraries = LinuxLibraries;
    type AtomicWait = wait::LinuxAtomicWait;
    type ReactorFactory = LinuxReactorFactory;
    type BlockingExecutorFactory = CallerDrivenBlockingFactory;

    fn files(&self) -> &Self::Files {
        &self.files
    }

    fn clock(&self) -> &Self::Clock {
        &self.clock
    }

    fn memory(&self) -> &Self::Memory {
        &self.memory
    }

    fn math(&self) -> &Self::Math {
        &self.math
    }

    fn artifacts(&self) -> &Self::Artifacts {
        &self.artifacts
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn diagnostics(&self) -> &Self::Diagnostics {
        &self.diagnostics
    }

    fn allocator(&self) -> &Self::Allocator {
        &self.allocator
    }

    fn libraries(&self) -> &Self::Libraries {
        &self.libraries
    }

    fn atomic_wait(&self) -> &Self::AtomicWait {
        &self.atomic_wait
    }

    fn reactor_factory(&self) -> &Self::ReactorFactory {
        &self.reactor_factory
    }

    fn blocking_executor_factory(&self) -> &Self::BlockingExecutorFactory {
        &self.blocking_executor_factory
    }
}

static BUILTIN: BuiltinHost = BuiltinHost {
    files: LinuxFiles,
    clock: LinuxClock,
    memory: LinuxMemoryStatus,
    math: LinuxMath,
    artifacts: LinuxArtifacts,
    config: LinuxConfig,
    diagnostics: LinuxDiagnostics,
    allocator: LinuxAllocator,
    libraries: LinuxLibraries,
    atomic_wait: wait::LinuxAtomicWait,
    reactor_factory: LinuxReactorFactory,
    blocking_executor_factory: CallerDrivenBlockingFactory,
};

pub(crate) const fn builtin() -> &'static BuiltinHost {
    &BUILTIN
}

#[cfg(all(test, feature = "bench-hardware"))]
#[derive(Debug)]
pub(crate) struct LinuxResourceSnapshot {
    pub open_fds: usize,
    pub rss_kib: Option<u64>,
    pub swap_kib: Option<u64>,
    pub available_kib: Option<u64>,
}

#[cfg(all(test, feature = "bench-hardware"))]
fn proc_kib(path: &str, key: &str) -> Option<u64> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut bytes = [0u8; 4096];
    let length = file.read(&mut bytes).ok()?;
    let text = core::str::from_utf8(&bytes[..length]).ok()?;
    text.lines()
        .find(|line| line.starts_with(key))?
        .split_ascii_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

#[cfg(all(test, feature = "bench-hardware"))]
pub(crate) fn resource_snapshot() -> LinuxResourceSnapshot {
    LinuxResourceSnapshot {
        open_fds: std::fs::read_dir("/proc/self/fd")
            .map(|entries| entries.count())
            .unwrap_or(0),
        rss_kib: proc_kib("/proc/self/status", "VmRSS:"),
        swap_kib: proc_kib("/proc/self/status", "VmSwap:"),
        available_kib: proc_kib("/proc/meminfo", "MemAvailable:"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::ClockService;
    use super::*;

    #[test]
    fn monotonic_clock_moves_forward() {
        let first = BUILTIN.clock.monotonic_now_ns().unwrap();
        let second = BUILTIN.clock.monotonic_now_ns().unwrap();
        assert!(second >= first);
    }
}
