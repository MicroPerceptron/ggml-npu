//! Built-in host composition for the neutral tiered artifact cache.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use crate::artifact_cache::{
    ArtifactKey, CacheBudgets, CacheCompatibility, CacheObserver, TieredArtifactCache,
};
use crate::host::linux::cache::LinuxCompression;
use crate::host::{ConfigService, HostEnvironment};

const MIB: u64 = 1024 * 1024;
const DEFAULT_RAM_MIB: u64 = 256;
const DEFAULT_RAM_ENTRY_MIB: u64 = 64;
const DEFAULT_DISK_MIB: u64 = 24 * 1024;

static CACHE: TieredArtifactCache = TieredArtifactCache::new();
static BUDGETS: AtomicPtr<CacheBudgets> = AtomicPtr::new(ptr::null_mut());

fn env_mib(name: &CStr) -> Option<u64> {
    let value = super::builtin().config().value(name)?;
    let bytes = value.to_bytes();
    if bytes.is_empty() || bytes.len() > 20 {
        return None;
    }
    let mut parsed = 0u64;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        parsed = parsed
            .checked_mul(10)?
            .checked_add(u64::from(byte - b'0'))?;
    }
    parsed.checked_mul(MIB)
}

fn budgets() -> CacheBudgets {
    let current = BUDGETS.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: published budgets are immutable and retained for process lifetime.
        return unsafe { *current };
    }
    let ram_bytes = env_mib(c"GGML_NPU_RAM_CACHE_MB").unwrap_or(DEFAULT_RAM_MIB * MIB);
    let ram_entry_bytes = env_mib(c"GGML_NPU_RAM_CACHE_ENTRY_MB")
        .unwrap_or(DEFAULT_RAM_ENTRY_MIB * MIB)
        .min(ram_bytes);
    let disk_bytes = env_mib(c"GGML_NPU_DISK_CACHE_MB").unwrap_or(DEFAULT_DISK_MIB * MIB);
    let candidate = Box::into_raw(Box::new(CacheBudgets::new(
        ram_bytes,
        ram_entry_bytes,
        disk_bytes,
    )));
    match BUDGETS.compare_exchange(
        ptr::null_mut(),
        candidate,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => unsafe { *candidate },
        Err(winner) => {
            // SAFETY: this candidate was never published.
            drop(unsafe { Box::from_raw(candidate) });
            // SAFETY: the winner is immutable and process-resident.
            unsafe { *winner }
        }
    }
}

pub(crate) fn get_or_compile(
    key: ArtifactKey,
    compatibility: CacheCompatibility,
    observer: &dyn CacheObserver,
    compile: impl FnOnce() -> Option<Vec<u8>>,
) -> Option<Vec<u8>> {
    let host = super::builtin();
    if !<LinuxCompression>::available(host.compression()) {
        return compile();
    }
    CACHE.get_or_compile(
        key,
        compatibility,
        budgets(),
        host.compression(),
        host.artifact_cache(),
        host.wait_provider(),
        host.clock(),
        observer,
        compile,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budgets_are_explicit_stable_and_hysteretic() {
        let first = budgets();
        let second = budgets();
        assert_eq!(first, second);
        assert!(first.valid());
        assert!(first.ram_entry_bytes <= first.ram_bytes);
        assert!(first.disk_low_water_bytes < first.disk_bytes);
    }
}
