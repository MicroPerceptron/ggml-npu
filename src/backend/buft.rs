//! Buffer-type policy and attribution for Direct vs Level Zero residency (#6).
//!
//! Direct IVPU can only import `SharedAllocationOwner` allocations from our `NPU_HOST` buft.
//! Accepting every host buft (Level Zero–era) left tensors on CPU buffers and forced staging.
//! This module:
//! - classifies bufts / tensors for profile attribution;
//! - answers the `supports_buft` contract without blindly widening;
//! - keeps a gated importable-buft allowlist off by default (hermetic tests lock that).

use core::sync::atomic::{AtomicU64, Ordering};

use crate::ffi::{ggml_backend_buffer_type_t, ggml_backend_buft_is_host, ggml_tensor};

/// Process-wide `supports_buft` probe attribution (Direct gap vs L0 broad host contract).
static PROBE_OURS_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static PROBE_FOREIGN_HOST_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static PROBE_FOREIGN_HOST_REFUSED: AtomicU64 = AtomicU64::new(0);
static PROBE_FOREIGN_DEVICE_REFUSED: AtomicU64 = AtomicU64::new(0);
static PROBE_NULL_REFUSED: AtomicU64 = AtomicU64::new(0);

/// Tensor residency seen when capturing Graph IR / operands for claimed NPU work.
static RES_OUR_SHARED: AtomicU64 = AtomicU64::new(0);
static RES_OUR_LEGACY_HOST: AtomicU64 = AtomicU64::new(0);
static RES_FOREIGN_HOST: AtomicU64 = AtomicU64::new(0);
static RES_FOREIGN_DEVICE: AtomicU64 = AtomicU64::new(0);
static RES_UNALLOCATED: AtomicU64 = AtomicU64::new(0);
static RES_DISHONEST_NPU_HOST: AtomicU64 = AtomicU64::new(0);

/// Class of a ggml buffer type relative to our singleton `NPU_HOST`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuftClass {
    Null,
    /// Our device buffer type (`NPU_HOST`).
    Ours,
    /// Another backend's host-visible buft (CPU, etc.).
    ForeignHost,
    /// Non-host device buft (Vulkan, etc.).
    ForeignDevice,
}

/// Where a tensor's storage actually lives for zero-copy binding purposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorResidency {
    Unallocated,
    /// Our buft + `SharedAllocationOwner` — Direct-importable.
    OurShared,
    /// Our buft with legacy `alloc_host` base only (Level Zero can bind; Direct cannot).
    OurLegacyHost,
    ForeignHost,
    ForeignDevice,
}

/// Snapshot of buft probe + residency counters for profile / tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuftAttribution {
    pub probe_ours_accepted: u64,
    pub probe_foreign_host_accepted: u64,
    pub probe_foreign_host_refused: u64,
    pub probe_foreign_device_refused: u64,
    pub probe_null_refused: u64,
    pub res_our_shared: u64,
    pub res_our_legacy_host: u64,
    pub res_foreign_host: u64,
    pub res_foreign_device: u64,
    pub res_unallocated: u64,
    pub res_dishonest_npu_host: u64,
}

/// Direct builds use the narrow contract; Level Zero / non-Direct keep broad host acceptance.
#[inline]
pub const fn direct_narrow_buft_contract() -> bool {
    cfg!(all(feature = "direct-ivpu", not(feature = "level-zero-compat")))
}

/// Pure policy used by `supports_buft` and hermetic tests.
///
/// - Direct narrow: only our buft.
/// - Otherwise: any host buft (includes ours; `is_host` must be true for ours).
///
/// Selective foreign-host allowlisting is intentionally not wired here — see
/// [`importable_foreign_host_allowlist_enabled`].
#[inline]
pub fn supports_buft_policy(is_ours: bool, is_host: bool) -> bool {
    if direct_narrow_buft_contract() {
        is_ours
    } else {
        is_host
    }
}

/// Gated spike for selective importable foreign-host bufts. Always false until a hermetic
/// allowlist + import proof lands; widening all host bufts remains a non-goal.
#[inline]
pub fn importable_foreign_host_allowlist_enabled() -> bool {
    false
}

/// Classify a buft pointer against our singleton and ggml's host flag.
pub(crate) unsafe fn classify_buft(buft: ggml_backend_buffer_type_t) -> BuftClass {
    if buft.is_null() {
        return BuftClass::Null;
    }
    if buft == super::buffer_type() {
        return BuftClass::Ours;
    }
    if ggml_backend_buft_is_host(buft) {
        BuftClass::ForeignHost
    } else {
        BuftClass::ForeignDevice
    }
}

/// Classify tensor storage for Direct-honest residency attribution.
pub(crate) unsafe fn classify_tensor(tensor: *const ggml_tensor) -> TensorResidency {
    if tensor.is_null() {
        return TensorResidency::Unallocated;
    }
    let buf = (*tensor).buffer;
    if buf.is_null() {
        return TensorResidency::Unallocated;
    }
    let buft = (*buf).buft;
    match classify_buft(buft) {
        BuftClass::Null => TensorResidency::Unallocated,
        BuftClass::Ours => {
            if super::buffer::context(buf)
                .and_then(|ctx| ctx.owner())
                .is_some()
            {
                TensorResidency::OurShared
            } else {
                TensorResidency::OurLegacyHost
            }
        }
        BuftClass::ForeignHost => TensorResidency::ForeignHost,
        BuftClass::ForeignDevice => TensorResidency::ForeignDevice,
    }
}

/// `supports_buft` entry: classify, attribute, apply policy (allowlist stays off).
pub(crate) unsafe fn supports_buft_attributed(buft: ggml_backend_buffer_type_t) -> bool {
    let class = classify_buft(buft);
    let is_ours = class == BuftClass::Ours;
    let is_host = matches!(class, BuftClass::Ours | BuftClass::ForeignHost);
    let accept = supports_buft_policy(is_ours, is_host)
        || (importable_foreign_host_allowlist_enabled() && class == BuftClass::ForeignHost);

    match class {
        BuftClass::Null => {
            PROBE_NULL_REFUSED.fetch_add(1, Ordering::Relaxed);
        }
        BuftClass::Ours => {
            if accept {
                PROBE_OURS_ACCEPTED.fetch_add(1, Ordering::Relaxed);
            }
        }
        BuftClass::ForeignHost => {
            if accept {
                PROBE_FOREIGN_HOST_ACCEPTED.fetch_add(1, Ordering::Relaxed);
            } else {
                // Direct refuses what Level Zero would accept — primary L0/Direct gap signal.
                PROBE_FOREIGN_HOST_REFUSED.fetch_add(1, Ordering::Relaxed);
            }
        }
        BuftClass::ForeignDevice => {
            PROBE_FOREIGN_DEVICE_REFUSED.fetch_add(1, Ordering::Relaxed);
        }
    }
    accept
}

/// Record residency for one captured tensor (Graph IR / operand path).
pub(crate) unsafe fn record_tensor_residency(tensor: *const ggml_tensor) {
    match classify_tensor(tensor) {
        TensorResidency::OurShared => {
            RES_OUR_SHARED.fetch_add(1, Ordering::Relaxed);
        }
        TensorResidency::OurLegacyHost => {
            RES_OUR_LEGACY_HOST.fetch_add(1, Ordering::Relaxed);
            // Direct must never present bare-host `NPU_HOST` as bindable (#6).
            if direct_narrow_buft_contract() {
                RES_DISHONEST_NPU_HOST.fetch_add(1, Ordering::Relaxed);
            }
        }
        TensorResidency::ForeignHost => {
            RES_FOREIGN_HOST.fetch_add(1, Ordering::Relaxed);
        }
        TensorResidency::ForeignDevice => {
            RES_FOREIGN_DEVICE.fetch_add(1, Ordering::Relaxed);
        }
        TensorResidency::Unallocated => {
            RES_UNALLOCATED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn buft_attribution() -> BuftAttribution {
    BuftAttribution {
        probe_ours_accepted: PROBE_OURS_ACCEPTED.load(Ordering::Relaxed),
        probe_foreign_host_accepted: PROBE_FOREIGN_HOST_ACCEPTED.load(Ordering::Relaxed),
        probe_foreign_host_refused: PROBE_FOREIGN_HOST_REFUSED.load(Ordering::Relaxed),
        probe_foreign_device_refused: PROBE_FOREIGN_DEVICE_REFUSED.load(Ordering::Relaxed),
        probe_null_refused: PROBE_NULL_REFUSED.load(Ordering::Relaxed),
        res_our_shared: RES_OUR_SHARED.load(Ordering::Relaxed),
        res_our_legacy_host: RES_OUR_LEGACY_HOST.load(Ordering::Relaxed),
        res_foreign_host: RES_FOREIGN_HOST.load(Ordering::Relaxed),
        res_foreign_device: RES_FOREIGN_DEVICE.load(Ordering::Relaxed),
        res_unallocated: RES_UNALLOCATED.load(Ordering::Relaxed),
        res_dishonest_npu_host: RES_DISHONEST_NPU_HOST.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub fn reset_buft_attribution_for_test() {
    for c in [
        &PROBE_OURS_ACCEPTED,
        &PROBE_FOREIGN_HOST_ACCEPTED,
        &PROBE_FOREIGN_HOST_REFUSED,
        &PROBE_FOREIGN_DEVICE_REFUSED,
        &PROBE_NULL_REFUSED,
        &RES_OUR_SHARED,
        &RES_OUR_LEGACY_HOST,
        &RES_FOREIGN_HOST,
        &RES_FOREIGN_DEVICE,
        &RES_UNALLOCATED,
        &RES_DISHONEST_NPU_HOST,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_stays_off() {
        assert!(!importable_foreign_host_allowlist_enabled());
    }

    #[test]
    fn direct_narrow_accepts_only_ours() {
        if !direct_narrow_buft_contract() {
            return;
        }
        assert!(supports_buft_policy(true, true));
        assert!(supports_buft_policy(true, false)); // ours is still ours even if host flag wrong
        assert!(!supports_buft_policy(false, true));
        assert!(!supports_buft_policy(false, false));
    }

    #[test]
    fn broad_host_accepts_any_host() {
        if direct_narrow_buft_contract() {
            return;
        }
        assert!(supports_buft_policy(true, true));
        assert!(supports_buft_policy(false, true));
        assert!(!supports_buft_policy(false, false));
        assert!(!supports_buft_policy(true, false));
    }

    #[test]
    fn attribution_resets_clean() {
        reset_buft_attribution_for_test();
        assert_eq!(buft_attribution(), BuftAttribution::default());
    }

    #[test]
    fn policy_never_accepts_foreign_host_under_direct_without_allowlist() {
        // Documents the L0/Direct gap: foreign host is refused under Direct.
        if direct_narrow_buft_contract() {
            assert!(!supports_buft_policy(false, true));
            assert!(!importable_foreign_host_allowlist_enabled());
        }
    }
}
