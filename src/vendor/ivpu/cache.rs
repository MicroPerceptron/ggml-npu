//! A lock-free, bounded-retention cache keyed by exact `PartialEq` semantics.
//!
//! Reads (the hot path) are a plain atomic-load traversal — no lock (per the atomics-over-locks
//! rule). Entries are not removed yet (they hold process-lifetime artifacts), so there is
//! no reclamation or ABA problem; writes prepend via CAS. Each node is its own cacheline so a
//! lookup touches one line per entry and the hot `head` never false-shares. Concurrent misses of
//! the same key could duplicate an entry (harmless — both valid, lookup returns one), but compute
//! is single-threaded per backend, so that path is a formality.
//!
//! Keys in use are all `Copy` tuples: `(weight pointer, M)` for the per-weight Constant blob,
//! the blob address for its per-blob graph, and a per-node pointer-triple for the recorded
//! execute list. Once its explicit entry budget is reached, misses fall back without compiling.

use alloc::boxed::Box;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

struct Node<K, V> {
    key: K,
    val: V,
    next: AtomicPtr<Node<K, V>>,
}

/// Aligned to a cacheline so the hot `head` atomic (read on every lookup) never false-shares
/// with adjacent statics. (Nodes are left unaligned — they're heap-scattered anyway, and forcing
/// one-per-line would only waste space, not improve traversal locality.)
#[repr(align(64))]
pub(super) struct Cache<K, V> {
    head: AtomicPtr<Node<K, V>>,
    count: AtomicUsize,
    limit: usize,
}

impl<K: PartialEq, V: Copy> Cache<K, V> {
    pub(super) const fn new(limit: usize) -> Self {
        Cache {
            head: AtomicPtr::new(core::ptr::null_mut()),
            count: AtomicUsize::new(0),
            limit,
        }
    }

    /// Lock-free lookup.
    pub(super) fn get(&self, key: &K) -> Option<V> {
        let mut p = self.head.load(Ordering::Acquire);
        while !p.is_null() {
            let node = unsafe { &*p };
            if node.key.eq(key) {
                return Some(node.val);
            }
            p = node.next.load(Ordering::Acquire);
        }
        None
    }

    /// Bounded insertion. Returns `None` without invoking `f` once the retention budget is full,
    /// allowing the caller to fall back instead of compiling another process-lifetime artifact.
    pub(super) fn try_get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> Option<V> {
        if let Some(v) = self.get(&key) {
            return Some(v);
        }
        let slot = self.count.fetch_add(1, Ordering::AcqRel);
        if slot >= self.limit {
            self.count.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        let node = Box::into_raw(Box::new(Node {
            key,
            val: f(),
            next: AtomicPtr::new(core::ptr::null_mut()),
        }));
        let v = unsafe { (*node).val };
        loop {
            if let Some(existing) = self.get(unsafe { &(*node).key }) {
                drop(unsafe { Box::from_raw(node) });
                self.count.fetch_sub(1, Ordering::AcqRel);
                return Some(existing);
            }
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*node).next.store(head, Ordering::Relaxed) };
            if self
                .head
                .compare_exchange(head, node, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(v);
            }
        }
    }
}

// SAFETY: shared across threads; all mutation is via atomics, keys/values are Copy + 'static.
unsafe impl<K: Send, V: Copy + Send> Sync for Cache<K, V> {}
