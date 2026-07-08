//! A lock-free, append-only cache keyed by a `Copy + PartialEq` key.
//!
//! Reads (the hot path) are a plain atomic-load traversal — no lock (per the atomics-over-locks
//! rule). Entries are never removed (they hold leaked, process-lifetime artifacts), so there is
//! no reclamation or ABA problem; writes prepend via CAS. Each node is its own cacheline so a
//! lookup touches one line per entry and the hot `head` never false-shares. Concurrent misses of
//! the same key could duplicate an entry (harmless — both valid, lookup returns one), but compute
//! is single-threaded per backend, so that path is a formality.
//!
//! Keys in use are all `Copy` tuples: `(weight pointer, M)` for the per-weight Constant blob,
//! the blob address for its per-blob graph, and a per-node pointer-triple for the recorded
//! execute list — see [`PtrKey`].

use alloc::boxed::Box;
use core::sync::atomic::{AtomicPtr, Ordering};

/// Per-node binding key: `(activation, output, blob)` data/artifact addresses. Uniquely
/// identifies one graph node's bindings, which are stable across tokens.
pub(super) type PtrKey = (usize, usize, usize);

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
}

impl<K: Copy + PartialEq, V: Copy> Cache<K, V> {
    pub(super) const fn new() -> Self {
        Cache { head: AtomicPtr::new(core::ptr::null_mut()) }
    }

    /// Lock-free lookup.
    pub(super) fn get(&self, key: K) -> Option<V> {
        let mut p = self.head.load(Ordering::Acquire);
        while !p.is_null() {
            let node = unsafe { &*p };
            if node.key == key {
                return Some(node.val);
            }
            p = node.next.load(Ordering::Acquire);
        }
        None
    }

    /// Cached value for `key`, computing + inserting it on a miss. `f` runs with **no lock held**
    /// (it may compile a blob / build a graph — seconds); on a race the loser's node is dropped.
    pub(super) fn get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> V {
        if let Some(v) = self.get(key) {
            return v;
        }
        let node = Box::into_raw(Box::new(Node {
            key,
            val: f(),
            next: AtomicPtr::new(core::ptr::null_mut()),
        }));
        let v = unsafe { (*node).val };
        loop {
            if let Some(existing) = self.get(key) {
                // Someone else published this key first; drop our redundant node.
                drop(unsafe { Box::from_raw(node) });
                return existing;
            }
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*node).next.store(head, Ordering::Relaxed) };
            if self
                .head
                .compare_exchange(head, node, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return v;
            }
        }
    }
}

// SAFETY: shared across threads; all mutation is via atomics, keys/values are Copy + 'static.
unsafe impl<K: Copy + Send, V: Copy + Send> Sync for Cache<K, V> {}
