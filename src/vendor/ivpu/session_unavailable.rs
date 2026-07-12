//! No-runtime compatibility stub for `--no-default-features` contract builds.

use alloc::vec::Vec;

use crate::identity::ExactSemanticKey;
use crate::ops::Operand;
use crate::runtime::RuntimeError;

pub(super) const fn synchronize() -> bool {
    true
}

pub(super) const fn runtime_available() -> bool {
    false
}

pub(super) fn run_const(
    _weight_ptr: usize,
    _m: i64,
    _inputs: &[&Operand],
    _outputs: &[&Operand],
    _compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    false
}

pub(super) fn run_subgraph(
    _key: ExactSemanticKey,
    _inputs: &[&Operand],
    _outputs: &[&Operand],
    _compile: impl FnOnce() -> Option<Vec<u8>>,
) -> bool {
    false
}

pub(super) fn run_blob_selftest_multi(
    _blob: &[u8],
    _inputs: &[&[f32]],
    _output: &mut [f32],
) -> Result<usize, RuntimeError> {
    Err(RuntimeError::Unsupported)
}
