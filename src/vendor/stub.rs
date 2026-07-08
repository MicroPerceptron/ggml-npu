//! CPU-echo HAL. No hardware: "device memory" is host memory, `submit`/`sync` are no-ops.
//! Exercises the seam end-to-end with no NPU, and gives Phase 1 a reference to diff a real
//! vendor impl against. Its buffer is written with the `allocator_api` `_in` constructors
//! (`A = Global` here) to establish the pattern vendors follow with device allocators.

use alloc::alloc::Global;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::hal::{BufFlags, NpuHal};

pub struct StubHal;

pub struct StubDevice {
    pub name: String,
}

pub struct StubQueue;

pub struct StubBinary {
    pub bytes: Vec<u8>,
    /// Bound argument buffers by index (mirrors a real kernel's positional args).
    pub args: Vec<*const u8>,
}

pub struct StubBuffer {
    /// `A = Global`; a real vendor swaps in its device allocator here.
    pub data: Vec<u8, Global>,
}

impl NpuHal for StubHal {
    type Device = StubDevice;
    type Queue = StubQueue;
    type Binary = StubBinary;
    type Buffer = StubBuffer;

    fn probe() -> Vec<StubDevice> {
        let mut v = Vec::new();
        v.push(StubDevice {
            name: "npu-stub0".to_string(),
        });
        v
    }

    fn queue_create(&self, _dev: &StubDevice) -> StubQueue {
        StubQueue
    }

    fn load(&self, _dev: &StubDevice, artifact: &[u8]) -> StubBinary {
        StubBinary {
            bytes: artifact.to_vec(),
            args: Vec::new(),
        }
    }

    fn alloc(&self, _dev: &StubDevice, bytes: usize, _flags: BufFlags) -> StubBuffer {
        let mut data = Vec::with_capacity_in(bytes, Global);
        data.resize(bytes, 0u8);
        StubBuffer { data }
    }

    fn map<'a>(&self, buf: &'a mut StubBuffer) -> &'a mut [u8] {
        &mut buf.data
    }

    fn sync_to_device(&self, _buf: &StubBuffer) {}
    fn sync_from_device(&self, _buf: &StubBuffer) {}

    fn bind(&self, bin: &mut StubBinary, index: u32, buf: &StubBuffer) {
        let i = index as usize;
        if bin.args.len() <= i {
            bin.args.resize(i + 1, core::ptr::null());
        }
        bin.args[i] = buf.data.as_ptr();
    }

    fn submit(&self, _q: &StubQueue, _bin: &StubBinary) {}
    fn sync(&self, _q: &StubQueue) {}
}
