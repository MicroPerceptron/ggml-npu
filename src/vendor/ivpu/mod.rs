//! Intel NPU (`intel_vpu` driver) — the active vendor.
//!
//! Runtime: **Level Zero** ([`ze`], [`graph`]). Compile: OpenVINO via the shim, cached
//! per-shape ([`compile`]). Op lowerings live under [`lower`]. [`IvpuVendor`] is the
//! [`crate::vendor::Vendor`] impl the agnostic backend dispatches to; [`IvpuHal`] is the
//! (still-aspirational) 5-verb submission seam.

pub(crate) mod address;
pub(crate) mod artifact;
mod cache;
mod compile;
pub(crate) use compile::wire_op_compile_supported;
#[cfg(feature = "direct-ivpu")]
pub mod direct;
#[cfg(all(feature = "bench-hardware", feature = "level-zero-compat"))]
#[path = "session_direct.rs"]
mod direct_benchmark_session;
#[cfg(feature = "level-zero-compat")]
mod graph;
mod lower;
mod profile;
#[cfg(feature = "level-zero-compat")]
mod session;
#[cfg(all(not(feature = "level-zero-compat"), feature = "direct-ivpu"))]
#[path = "session_direct.rs"]
mod session;
#[cfg(all(not(feature = "level-zero-compat"), not(feature = "direct-ivpu")))]
#[path = "session_unavailable.rs"]
mod session;
#[cfg(feature = "level-zero-compat")]
mod ze;

use alloc::vec::Vec;
use core::alloc::Layout;
#[cfg(feature = "level-zero-compat")]
use core::alloc::{AllocError, Allocator};
use core::ffi::c_void;
#[cfg(feature = "level-zero-compat")]
use core::ptr::NonNull;

#[cfg(feature = "level-zero-compat")]
use crate::hal::{BufFlags, NpuHal};
#[cfg(not(feature = "level-zero-compat"))]
use crate::host::{AllocationService, HostEnvironment};
use crate::ops::{Op, Operand};
use crate::vendor::{DeviceInfo, LoweringSupport, Vendor};

#[cfg(feature = "level-zero-compat")]
pub use ze::Device as IvpuDevice;

/// [`Allocator`] over NPU-MMU-mapped host memory (`zeMemAllocHost`), so NPU-resident buffers are
/// first-class `Vec<u8, NpuHostAlloc>` instead of the raw-pointer + length bookkeeping the session
/// does today (issue #1). The driver page-aligns (4096), satisfying any alignment we bind. Returns
/// [`AllocError`] when the Level Zero runtime is absent (no NPU) or the request is over-aligned —
/// a caller then falls back to [`crate::HostAlloc`]. ZST: free to pass by value.
#[cfg(feature = "level-zero-compat")]
#[derive(Clone, Copy, Default)]
pub(super) struct NpuHostAlloc;

#[cfg(feature = "level-zero-compat")]
unsafe impl Allocator for NpuHostAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            let d = unsafe { NonNull::new_unchecked(layout.align() as *mut u8) };
            return Ok(NonNull::slice_from_raw_parts(d, 0));
        }
        if layout.align() > 4096 {
            return Err(AllocError);
        }
        let p = ze::alloc_host(layout.size()) as *mut u8;
        NonNull::new(p)
            .map(|p| NonNull::slice_from_raw_parts(p, layout.size()))
            .ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            ze::free_host(ptr.as_ptr() as *mut c_void);
        }
    }
}

/// The Intel NPU vendor: probe via Level Zero, lower/run ops via the JIT cache + graph submit.
pub struct IvpuVendor;

impl Vendor for IvpuVendor {
    fn name(&self) -> &'static str {
        "ivpu"
    }

    fn probe(&self) -> Vec<DeviceInfo> {
        #[cfg(any(feature = "level-zero-compat", feature = "direct-ivpu"))]
        let mut v = Vec::new();
        #[cfg(not(any(feature = "level-zero-compat", feature = "direct-ivpu")))]
        let v = Vec::new();
        #[cfg(feature = "level-zero-compat")]
        if let Some(r) = ze::runtime() {
            v.push(DeviceInfo {
                name: r.name.clone(),
            });
        }
        #[cfg(all(not(feature = "level-zero-compat"), feature = "direct-ivpu"))]
        if let Ok(device) = direct::IvpuDevice::open_first() {
            if let Ok(parameters) = device.parameters() {
                v.push(DeviceInfo {
                    name: alloc::format!(
                        "Intel NPU (direct ivpu {:#x} rev {})",
                        parameters.device_id,
                        parameters.revision
                    ),
                });
            }
        }
        v
    }

    fn can_lower(&self, op: Op, inputs: &[Operand]) -> bool {
        match op {
            Op::MatMul { m, k, n } => lower::matmul::can_lower(m, k, n, inputs),
        }
    }

    fn run(&self, op: Op, inputs: &[Operand], output: &Operand) -> bool {
        match op {
            Op::MatMul { m, k, n } => lower::matmul::run(m, k, n, inputs, output),
        }
    }

    fn run_subgraph(&self, sg: &crate::ops::subgraph::Subgraph) -> bool {
        lower::subgraph::run(sg)
    }

    fn graph_support(&self, graph: &crate::graph::GraphIr) -> LoweringSupport {
        if lower::graph::can_lower(graph) {
            LoweringSupport::Supported
        } else {
            LoweringSupport::Unsupported("Intel graph compiler rejected split")
        }
    }

    fn run_graph(&self, graph: &crate::graph::GraphIr) -> bool {
        lower::graph::run(graph)
    }

    unsafe fn node_support(
        &self,
        node: *const crate::ffi::ggml_tensor,
        spec: &crate::graph::OpSpec,
    ) -> LoweringSupport {
        if compile::graph_node_feasible(node, spec) {
            LoweringSupport::Supported
        } else {
            LoweringSupport::Unsupported("Intel node lowering or policy rejected operation")
        }
    }

    fn synchronize(&self) -> bool {
        session::synchronize()
    }

    fn debug_note(&self, message: &str) {
        profile::note(message);
    }

    unsafe fn alloc_host(&self, size: usize) -> *mut c_void {
        #[cfg(feature = "level-zero-compat")]
        {
            return ze::alloc_host(size);
        }
        #[cfg(not(feature = "level-zero-compat"))]
        {
            let Ok(layout) = Layout::from_size_align(size.max(1), 4096) else {
                return core::ptr::null_mut();
            };
            crate::host::builtin().allocator().allocate(layout).cast()
        }
    }

    unsafe fn free_host(&self, ptr: *mut c_void, size: usize) {
        #[cfg(feature = "level-zero-compat")]
        {
            ze::free_host(ptr);
        }
        #[cfg(not(feature = "level-zero-compat"))]
        {
            if let Ok(layout) = Layout::from_size_align(size.max(1), 4096) {
                crate::host::builtin()
                    .allocator()
                    .deallocate(ptr.cast(), layout);
            }
        }
    }

    fn alloc_shared_backend(
        &self,
        size: usize,
    ) -> Option<alloc::boxed::Box<dyn crate::interop::SharedAllocationOwner>> {
        #[cfg(all(feature = "direct-ivpu", not(feature = "level-zero-compat")))]
        {
            return direct::linux::DirectSharedBuffer::allocate(size).ok();
        }
        #[cfg(any(not(feature = "direct-ivpu"), feature = "level-zero-compat"))]
        {
            let _ = size;
            None
        }
    }

    fn note_shared_allocation(&self, identity: crate::interop::AllocationIdentity, bytes: usize) {
        profile::event(crate::metrics::MetricEvent::SharedAllocation {
            allocation: identity.external_id(),
            bytes: bytes as u64,
            imported: false,
        });
    }
}

/// Bring-up self-test: run a native (OpenVINO-compiled) blob on the first discovered NPU,
/// feeding `input` (f32) and writing the first output into `output` (f32). Returns the
/// number of outputs written, or a negative error. Scaffold for the submit path.
#[cfg(feature = "level-zero-compat")]
pub fn run_blob_selftest(blob: &[u8], input: &[f32], output: &mut [f32]) -> i32 {
    run_blob_selftest_multi(blob, &[input], output)
}

#[cfg(feature = "level-zero-compat")]
pub fn run_blob_selftest_multi(blob: &[u8], inputs: &[&[f32]], output: &mut [f32]) -> i32 {
    let devices = <IvpuHal as NpuHal>::probe();
    let Some(dev) = devices.first() else {
        return -100;
    };
    match graph::run_native_blob(dev, blob, inputs, output) {
        Ok(n) => n as i32,
        Err(code) if code > 0 => -code,
        Err(0) => -1,
        Err(code) => code,
    }
}

#[cfg(not(feature = "level-zero-compat"))]
pub fn run_blob_selftest(blob: &[u8], input: &[f32], output: &mut [f32]) -> i32 {
    run_blob_selftest_multi(blob, &[input], output)
}

#[cfg(not(feature = "level-zero-compat"))]
pub fn run_blob_selftest_multi(blob: &[u8], inputs: &[&[f32]], output: &mut [f32]) -> i32 {
    match session::run_blob_selftest_multi(blob, inputs, output) {
        Ok(written) => written as i32,
        Err(_) => -1,
    }
}

/// Persistent Level Zero fixture exposed only to the guarded hardware Criterion target.
#[cfg(all(feature = "bench-hardware", feature = "level-zero-compat"))]
pub struct LevelZeroBenchmarkFixture(graph::PersistentNativeGraph);

#[cfg(all(feature = "bench-hardware", feature = "level-zero-compat"))]
impl LevelZeroBenchmarkFixture {
    pub fn new(blob: &[u8], inputs: &[&[f32]]) -> Result<Self, i32> {
        let devices = ze::probe();
        let device = devices.first().ok_or(-100)?;
        graph::PersistentNativeGraph::new(device, blob, inputs).map(Self)
    }

    pub fn execute(&mut self, timeout_ns: u64) -> Result<(), i32> {
        self.0.execute(timeout_ns)
    }

    pub fn copy_output(&self, output: &mut [f32]) -> Result<usize, i32> {
        self.0.copy_output(output)
    }
}

#[cfg(feature = "bench-hardware")]
pub(crate) enum HardwareBenchmarkFixture {
    Direct(direct_benchmark_session::DirectBenchmarkFixture),
    LevelZero(LevelZeroBenchmarkFixture),
}

#[cfg(feature = "bench-hardware")]
impl HardwareBenchmarkFixture {
    pub fn new(kind: u32, blob: &[u8], inputs: &[&[f32]]) -> Result<Self, i32> {
        match kind {
            0 => direct_benchmark_session::DirectBenchmarkFixture::new(blob, inputs)
                .map(Self::Direct)
                .map_err(|_| -1),
            1 => LevelZeroBenchmarkFixture::new(blob, inputs)
                .map(Self::LevelZero)
                .map_err(|_| -1),
            _ => Err(-1),
        }
    }

    pub fn execute(&mut self) -> Result<(), i32> {
        match self {
            Self::Direct(fixture) => fixture.execute().map_err(|_| -1),
            Self::LevelZero(fixture) => fixture.execute(2_000_000_000),
        }
    }

    pub fn copy_output(&self, output: &mut [f32]) -> Result<usize, i32> {
        match self {
            Self::Direct(fixture) => fixture.copy_output(output).map_err(|_| -1),
            Self::LevelZero(fixture) => fixture.copy_output(output),
        }
    }
}

#[cfg(feature = "level-zero-compat")]
pub struct IvpuHal; // ze driver/context + graph-ext dispatch table
#[cfg(feature = "level-zero-compat")]
pub struct IvpuQueue; // ze_command_queue_handle_t (+ command list)
#[cfg(feature = "level-zero-compat")]
pub struct IvpuBinary; // ze_graph_handle_t created from a NATIVE blob
#[cfg(feature = "level-zero-compat")]
pub struct IvpuBuffer; // zeMemAllocHost / zeMemAllocDevice allocation

#[cfg(feature = "level-zero-compat")]
impl NpuHal for IvpuHal {
    type Device = IvpuDevice;
    type Queue = IvpuQueue;
    type Binary = IvpuBinary;
    type Buffer = IvpuBuffer;

    fn probe() -> Vec<IvpuDevice> {
        ze::probe()
    }

    fn queue_create(&self, dev: &IvpuDevice) -> IvpuQueue {
        unimplemented!("zeCommandQueueCreate")
    }
    fn load(&self, dev: &IvpuDevice, artifact: &[u8]) -> IvpuBinary {
        unimplemented!("graph ext pfnCreate(blob, ZE_GRAPH_FORMAT_NATIVE)")
    }
    fn alloc(&self, dev: &IvpuDevice, bytes: usize, flags: BufFlags) -> IvpuBuffer {
        unimplemented!("zeMemAllocHost when flags.host_shared, else zeMemAllocDevice")
    }
    fn map<'a>(&self, buf: &'a mut IvpuBuffer) -> &'a mut [u8] {
        unimplemented!("host allocations are already mapped")
    }
    fn sync_to_device(&self, buf: &IvpuBuffer) {
        unimplemented!()
    }
    fn sync_from_device(&self, buf: &IvpuBuffer) {
        unimplemented!()
    }
    fn bind(&self, bin: &mut IvpuBinary, index: u32, buf: &IvpuBuffer) {
        // INDEXED: `index` is the graph argument index queried from the blob.
        unimplemented!("pfnSetArgumentValue(graph, index, ptr)")
    }
    fn submit(&self, q: &IvpuQueue, bin: &IvpuBinary) {
        unimplemented!("pfnAppendGraphExecute + zeCommandQueueExecuteCommandLists")
    }
    fn sync(&self, q: &IvpuQueue) {
        unimplemented!("zeCommandQueueSynchronize")
    }
}

#[cfg(all(test, not(feature = "level-zero-compat")))]
mod direct_tests {
    use super::*;

    #[test]
    #[ignore = "explicit direct-runtime hardware execution using an existing bounded artifact"]
    fn default_runtime_executes_tiny_known_answer_without_level_zero() {
        let artifact = direct::load_test_artifact(1024 * 1024);
        let input0 = core::array::from_fn::<_, 32, _>(|index| index as f32 * 0.125 - 1.5);
        let input1 =
            core::array::from_fn::<_, 32, _>(|index| ((index * 7) % 17) as f32 * 0.25 - 2.0);
        let mut output = [0.0f32; 32];
        assert_eq!(
            run_blob_selftest_multi(&artifact, &[&input0, &input1], &mut output),
            32
        );
        for ((actual, left), right) in output.iter().zip(input0).zip(input1) {
            assert_eq!(actual.to_bits(), (left + right).to_bits());
        }
    }
}
