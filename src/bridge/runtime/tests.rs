use super::*;
use std::{cell::RefCell, rc::Rc};
use virtio_accel_core::*;
use virtio_accel_tosa::{ExtensionSet, Level, ProfileSet, Target, Version};
use virtio_accel_tosa_build::{Graph, Operator, OperatorKind, Tensor};

#[derive(Default)]
struct State {
    loads: usize,
    unloads: usize,
    allocs: usize,
    frees: usize,
    submits: usize,
    polls: usize,
    events: usize,
    contexts: usize,
    queues: usize,
    handle_drops: usize,
    reject: bool,
    indeterminate: bool,
    poll_error: bool,
    allocation_error: bool,
}
struct TestAccel(Rc<RefCell<State>>);
struct Handle(Rc<RefCell<State>>);
impl Drop for Handle {
    fn drop(&mut self) {
        self.0.borrow_mut().handle_drops += 1;
    }
}
struct TestBuffer {
    bytes: RefCell<Vec<u8>>,
    _owner: Handle,
}
impl TestAccel {
    fn handle(&self) -> Handle {
        Handle(self.0.clone())
    }
}
impl Accelerator for TestAccel {
    type Context = Handle;
    type Buffer = TestBuffer;
    type Program = Handle;
    type Queue = Handle;
    type Event = Handle;
    fn device_info(&self) -> Result<DeviceInfo, BackendError> {
        Ok(DeviceInfo {
            identity: DeviceIdentity {
                uuid: [0; 16],
                class: AcceleratorClass::OTHER,
                vendor_id: 0,
                device_id: 0,
            },
            capabilities: Capabilities::HOST_VISIBLE_MEMORY,
            limits: DeviceLimits {
                max_contexts: 1,
                max_buffers_per_context: 16,
                max_programs_per_context: 1,
                max_queues_per_context: 1,
                max_events_per_context: 1,
                max_bindings_per_submission: 16,
                max_buffer_bytes: 1024,
                max_artifact_bytes: 16384,
            },
        })
    }
    fn create_context(&self, _: ContextDesc) -> Result<Handle, BackendError> {
        Ok(self.handle())
    }
    fn destroy_context(&self, _: Handle) -> Result<(), ReleaseFailure<Handle>> {
        self.0.borrow_mut().contexts += 1;
        Ok(())
    }
    fn allocate_buffer(
        &self,
        _: &Handle,
        d: BufferDesc,
    ) -> Result<AllocatedBuffer<TestBuffer>, BackendError> {
        let mut s = self.0.borrow_mut();
        if s.allocation_error && s.allocs == 1 {
            return Err(BackendError::OutOfMemory);
        }
        s.allocs += 1;
        Ok(AllocatedBuffer::new(
            TestBuffer {
                bytes: RefCell::new(vec![0; d.bytes() as usize]),
                _owner: self.handle(),
            },
            BufferInfo::new(
                d,
                d.bytes(),
                64,
                BufferProperties::HOST_VISIBLE | BufferProperties::DIRECT_BINDING,
            )?,
        ))
    }
    fn write_buffer(
        &self,
        b: &mut TestBuffer,
        o: u64,
        s: &dyn ByteSource,
    ) -> Result<(), BackendError> {
        assert_eq!(o, 0);
        s.read_at(0, b.bytes.get_mut())
    }
    fn read_buffer(
        &self,
        b: &TestBuffer,
        o: u64,
        s: &mut dyn ByteSink,
    ) -> Result<(), BackendError> {
        assert_eq!(o, 0);
        s.write_at(0, &b.bytes.borrow())
    }
    fn free_buffer(&self, _: TestBuffer) -> Result<(), ReleaseFailure<TestBuffer>> {
        self.0.borrow_mut().frees += 1;
        Ok(())
    }
    fn load_program(&self, _: &Handle, _: ArtifactRef<'_>) -> Result<Handle, BackendError> {
        self.0.borrow_mut().loads += 1;
        Ok(self.handle())
    }
    fn unload_program(&self, _: Handle) -> Result<(), ReleaseFailure<Handle>> {
        self.0.borrow_mut().unloads += 1;
        Ok(())
    }
    fn create_queue(&self, _: &Handle, _: QueueDesc) -> Result<Handle, BackendError> {
        Ok(self.handle())
    }
    fn destroy_queue(&self, _: Handle) -> Result<(), ReleaseFailure<Handle>> {
        self.0.borrow_mut().queues += 1;
        Ok(())
    }
    fn submit(
        &self,
        _: &Handle,
        _: &Handle,
        bindings: &[BindingRef<'_, TestBuffer>],
        _: Timeout,
    ) -> Result<Handle, SubmitFailure<Handle>> {
        let mut s = self.0.borrow_mut();
        s.submits += 1;
        if s.reject {
            return Err(SubmitFailure::Rejected(BackendError::Busy));
        }
        bindings[1]
            .buffer
            .bytes
            .borrow_mut()
            .copy_from_slice(&bindings[0].buffer.bytes.borrow());
        let e = self.handle();
        if s.indeterminate {
            Err(SubmitFailure::Indeterminate {
                error: BackendError::DeviceLost,
                event: e,
            })
        } else {
            Ok(e)
        }
    }
    fn poll_event(&self, _: &Handle) -> Result<EventState, BackendError> {
        let mut s = self.0.borrow_mut();
        s.polls += 1;
        if s.poll_error {
            Err(BackendError::DeviceLost)
        } else if s.polls % 2 == 1 {
            Ok(EventState::Pending)
        } else {
            Ok(EventState::Complete)
        }
    }
    fn destroy_event(&self, _: Handle) -> Result<(), ReleaseFailure<Handle>> {
        self.0.borrow_mut().events += 1;
        Ok(())
    }
}
fn cap() -> CapabilityDescriptor {
    CapabilityDescriptor {
        target: Target::new(
            Version::TOSA_1_0,
            ProfileSet::FLOATING_POINT,
            Level::Level8K,
            ExtensionSet::NONE,
        ),
        dtypes: &[],
        operators: &[],
        graph: virtio_accel_tosa::GraphCapabilities {
            max_regions: 1,
            max_blocks: 1,
            dynamic_shapes: false,
            runtime_conditions: virtio_accel_tosa::RuntimeConditionSupport::None,
        },
    }
}
fn job(n: usize, value: u8) -> Job {
    let shape = [n as i32];
    let tensors = [
        Tensor::new("x", &shape, DType::FP32),
        Tensor::new("y", &shape, DType::FP32),
    ];
    let ops = [Operator::new(OperatorKind::Identity, &["x"], &["y"])];
    let artifact = Graph::new("test", &tensors, &ops, &["x"], &["y"])
        .build(cap().target)
        .unwrap();
    Job {
        artifact,
        inputs: vec![vec![value; n * 4]],
        output_sizes: vec![n * 4],
    }
}
#[test]
fn cache_reuses_program_but_refreshes_inputs_and_releases_shape_changes() {
    let state = Rc::new(RefCell::new(State::default()));
    let mut s = Session::new(TestAccel(state.clone())).unwrap();
    assert_eq!(s.execute(job(2, 3), cap()).unwrap(), vec![vec![3; 8]]);
    assert_eq!(s.execute(job(2, 7), cap()).unwrap(), vec![vec![7; 8]]);
    assert_eq!(state.borrow().loads, 1);
    assert_eq!(state.borrow().allocs, 2);
    assert_eq!(s.execute(job(3, 9), cap()).unwrap(), vec![vec![9; 12]]);
    assert_eq!(state.borrow().loads, 2);
    assert_eq!(state.borrow().unloads, 1);
    drop(s);
    let s = state.borrow();
    assert_eq!(
        (
            s.allocs, s.frees, s.loads, s.unloads, s.events, s.queues, s.contexts
        ),
        (4, 4, 2, 2, 3, 1, 1)
    );
}
#[test]
fn partial_allocation_failure_cleans_up_before_retry() {
    let state = Rc::new(RefCell::new(State {
        allocation_error: true,
        ..Default::default()
    }));
    let mut s = Session::new(TestAccel(state.clone())).unwrap();
    assert!(s.execute(job(2, 3), cap()).is_err());
    assert_eq!((state.borrow().frees, state.borrow().unloads), (1, 1));
    state.borrow_mut().allocation_error = false;
    assert!(s.execute(job(2, 3), cap()).is_ok());
    drop(s);
    assert_eq!(state.borrow().allocs, state.borrow().frees);
}
#[test]
fn rejected_submission_has_no_event_and_can_be_retried() {
    let state = Rc::new(RefCell::new(State {
        reject: true,
        ..Default::default()
    }));
    let mut s = Session::new(TestAccel(state.clone())).unwrap();
    assert!(s.execute(job(2, 3), cap()).is_err());
    assert_eq!(state.borrow().polls, 0);
    state.borrow_mut().reject = false;
    assert!(s.execute(job(2, 7), cap()).is_ok());
    assert_eq!(state.borrow().loads, 1);
}
#[test]
fn indeterminate_submission_is_drained_without_retry_or_output_publication() {
    let state = Rc::new(RefCell::new(State {
        indeterminate: true,
        ..Default::default()
    }));
    let mut s = Session::new(TestAccel(state.clone())).unwrap();
    assert!(s.execute(job(2, 3), cap()).is_err());
    assert_eq!(
        (
            state.borrow().submits,
            state.borrow().polls,
            state.borrow().events
        ),
        (1, 2, 1)
    );
    drop(s);
    assert_eq!(state.borrow().allocs, state.borrow().frees);
}
#[test]
fn failed_poll_retains_resources_and_prevents_resubmission() {
    let state = Rc::new(RefCell::new(State {
        poll_error: true,
        ..Default::default()
    }));
    let mut s = Session::new(TestAccel(state.clone())).unwrap();
    assert!(s.execute(job(2, 3), cap()).is_err());
    assert!(s.execute(job(2, 3), cap()).is_err());
    drop(s);
    let s = state.borrow();
    assert_eq!(s.submits, 1);
    assert_eq!(s.handle_drops, 0);
    assert_eq!(s.frees, 0);
    assert_eq!(s.contexts, 0);
}
