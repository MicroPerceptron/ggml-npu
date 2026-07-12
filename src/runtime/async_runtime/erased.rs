use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr::NonNull;
use core::task::{Context, Poll, Waker};

use super::completion::SpinMutex;
use super::{
    CancellationRequest, Completion, CompletionError, CompletionInfo, CompletionResult,
    CompletionSource, DeviceEpoch, OperationId, Progress,
};

const DYNAMIC_DOMAIN: u32 = u32::from_be_bytes(*b"DYN ");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum PollAbiStatus {
    Pending = 0,
    Ready = 1,
}

pub type ForeignWakeCallback = extern "C" fn(*mut c_void);

/// Primitive-only completion result crossing the C ABI.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CompletionAbiResult {
    /// 0 success, 1 execution failure, 2 cancelled, 3 device lost.
    pub status: u32,
    pub domain: u32,
    pub code: i64,
    pub epoch: u64,
    pub queue_sequence: u64,
    pub reason: u32,
    pub may_have_executed: u8,
    pub reserved: [u8; 3],
}

impl CompletionAbiResult {
    pub const fn success(epoch: DeviceEpoch, queue_sequence: u64) -> Self {
        Self {
            status: 0,
            domain: 0,
            code: 0,
            epoch: epoch.get(),
            queue_sequence,
            reason: 0,
            may_have_executed: 0,
            reserved: [0; 3],
        }
    }
}

/// Versioned provider table. Rust trait objects, futures, and wakers never cross this boundary.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CompletionAbiVTable {
    pub abi_version: u32,
    pub retain: unsafe extern "C" fn(context: *mut c_void),
    pub release: unsafe extern "C" fn(context: *mut c_void),
    /// Returns [`PollAbiStatus`] or a negative adapter error.
    pub try_poll:
        unsafe extern "C" fn(context: *mut c_void, output: *mut CompletionAbiResult) -> i32,
    /// Registers a C callback and returns an opaque registration token.
    pub register_wake: unsafe extern "C" fn(
        context: *mut c_void,
        callback: ForeignWakeCallback,
        callback_context: *mut c_void,
        registration: *mut u64,
    ) -> i32,
    /// Success guarantees no callback using that registration is running or can run afterward.
    pub unregister_wake: unsafe extern "C" fn(context: *mut c_void, registration: u64) -> i32,
    /// 0 requested, 1 already requested, 2 terminal, 3 unsupported, negative adapter error.
    pub request_cancel: unsafe extern "C" fn(context: *mut c_void) -> i32,
}

struct WakeBridge {
    waker: SpinMutex<Option<Waker>>,
}

extern "C" fn foreign_wake(context: *mut c_void) {
    if context.is_null() {
        return;
    }
    // SAFETY: the bridge remains allocated until unregister has quiesced this callback.
    let bridge = unsafe { &*context.cast::<WakeBridge>() };
    let waker = bridge.waker.lock().clone();
    if let Some(waker) = waker {
        waker.wake();
    }
}

struct ForeignRegistration {
    observer: u64,
    token: u64,
    bridge: NonNull<WakeBridge>,
}

struct ErasedSource {
    context: NonNull<c_void>,
    vtable: &'static CompletionAbiVTable,
    operation_id: OperationId,
    epoch: DeviceEpoch,
    registrations: SpinMutex<Vec<ForeignRegistration>>,
    adapter_failure: SpinMutex<Option<CompletionResult>>,
}

// SAFETY: the ABI contract requires provider contexts and vtable functions to be thread-safe.
unsafe impl Send for ErasedSource {}
unsafe impl Sync for ErasedSource {}

impl ErasedSource {
    fn decode(&self, result: CompletionAbiResult) -> CompletionResult {
        let epoch = DeviceEpoch::new(result.epoch).unwrap_or(self.epoch);
        match result.status {
            0 => Ok(CompletionInfo {
                operation_id: self.operation_id,
                device_epoch: epoch,
                queue_sequence: result.queue_sequence,
                progress: Progress::default(),
            }),
            1 => Err(CompletionError::ExecutionFailed {
                domain: result.domain,
                code: result.code,
            }),
            2 => Err(CompletionError::Cancelled {
                may_have_executed: result.may_have_executed != 0,
            }),
            3 => Err(CompletionError::DeviceLost {
                epoch,
                reason: result.reason,
            }),
            _ => Err(CompletionError::ExecutionFailed {
                domain: DYNAMIC_DOMAIN,
                code: i64::from(result.status),
            }),
        }
    }

    fn latch_adapter_failure(&self, code: i32) {
        let inserted = {
            let mut failure = self.adapter_failure.lock();
            if failure.is_some() {
                false
            } else {
                *failure = Some(Err(CompletionError::ExecutionFailed {
                    domain: DYNAMIC_DOMAIN,
                    code: i64::from(code),
                }));
                true
            }
        };
        if inserted {
            let wakers = self
                .registrations
                .lock()
                .iter()
                .filter_map(|registration| {
                    // SAFETY: registrations own their bridges until successful unregister.
                    unsafe { registration.bridge.as_ref() }.waker.lock().clone()
                })
                .collect::<Vec<_>>();
            for waker in wakers {
                waker.wake();
            }
        }
    }
}

impl CompletionSource for ErasedSource {
    fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    fn device_epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    fn try_complete(&self) -> Option<CompletionResult> {
        if let Some(failure) = *self.adapter_failure.lock() {
            return Some(failure);
        }
        let mut output = MaybeUninit::<CompletionAbiResult>::uninit();
        // SAFETY: context and table were validated by the constructor; output is decoded only when
        // the provider reports Ready.
        let status = unsafe { (self.vtable.try_poll)(self.context.as_ptr(), output.as_mut_ptr()) };
        if status == PollAbiStatus::Pending as i32 {
            None
        } else if status == PollAbiStatus::Ready as i32 {
            // SAFETY: Ready requires the provider to initialize the entire repr(C) output.
            Some(self.decode(unsafe { output.assume_init() }))
        } else {
            self.latch_adapter_failure(status);
            *self.adapter_failure.lock()
        }
    }

    fn register_waker(&self, observer: u64, waker: &Waker) {
        let mut registrations = self.registrations.lock();
        if let Some(registration) = registrations
            .iter_mut()
            .find(|registration| registration.observer == observer)
        {
            // SAFETY: registration owns this bridge until unregister.
            let bridge = unsafe { registration.bridge.as_ref() };
            *bridge.waker.lock() = Some(waker.clone());
            return;
        }

        let bridge = Box::new(WakeBridge {
            waker: SpinMutex::new(Some(waker.clone())),
        });
        let bridge = NonNull::from(Box::leak(bridge));
        let mut token = 0;
        // SAFETY: all pointers remain live through the call; successful registration retains only
        // the callback context under the table's unregister contract.
        let status = unsafe {
            (self.vtable.register_wake)(
                self.context.as_ptr(),
                foreign_wake,
                bridge.as_ptr().cast(),
                &mut token,
            )
        };
        if status == 0 {
            registrations.push(ForeignRegistration {
                observer,
                token,
                bridge,
            });
        } else {
            // SAFETY: failed registration did not retain the callback context.
            drop(unsafe { Box::from_raw(bridge.as_ptr()) });
            drop(registrations);
            self.latch_adapter_failure(status);
        }
    }

    fn unregister_waker(&self, observer: u64) {
        let registration = {
            let mut registrations = self.registrations.lock();
            registrations
                .iter()
                .position(|registration| registration.observer == observer)
                .map(|index| registrations.swap_remove(index))
        };
        let Some(registration) = registration else {
            return;
        };
        // SAFETY: token belongs to this context. A zero return quiesces the callback.
        let status =
            unsafe { (self.vtable.unregister_wake)(self.context.as_ptr(), registration.token) };
        if status == 0 {
            // SAFETY: callback is quiesced and this registration uniquely owns the bridge.
            drop(unsafe { Box::from_raw(registration.bridge.as_ptr()) });
        } else {
            // Leak on a broken unregister contract rather than permit callback use-after-free.
            self.latch_adapter_failure(status);
        }
    }

    fn request_cancel(&self) -> CancellationRequest {
        // SAFETY: context and vtable remain retained by this source.
        match unsafe { (self.vtable.request_cancel)(self.context.as_ptr()) } {
            0 => CancellationRequest::Requested,
            1 => CancellationRequest::AlreadyRequested,
            2 => CancellationRequest::AlreadyTerminal,
            3 => CancellationRequest::Unsupported,
            code => {
                self.latch_adapter_failure(code);
                CancellationRequest::Unsupported
            }
        }
    }
}

impl Drop for ErasedSource {
    fn drop(&mut self) {
        let registrations = self.registrations.lock().drain(..).collect::<Vec<_>>();
        for registration in registrations {
            // SAFETY: registrations and context were created by this vtable.
            let status =
                unsafe { (self.vtable.unregister_wake)(self.context.as_ptr(), registration.token) };
            if status == 0 {
                // SAFETY: successful unregister quiesced the callback.
                drop(unsafe { Box::from_raw(registration.bridge.as_ptr()) });
            }
        }
        // SAFETY: constructor retained one provider reference for this source.
        unsafe { (self.vtable.release)(self.context.as_ptr()) };
    }
}

/// Safe Rust future owning one retained foreign completion source.
pub struct ErasedCompletion {
    completion: Completion,
}

impl ErasedCompletion {
    /// # Safety
    ///
    /// `context` and `vtable` must implement the documented thread-safe retain/release and callback
    /// quiescence contracts for the lifetime of the retained context.
    pub unsafe fn from_raw(
        context: NonNull<c_void>,
        vtable: &'static CompletionAbiVTable,
        operation_id: OperationId,
        epoch: DeviceEpoch,
    ) -> Self {
        unsafe { (vtable.retain)(context.as_ptr()) };
        let source = Arc::new(ErasedSource {
            context,
            vtable,
            operation_id,
            epoch,
            registrations: SpinMutex::new(Vec::new()),
            adapter_failure: SpinMutex::new(None),
        });
        Self {
            completion: Completion::from_source(source),
        }
    }

    pub fn completion(&self) -> &Completion {
        &self.completion
    }

    pub fn into_completion(self) -> Completion {
        self.completion
    }
}

impl Future for ErasedCompletion {
    type Output = CompletionResult;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.completion).poll(context)
    }
}
