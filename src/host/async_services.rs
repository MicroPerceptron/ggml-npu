//! Focused construction ports for instance-local async services.

use crate::runtime::async_runtime::{
    BlockingExecutorPort, CallerDrivenBlockingExecutor, ProgressDriver, WaitError,
};

pub(crate) trait ReactorFactory {
    type Reactor: ProgressDriver;

    fn create(&self) -> Result<Self::Reactor, WaitError>;
}

pub(crate) trait BlockingExecutorFactory {
    type Executor: BlockingExecutorPort;

    fn create(&self, capacity: usize) -> Result<Self::Executor, WaitError>;
}

#[derive(Clone, Copy, Default)]
pub(crate) struct CallerDrivenBlockingFactory;

impl BlockingExecutorFactory for CallerDrivenBlockingFactory {
    type Executor = CallerDrivenBlockingExecutor;

    fn create(&self, capacity: usize) -> Result<Self::Executor, WaitError> {
        Ok(CallerDrivenBlockingExecutor::new(capacity))
    }
}
