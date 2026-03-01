#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    Recoverable,
    Fatal,
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveryPolicy {
    pub checkpoint_interval_steps: u64,
    pub max_retries: u8,
}

impl RecoveryPolicy {
    pub fn should_retry(self, retries: u8, failure: FailureClass) -> bool {
        matches!(failure, FailureClass::Recoverable) && retries < self.max_retries
    }
}

