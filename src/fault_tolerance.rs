use std::collections::{BTreeMap, BTreeSet};

use crate::error::{DistributedError, DistributedResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureDomain {
    Process { rank: usize },
    Node { node_id: u32 },
    Network,
    Device { rank: usize },
    Storage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FailureClass {
    Transient,
    Recoverable,
    ProcessFailure,
    NodeFailure,
    NetworkPartition,
    DeviceError,
    StorageError,
    Fatal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureEvent {
    pub step: u64,
    pub class: FailureClass,
    pub domain: FailureDomain,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    RetryAfter { millis: u64 },
    ReloadFromCheckpoint,
    ReinitializeTransport,
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryPolicy {
    pub checkpoint_interval_steps: u64,
    pub max_retries: u8,
    pub base_backoff_millis: u64,
    pub heartbeat_timeout_steps: u64,
}

impl RecoveryPolicy {
    pub fn should_retry(self, retries: u8, class: FailureClass) -> bool {
        matches!(class, FailureClass::Transient | FailureClass::Recoverable)
            && retries < self.max_retries
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryCoordinator {
    policy: RecoveryPolicy,
    failure_retries: BTreeMap<(u64, FailureClass), u8>,
    failed_ranks: BTreeSet<usize>,
    last_heartbeat_step: BTreeMap<usize, u64>,
}

impl RecoveryCoordinator {
    pub fn new(policy: RecoveryPolicy) -> Self {
        Self {
            policy,
            failure_retries: BTreeMap::new(),
            failed_ranks: BTreeSet::new(),
            last_heartbeat_step: BTreeMap::new(),
        }
    }

    pub fn record_heartbeat(&mut self, rank: usize, step: u64) {
        self.last_heartbeat_step.insert(rank, step);
    }

    pub fn detect_timeouts(&self, current_step: u64) -> Vec<usize> {
        let mut timed_out = Vec::new();
        for (rank, step) in &self.last_heartbeat_step {
            if current_step.saturating_sub(*step) > self.policy.heartbeat_timeout_steps {
                timed_out.push(*rank);
            }
        }
        timed_out
    }

    pub fn failed_ranks(&self) -> &BTreeSet<usize> {
        &self.failed_ranks
    }

    pub fn handle_failure(&mut self, event: &FailureEvent) -> DistributedResult<RecoveryAction> {
        if event.description.is_empty() {
            return Err(DistributedError::InvalidFailureEvent(
                "failure description cannot be empty",
            ));
        }

        let key = (event.step, event.class);
        let retries = self.failure_retries.get(&key).copied().unwrap_or(0);

        let action = if self.policy.should_retry(retries, event.class) {
            self.failure_retries.insert(key, retries.saturating_add(1));
            RecoveryAction::RetryAfter {
                millis: self
                    .policy
                    .base_backoff_millis
                    .saturating_mul(2_u64.saturating_pow(retries as u32)),
            }
        } else {
            match event.class {
                FailureClass::Transient | FailureClass::Recoverable => {
                    RecoveryAction::ReloadFromCheckpoint
                }
                FailureClass::ProcessFailure
                | FailureClass::NodeFailure
                | FailureClass::NetworkPartition
                | FailureClass::DeviceError
                | FailureClass::StorageError => RecoveryAction::ReinitializeTransport,
                FailureClass::Fatal => RecoveryAction::Abort,
            }
        };

        match event.domain {
            FailureDomain::Process { rank } | FailureDomain::Device { rank } => {
                self.failed_ranks.insert(rank);
            }
            FailureDomain::Node { .. } | FailureDomain::Network | FailureDomain::Storage => {}
        }

        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FailureClass, FailureDomain, FailureEvent, RecoveryAction, RecoveryCoordinator, RecoveryPolicy,
    };

    fn policy() -> RecoveryPolicy {
        RecoveryPolicy {
            checkpoint_interval_steps: 100,
            max_retries: 2,
            base_backoff_millis: 10,
            heartbeat_timeout_steps: 5,
        }
    }

    #[test]
    fn transient_failure_retries_then_checkpoint_reload() {
        let mut coordinator = RecoveryCoordinator::new(policy());
        let event = FailureEvent {
            step: 7,
            class: FailureClass::Transient,
            domain: FailureDomain::Network,
            description: "packet loss".to_owned(),
        };

        assert_eq!(
            coordinator.handle_failure(&event).expect("retry 1"),
            RecoveryAction::RetryAfter { millis: 10 }
        );
        assert_eq!(
            coordinator.handle_failure(&event).expect("retry 2"),
            RecoveryAction::RetryAfter { millis: 20 }
        );
        assert_eq!(
            coordinator.handle_failure(&event).expect("fallback"),
            RecoveryAction::ReloadFromCheckpoint
        );
    }

    #[test]
    fn process_failure_marks_failed_rank() {
        let mut coordinator = RecoveryCoordinator::new(policy());
        let event = FailureEvent {
            step: 9,
            class: FailureClass::ProcessFailure,
            domain: FailureDomain::Process { rank: 3 },
            description: "worker crashed".to_owned(),
        };

        let action = coordinator
            .handle_failure(&event)
            .expect("handler should succeed");
        assert_eq!(action, RecoveryAction::ReinitializeTransport);
        assert!(coordinator.failed_ranks().contains(&3));
    }

    #[test]
    fn heartbeat_timeout_detection_is_deterministic() {
        let mut coordinator = RecoveryCoordinator::new(policy());
        coordinator.record_heartbeat(0, 10);
        coordinator.record_heartbeat(1, 15);

        let timed_out = coordinator.detect_timeouts(20);
        assert_eq!(timed_out, vec![0]);
    }
}


