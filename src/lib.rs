pub mod collectives;
pub mod fault_tolerance;
pub mod parallelism;
pub mod transport;

pub use collectives::{CollectiveError, CollectiveOps};
pub use fault_tolerance::{FailureClass, RecoveryPolicy};
pub use parallelism::ParallelismConfig;
pub use transport::{TransportBackend, TransportConfig};

