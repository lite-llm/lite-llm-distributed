use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedError {
    InvalidTopology(&'static str),
    RankOutOfRange {
        rank: usize,
        world_size: usize,
    },
    InvalidCollectiveInput(&'static str),
    TagOrderViolation {
        from_rank: usize,
        to_rank: usize,
    },
    MissingMessage {
        from_rank: usize,
        to_rank: usize,
    },
    ConsensusMismatch {
        expected: u64,
        found: u64,
        rank: usize,
    },
    InvalidFailureEvent(&'static str),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTopology(msg) => write!(f, "invalid topology: {msg}"),
            Self::RankOutOfRange { rank, world_size } => {
                write!(f, "rank {rank} out of range for world size {world_size}")
            }
            Self::InvalidCollectiveInput(msg) => write!(f, "invalid collective input: {msg}"),
            Self::TagOrderViolation { from_rank, to_rank } => write!(
                f,
                "tag order violation for pair ({from_rank} -> {to_rank})"
            ),
            Self::MissingMessage { from_rank, to_rank } => {
                write!(f, "missing message from rank {from_rank} to rank {to_rank}")
            }
            Self::ConsensusMismatch {
                expected,
                found,
                rank,
            } => write!(
                f,
                "routing consensus mismatch at rank {rank}: expected checksum {expected}, found {found}"
            ),
            Self::InvalidFailureEvent(msg) => write!(f, "invalid failure event: {msg}"),
        }
    }
}

impl Error for DistributedError {}

pub type DistributedResult<T> = Result<T, DistributedError>;
