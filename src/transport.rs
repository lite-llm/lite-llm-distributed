use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::error::{DistributedError, DistributedResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportBackend {
    Rdma,
    Nccl,
    Quic,
}

#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub backend: TransportBackend,
    pub endpoint: String,
    pub world_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MessagePhase {
    Dispatch,
    Return,
    Collective,
    Heartbeat,
    Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MessageTag {
    pub step: u64,
    pub layer: u32,
    pub phase: MessagePhase,
    pub sequence: u32,
}

impl MessageTag {
    pub const fn new(step: u64, layer: u32, phase: MessagePhase, sequence: u32) -> Self {
        Self {
            step,
            layer,
            phase,
            sequence,
        }
    }
}

pub trait Transport {
    fn send(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()>;
    fn recv(&self, to_rank: usize, from_rank: usize, tag: MessageTag) -> DistributedResult<Vec<u8>>;
    fn barrier(&self, rank: usize, tag: MessageTag) -> DistributedResult<()>;
}

#[derive(Debug, Default)]
struct TransportState {
    queues: BTreeMap<(usize, usize, MessageTag), VecDeque<Vec<u8>>>,
    last_sent_tag: HashMap<(usize, usize), MessageTag>,
    barriers: BTreeMap<MessageTag, HashSet<usize>>,
}

#[derive(Debug, Clone)]
pub struct InMemoryTaggedTransport {
    world_size: usize,
    state: Arc<Mutex<TransportState>>,
}

impl InMemoryTaggedTransport {
    pub fn new(world_size: usize) -> DistributedResult<Self> {
        if world_size == 0 {
            return Err(DistributedError::InvalidTopology(
                "world_size must be greater than zero",
            ));
        }

        Ok(Self {
            world_size,
            state: Arc::new(Mutex::new(TransportState::default())),
        })
    }

    fn validate_rank(&self, rank: usize) -> DistributedResult<()> {
        if rank >= self.world_size {
            return Err(DistributedError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        Ok(())
    }
}

impl Transport for InMemoryTaggedTransport {
    fn send(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        self.validate_rank(from_rank)?;
        self.validate_rank(to_rank)?;

        let mut state = self.state.lock().expect("transport mutex poisoned");
        if let Some(last) = state.last_sent_tag.get(&(from_rank, to_rank)) {
            if tag <= *last {
                return Err(DistributedError::TagOrderViolation { from_rank, to_rank });
            }
        }

        state.last_sent_tag.insert((from_rank, to_rank), tag);
        state
            .queues
            .entry((from_rank, to_rank, tag))
            .or_default()
            .push_back(payload);

        Ok(())
    }

    fn recv(&self, to_rank: usize, from_rank: usize, tag: MessageTag) -> DistributedResult<Vec<u8>> {
        self.validate_rank(to_rank)?;
        self.validate_rank(from_rank)?;

        let mut state = self.state.lock().expect("transport mutex poisoned");
        let key = (from_rank, to_rank, tag);
        let queue = state
            .queues
            .get_mut(&key)
            .ok_or(DistributedError::MissingMessage { from_rank, to_rank })?;

        let payload = queue
            .pop_front()
            .ok_or(DistributedError::MissingMessage { from_rank, to_rank })?;
        if queue.is_empty() {
            state.queues.remove(&key);
        }

        Ok(payload)
    }

    fn barrier(&self, rank: usize, tag: MessageTag) -> DistributedResult<()> {
        self.validate_rank(rank)?;

        let mut state = self.state.lock().expect("transport mutex poisoned");
        let participants = state.barriers.entry(tag).or_default();
        participants.insert(rank);

        if participants.len() == self.world_size {
            state.barriers.remove(&tag);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryTaggedTransport, MessagePhase, MessageTag, Transport};

    #[test]
    fn tagged_send_receive_roundtrip() {
        let transport = InMemoryTaggedTransport::new(2).expect("transport should initialize");
        let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);

        transport
            .send(0, 1, tag, b"hello".to_vec())
            .expect("send should succeed");
        let payload = transport
            .recv(1, 0, tag)
            .expect("receive should succeed");

        assert_eq!(payload, b"hello".to_vec());
    }

    #[test]
    fn send_requires_monotonic_tag_order() {
        let transport = InMemoryTaggedTransport::new(2).expect("transport should initialize");
        let first = MessageTag::new(1, 0, MessagePhase::Dispatch, 1);
        let second = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);

        transport
            .send(0, 1, first, b"ok".to_vec())
            .expect("first send should succeed");
        assert!(transport.send(0, 1, second, b"bad".to_vec()).is_err());
    }
}
