//! gRPC service implementation for the `TransportService` defined in `proto/transport.proto`.
//!
//! Each node in the distributed cluster runs a `TransportServiceServer` backed by
//! a shared `TransportState`. The service handles incoming `SendMessage`, `RecvMessage`,
//! `Barrier`, and `HealthCheck` RPCs from peer nodes.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::error::{DistributedError, DistributedResult};
use crate::transport::MessagePhase;

// Re-export the tonic-generated module.
include!(concat!(env!("OUT_DIR"), "/transport.rs"));

/// Convert a `transport::MessageTag` (prost-generated) to the crate's `MessageTag`.
pub fn proto_tag_to_crate_tag(tag: &MessageTag) -> crate::transport::MessageTag {
    crate::transport::MessageTag {
        step: tag.step,
        layer: tag.layer,
        phase: match MessagePhase::try_from(tag.phase).unwrap_or(MessagePhase::Dispatch) {
            MessagePhase::Dispatch => crate::transport::MessagePhase::Dispatch,
            MessagePhase::Return => crate::transport::MessagePhase::Return,
            MessagePhase::Collective => crate::transport::MessagePhase::Collective,
            MessagePhase::Heartbeat => crate::transport::MessagePhase::Heartbeat,
            MessagePhase::Control => crate::transport::MessagePhase::Control,
        },
        sequence: tag.sequence,
    }
}

/// Convert the crate's `MessageTag` to the prost-generated `MessageTag`.
pub fn crate_tag_to_proto_tag(tag: crate::transport::MessageTag) -> MessageTag {
    let phase = match tag.phase {
        crate::transport::MessagePhase::Dispatch => MessagePhase::Dispatch,
        crate::transport::MessagePhase::Return => MessagePhase::Return,
        crate::transport::MessagePhase::Collective => MessagePhase::Collective,
        crate::transport::MessagePhase::Heartbeat => MessagePhase::Heartbeat,
        crate::transport::MessagePhase::Control => MessagePhase::Control,
    } as i32;

    MessageTag {
        step: tag.step,
        layer: tag.layer,
        phase,
        sequence: tag.sequence,
    }
}

/// Shared state between the gRPC server and the transport layer.
#[derive(Debug, Default)]
pub struct TransportState {
    /// Incoming message queues: (from_rank, to_rank, tag) -> queue of payloads.
    queues: BTreeMap<(usize, usize, crate::transport::MessageTag), VecDeque<Vec<u8>>>,
    /// Monotonic tag tracking per (from, to) pair.
    last_sent_tag: HashMap<(usize, usize), crate::transport::MessageTag>,
    /// Barrier participants: tag -> set of ranks that have arrived.
    barriers: BTreeMap<crate::transport::MessageTag, HashSet<usize>>,
    /// World size for barrier coordination.
    world_size: usize,
}

impl TransportState {
    pub fn new(world_size: usize) -> Self {
        Self {
            queues: BTreeMap::new(),
            last_sent_tag: HashMap::new(),
            barriers: BTreeMap::new(),
            world_size,
        }
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Enqueue a message for local delivery.
    pub fn enqueue(
        &mut self,
        from_rank: usize,
        to_rank: usize,
        tag: crate::transport::MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        if let Some(last) = self.last_sent_tag.get(&(from_rank, to_rank)) {
            if tag <= *last {
                return Err(DistributedError::TagOrderViolation { from_rank, to_rank });
            }
        }
        self.last_sent_tag.insert((from_rank, to_rank), tag);
        self.queues.entry((from_rank, to_rank, tag)).or_default().push_back(payload);
        Ok(())
    }

    /// Dequeue a message for local delivery.
    pub fn dequeue(
        &mut self,
        to_rank: usize,
        from_rank: usize,
        tag: crate::transport::MessageTag,
    ) -> DistributedResult<Vec<u8>> {
        let key = (from_rank, to_rank, tag);
        let queue = self
            .queues
            .get_mut(&key)
            .ok_or(DistributedError::MissingMessage { from_rank, to_rank })?;

        let payload = queue
            .pop_front()
            .ok_or(DistributedError::MissingMessage { from_rank, to_rank })?;
        if queue.is_empty() {
            self.queues.remove(&key);
        }
        Ok(payload)
    }

    /// Register a rank arrival at a barrier.
    /// Returns `true` if all ranks have arrived (barrier cleared).
    pub fn barrier_arrive(&mut self, rank: usize, tag: crate::transport::MessageTag) -> bool {
        let participants = self.barriers.entry(tag).or_default();
        participants.insert(rank);
        if participants.len() >= self.world_size {
            self.barriers.remove(&tag);
            true
        } else {
            false
        }
    }
}

/// The actual gRPC service implementation.
pub struct GrpcServiceImpl {
    pub(crate) state: Arc<Mutex<TransportState>>,
    pub(crate) local_rank: usize,
    pub(crate) world_size: usize,
    pub(crate) listen_addr: String,
    pub(crate) start_time: Instant,
}

impl GrpcServiceImpl {
    pub fn new(
        state: Arc<Mutex<TransportState>>,
        local_rank: usize,
        world_size: usize,
        listen_addr: String,
    ) -> Self {
        Self {
            state,
            local_rank,
            world_size,
            listen_addr,
            start_time: Instant::now(),
        }
    }
}

#[tonic::async_trait]
impl transport_service_server::TransportService for GrpcServiceImpl {
    async fn send_message(
        &self,
        request: Request<SendMessageRequest>,
    ) -> Result<Response<SendMessageResponse>, Status> {
        let req = request.into_inner();

        let tag = proto_tag_to_crate_tag(req.tag.as_ref().ok_or_else(|| {
            Status::invalid_argument("MessageTag is required")
        })?);

        let from_rank = req.from_rank as usize;
        let to_rank = req.to_rank as usize;

        let mut state = self.state.lock().await;
        match state.enqueue(from_rank, to_rank, tag, req.payload) {
            Ok(()) => Ok(Response::new(SendMessageResponse {
                ok: true,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(SendMessageResponse {
                ok: false,
                error: e.to_string(),
            })),
        }
    }

    async fn recv_message(
        &self,
        request: Request<RecvMessageRequest>,
    ) -> Result<Response<RecvMessageResponse>, Status> {
        let req = request.into_inner();

        let tag = proto_tag_to_crate_tag(req.tag.as_ref().ok_or_else(|| {
            Status::invalid_argument("MessageTag is required")
        })?);

        let to_rank = req.to_rank as usize;
        let from_rank = req.from_rank as usize;

        let mut state = self.state.lock().await;
        match state.dequeue(to_rank, from_rank, tag) {
            Ok(payload) => Ok(Response::new(RecvMessageResponse {
                ok: true,
                payload,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(RecvMessageResponse {
                ok: false,
                payload: Vec::new(),
                error: e.to_string(),
            })),
        }
    }

    async fn barrier(
        &self,
        request: Request<BarrierRequest>,
    ) -> Result<Response<BarrierResponse>, Status> {
        let req = request.into_inner();

        let tag = proto_tag_to_crate_tag(req.tag.as_ref().ok_or_else(|| {
            Status::invalid_argument("MessageTag is required")
        })?);

        let rank = req.rank as usize;
        // Use the world_size from the local state (authoritative), not from the request.
        // We still validate that the request's world_size matches ours.
        if req.world_size as usize != self.world_size {
            return Ok(Response::new(BarrierResponse {
                ok: false,
                error: format!(
                    "world_size mismatch: request has {}, local has {}",
                    req.world_size, self.world_size
                ),
            }));
        }

        let mut state = self.state.lock().await;
        let _all_arrived = state.barrier_arrive(rank, tag);

        // If all ranks arrived, the barrier is cleared on this node.
        // The caller can proceed.
        Ok(Response::new(BarrierResponse {
            ok: true,
            error: String::new(),
        }))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let uptime_secs = self.start_time.elapsed().as_secs();
        Ok(Response::new(HealthCheckResponse {
            status: HealthStatus::Serving as i32,
            local_rank: self.local_rank as u32,
            world_size: self.world_size as u32,
            listen_addr: self.listen_addr.clone(),
            uptime_secs,
        }))
    }
}
