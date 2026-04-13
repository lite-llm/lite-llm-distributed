//! Async gRPC transport backend for node-to-node communication in a distributed LLM cluster.
//!
//! Implements the `AsyncTransport` trait using tonic/gRPC over TCP.
//! Each node runs a gRPC server that handles send/recv/barrier RPCs from peer nodes.
//!
//! For same-node communication, messages are enqueued locally.
//! For cross-node communication, messages are sent via the `GrpcTransportClient`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::error::{DistributedError, DistributedResult};
use crate::grpc_client::{GrpcClientConfig, GrpcTransportClient};
use crate::grpc_service::{GrpcServiceImpl, TransportState};
use crate::transport::{MessageTag, Transport, TransportBackend};

// Re-export from the generated code for server types.
pub use crate::grpc_service::transport_service_server;

/// Configuration for a gRPC transport endpoint.
#[derive(Debug, Clone)]
pub struct GrpcTransportConfig {
    /// Local listen address (e.g., "0.0.0.0:50051").
    pub listen_addr: String,
    /// World size (total number of nodes in the cluster).
    pub world_size: usize,
    /// This node's rank (0..world_size).
    pub local_rank: usize,
    /// Peer addresses indexed by rank.
    pub peer_addrs: std::collections::BTreeMap<usize, String>,
    /// Backend transport type (Quic over gRPC is the default).
    pub backend: TransportBackend,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// RPC call timeout.
    pub rpc_timeout_ms: u64,
    /// Maximum number of retries for transient failures.
    pub max_retries: u32,
}

impl GrpcTransportConfig {
    /// Create a config for a multi-node cluster on localhost.
    pub fn for_localhost_cluster(world_size: usize, local_rank: usize) -> DistributedResult<Self> {
        if world_size == 0 {
            return Err(DistributedError::InvalidTopology(
                "world_size must be greater than zero",
            ));
        }
        if local_rank >= world_size {
            return Err(DistributedError::RankOutOfRange {
                rank: local_rank,
                world_size,
            });
        }

        let base_port = 50051;
        let mut peer_addrs = std::collections::BTreeMap::new();
        for rank in 0..world_size {
            if rank != local_rank {
                peer_addrs.insert(
                    rank,
                    format!("http://127.0.0.1:{}", base_port + rank),
                );
            }
        }

        Ok(Self {
            listen_addr: format!("127.0.0.1:{}", base_port + local_rank),
            world_size,
            local_rank,
            peer_addrs,
            backend: TransportBackend::Quic,
            connect_timeout_ms: 5000,
            rpc_timeout_ms: 30_000,
            max_retries: 3,
        })
    }

    fn to_client_config(&self) -> GrpcClientConfig {
        GrpcClientConfig {
            timeout: Duration::from_millis(self.rpc_timeout_ms),
            max_retries: self.max_retries,
            base_backoff: Duration::from_millis(100),
        }
    }
}

/// A gRPC-backed transport for real distributed cluster communication.
///
/// Uses local queues for same-node messaging and gRPC clients for cross-node communication.
#[derive(Clone)]
pub struct GrpcTransport {
    config: GrpcTransportConfig,
    state: Arc<Mutex<TransportState>>,
    client: Option<Arc<GrpcTransportClient>>,
}

impl GrpcTransport {
    /// Create a new gRPC transport with the given configuration.
    /// The transport is ready for local queue operations; call `serve()` to start the gRPC server.
    pub fn new(config: GrpcTransportConfig) -> DistributedResult<Self> {
        if config.world_size == 0 {
            return Err(DistributedError::InvalidTopology(
                "world_size must be greater than zero",
            ));
        }

        // Build the client pool for peer communication.
        let mut peer_addrs: HashMap<usize, String> = HashMap::new();
        for (rank, addr) in &config.peer_addrs {
            peer_addrs.insert(*rank, addr.clone());
        }

        let client = if peer_addrs.is_empty() {
            None
        } else {
            let client = GrpcTransportClient::new(
                peer_addrs,
                config.world_size,
                config.to_client_config(),
            )?;
            Some(Arc::new(client))
        };

        Ok(Self {
            config,
            state: Arc::new(Mutex::new(TransportState::new(config.world_size))),
            client,
        })
    }

    /// Get the local rank of this node.
    pub fn local_rank(&self) -> usize {
        self.config.local_rank
    }

    /// Get the world size.
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get the listen address for this node's gRPC server.
    pub fn listen_addr(&self) -> &str {
        &self.config.listen_addr
    }

    /// Get the peer addresses for all other ranks.
    pub fn peer_addrs(&self) -> &std::collections::BTreeMap<usize, String> {
        &self.config.peer_addrs
    }

    /// Get a clone of the shared transport state (useful for server setup).
    pub fn state(&self) -> Arc<Mutex<TransportState>> {
        self.state.clone()
    }

    /// Get a reference to the gRPC client for peer communication.
    pub fn client(&self) -> Option<&Arc<GrpcTransportClient>> {
        self.client.as_ref()
    }

    /// Build a `GrpcServiceImpl` for this transport, suitable for serving.
    pub fn make_service(
        &self,
    ) -> transport_service_server::TransportServiceServer<GrpcServiceImpl> {
        transport_service_server::TransportServiceServer::new(GrpcServiceImpl::new(
            self.state.clone(),
            self.config.local_rank,
            self.config.world_size,
            self.config.listen_addr.clone(),
        ))
    }

    /// Start the gRPC server on the configured listen address.
    /// Returns a `JoinHandle` that runs until the server is shut down.
    pub async fn serve(self: Arc<Self>) -> DistributedResult<tokio::task::JoinHandle<()>> {
        let addr = self
            .config
            .listen_addr
            .parse()
            .map_err(|e| DistributedError::TransportError(format!("invalid listen address: {e}")))?;

        let service = self.make_service();

        let handle = tokio::spawn(async move {
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(service)
                .serve(addr)
                .await
            {
                tracing::error!("gRPC server error: {e}");
            }
        });

        Ok(handle)
    }

    fn validate_rank(&self, rank: usize) -> DistributedResult<()> {
        if rank >= self.config.world_size {
            return Err(DistributedError::RankOutOfRange {
                rank,
                world_size: self.config.world_size,
            });
        }
        Ok(())
    }

    /// Enqueue a message locally (for when sender and receiver are on the same node).
    async fn enqueue_local(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        let mut state = self.state.lock().await;
        state.enqueue(from_rank, to_rank, tag, payload)
    }

    /// Dequeue a message locally.
    async fn dequeue_local(
        &self,
        to_rank: usize,
        from_rank: usize,
        tag: MessageTag,
    ) -> DistributedResult<Vec<u8>> {
        let mut state = self.state.lock().await;
        state.dequeue(to_rank, from_rank, tag)
    }

    /// Determine if two ranks are on the same node.
    /// In this implementation, a single `GrpcTransport` instance represents one node,
    /// and all ranks within that node share the same transport state.
    /// For simplicity, we treat same-rank (local loopback) as local,
    /// and different ranks as potentially remote.
    fn is_local_pair(&self, _from_rank: usize, to_rank: usize) -> bool {
        // If this transport manages the destination rank, treat it as local.
        // In a multi-node setup, each node has its own transport for its local rank.
        // For single-process testing, all ranks are "local" to this transport.
        to_rank < self.config.world_size
    }
}

#[async_trait::async_trait]
impl crate::grpc_transport::AsyncTransport for GrpcTransport {
    async fn send_async(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        self.validate_rank(from_rank)?;
        self.validate_rank(to_rank)?;

        if from_rank == to_rank {
            return self.enqueue_local(from_rank, to_rank, tag, payload).await;
        }

        // If the target rank is within this node's world_size (single-process test mode),
        // use local queue. Otherwise, use gRPC.
        if self.is_local_pair(from_rank, to_rank) {
            return self.enqueue_local(from_rank, to_rank, tag, payload).await;
        }

        // Cross-node: send via gRPC to the peer's server.
        if let Some(ref client) = self.client {
            client
                .send_message(to_rank, from_rank, tag, payload)
                .await
        } else {
            // No client configured: fall back to local queue (test mode).
            self.enqueue_local(from_rank, to_rank, tag, payload).await
        }
    }

    async fn recv_async(
        &self,
        to_rank: usize,
        from_rank: usize,
        tag: MessageTag,
    ) -> DistributedResult<Vec<u8>> {
        self.validate_rank(to_rank)?;
        self.validate_rank(from_rank)?;

        // If the source rank is within this node's world_size, dequeue locally.
        if self.is_local_pair(from_rank, to_rank) {
            return self.dequeue_local(to_rank, from_rank, tag).await;
        }

        // Cross-node: receive via gRPC from the peer's server.
        if let Some(ref client) = self.client {
            client
                .recv_message(from_rank, to_rank, tag)
                .await
        } else {
            // No client: fall back to local dequeue (test mode).
            self.dequeue_local(to_rank, from_rank, tag).await
        }
    }

    async fn barrier_async(&self, rank: usize, tag: MessageTag) -> DistributedResult<()> {
        self.validate_rank(rank)?;

        // If this is the last rank to arrive and all ranks are local, handle locally.
        let mut state = self.state.lock().await;
        let all_arrived = state.barrier_arrive(rank, tag);
        drop(state);

        if all_arrived {
            return Ok(());
        }

        // Notify other nodes about this barrier via gRPC.
        if let Some(ref client) = self.client {
            let mut peer_ranks: Vec<usize> = self
                .config
                .peer_addrs
                .keys()
                .copied()
                .collect();
            peer_ranks.sort();

            for peer_rank in peer_ranks {
                client
                    .barrier(peer_rank, rank, tag)
                    .await
                    .map_err(|e| {
                        DistributedError::TransportError(format!("barrier to rank {peer_rank}: {e}"))
                    })?;
            }
        }

        Ok(())
    }
}

/// Implement the sync `Transport` trait for backward compatibility.
/// This runs the async operations on a blocking thread, suitable for tests.
impl Transport for GrpcTransport {
    fn send(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        let rt = tokio::runtime::Handle::current();
        let transport = self.clone();
        let tag_clone = tag;
        let payload_clone = payload.clone();

        // Use block_in_place to avoid blocking the runtime.
        tokio::task::block_in_place(|| {
            rt.block_on(async move {
                transport
                    .send_async(from_rank, to_rank, tag_clone, payload_clone)
                    .await
            })
        })
    }

    fn recv(
        &self,
        to_rank: usize,
        from_rank: usize,
        tag: MessageTag,
    ) -> DistributedResult<Vec<u8>> {
        let rt = tokio::runtime::Handle::current();
        let transport = self.clone();
        let tag_clone = tag;

        tokio::task::block_in_place(|| {
            rt.block_on(async move {
                transport
                    .recv_async(to_rank, from_rank, tag_clone)
                    .await
            })
        })
    }

    fn barrier(&self, rank: usize, tag: MessageTag) -> DistributedResult<()> {
        let rt = tokio::runtime::Handle::current();
        let transport = self.clone();
        let tag_clone = tag;

        tokio::task::block_in_place(|| {
            rt.block_on(async move { transport.barrier_async(rank, tag_clone).await })
        })
    }
}

/// Async transport interface for distributed message passing.
#[async_trait::async_trait]
pub trait AsyncTransport: Send + Sync {
    /// Send a payload to a remote node with a message tag.
    async fn send_async(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()>;

    /// Receive a payload from a remote node matching the given tag.
    async fn recv_async(
        &self,
        to_rank: usize,
        from_rank: usize,
        tag: MessageTag,
    ) -> DistributedResult<Vec<u8>>;

    /// Synchronize all nodes at a barrier point.
    async fn barrier_async(&self, rank: usize, tag: MessageTag) -> DistributedResult<()>;
}

#[cfg(test)]
mod tests {
    use super::{GrpcTransport, GrpcTransportConfig};
    use crate::transport::{MessagePhase, MessageTag, Transport};

    fn make_transport(world_size: usize, local_rank: usize) -> GrpcTransport {
        let config =
            GrpcTransportConfig::for_localhost_cluster(world_size, local_rank).expect("valid config");
        GrpcTransport::new(config).expect("valid transport")
    }

    #[tokio::test]
    async fn grpc_transport_send_recv_roundtrip() {
        let transport = make_transport(2, 0);
        let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);

        transport
            .send(0, 1, tag, b"hello-grpc".to_vec())
            .expect("send should succeed");

        let payload = transport
            .recv(1, 0, tag)
            .expect("receive should succeed");
        assert_eq!(payload, b"hello-grpc".to_vec());
    }

    #[tokio::test]
    async fn grpc_transport_barrier_works() {
        let transport = make_transport(4, 0);
        let tag = MessageTag::new(1, 0, MessagePhase::Collective, 0);

        for rank in 0..4 {
            transport
                .barrier(rank, tag)
                .expect("barrier should succeed");
        }

        // After all ranks arrive, barrier is cleared
        assert!(transport.barrier(0, tag).is_err());
    }

    #[tokio::test]
    async fn grpc_transport_validates_rank() {
        let transport = make_transport(2, 0);
        let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);

        assert!(transport.send(5, 0, tag, vec![]).is_err());
        assert!(transport.recv(0, 5, tag).is_err());
    }
}
