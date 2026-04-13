//! gRPC client wrapper for communicating with remote `TransportService` servers.
//!
//! Provides a `GrpcTransportClient` with:
//! - Per-connection `TransportServiceClient` instances keyed by rank.
//! - Automatic retry with exponential backoff on transient failures.
//! - Configurable per-call timeout.
//! - Connection pooling via lazy channel creation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio_retry::strategy::{jitter, ExponentialBackoff};
use tokio_retry::Retry;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use crate::error::{DistributedError, DistributedResult};
use crate::transport::{MessagePhase, MessageTag};

// Re-export the tonic-generated client.
include!(concat!(env!("OUT_DIR"), "/transport.rs"));

/// Re-export from the service module for tag conversion.
use crate::grpc_service::{crate_tag_to_proto_tag, proto_tag_to_crate_tag};

/// Configuration for a gRPC transport client.
#[derive(Debug, Clone)]
pub struct GrpcClientConfig {
    /// Timeout for each RPC call (default: 30s).
    pub timeout: Duration,
    /// Maximum number of retries for transient failures (default: 3).
    pub max_retries: u32,
    /// Base backoff duration for retries (default: 100ms).
    pub base_backoff: Duration,
}

impl Default for GrpcClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            base_backoff: Duration::from_millis(100),
        }
    }
}

/// A client for a single peer's `TransportService`.
/// Manages a tonic channel and a lazily-created gRPC client.
#[derive(Clone)]
struct PeerClient {
    channel: Channel,
    timeout: Duration,
    max_retries: u32,
    base_backoff: Duration,
}

impl PeerClient {
    /// Connect to the peer at `uri` and create a new client.
    async fn connect(
        uri: &str,
        config: &GrpcClientConfig,
    ) -> DistributedResult<Self> {
        let endpoint: Endpoint = uri.parse().map_err(|e| {
            DistributedError::TransportError(format!("invalid peer URI: {e}"))
        })?;

        let channel = endpoint.connect_lazy().connect();
        let channel = Channel::balance_list(vec![channel].into_iter()).await;
        let channel = endpoint.connect().await.map_err(|e| {
            DistributedError::TransportError(format!("failed to connect to {uri}: {e}"))
        })?;

        Ok(Self {
            channel,
            timeout: config.timeout,
            max_retries: config.max_retries,
            base_backoff: config.base_backoff,
        })
    }

    /// Get a fresh `TransportServiceClient` bound to the channel.
    fn client(&self) -> transport_service_client::TransportServiceClient<Channel> {
        transport_service_client::TransportServiceClient::new(self.channel.clone())
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX)
    }

    /// Execute an RPC with retry logic.
    async fn with_retry<F, Fut, T>(&self, f: F) -> DistributedResult<T>
    where
        F: Fn(
            transport_service_client::TransportServiceClient<Channel>,
        ) -> Fut
            + Send
            + 'static,
        Fut: std::future::Future<Output = Result<T, Status>> + Send,
        T: Send + 'static,
    {
        let timeout = self.timeout;
        let max_retries = self.max_retries;
        let base = self.base_backoff;
        let channel = self.channel.clone();

        // Build the retry strategy.
        let strategy = ExponentialBackoff::from_millis(base.as_millis() as u64)
            .map(jitter)
            .take(max_retries as usize);

        Retry::spawn(strategy, move || {
            let channel = channel.clone();
            let f = &f;
            async move {
                let mut client = transport_service_client::TransportServiceClient::new(channel.clone())
                    .max_decoding_message_size(usize::MAX)
                    .max_encoding_message_size(usize::MAX);

                let result = tokio::time::timeout(timeout, f(client)).await
                    .map_err(|_| {
                        Status::deadline_exceeded("RPC call timed out")
                    })?;

                result.map_err(|status| {
                    // Only retry on transient errors.
                    if status.code() == tonic::Code::Unavailable
                        || status.code() == tonic::Code::DeadlineExceeded
                        || status.code() == tonic::Code::ResourceExhausted
                    {
                        status
                    } else {
                        // Non-retryable: wrap in a special status to stop retries.
                        Status::internal(format!("non-retryable error: {status}"))
                    }
                })
            }
        })
        .await
        .map_err(|e| DistributedError::TransportError(format!("RPC failed after retries: {e}")))
    }
}

/// A pool of gRPC clients, one per peer rank.
/// Handles connection pooling and retry logic.
#[derive(Clone)]
pub struct GrpcTransportClient {
    /// Peer clients keyed by rank.
    peers: Arc<Mutex<HashMap<usize, PeerClient>>>,
    /// Peer addresses keyed by rank (for lazy connection).
    peer_addrs: HashMap<usize, String>,
    /// Client configuration.
    config: GrpcClientConfig,
    /// World size.
    world_size: usize,
}

impl GrpcTransportClient {
    /// Create a new client pool.
    /// Connections are established lazily on first use.
    pub fn new(
        peer_addrs: HashMap<usize, String>,
        world_size: usize,
        config: GrpcClientConfig,
    ) -> DistributedResult<Self> {
        if peer_addrs.len() != world_size.saturating_sub(1) && world_size > 1 {
            // We expect addresses for all peers (world_size - 1 other nodes).
            // But allow if the caller provides partial addresses (some nodes may be local).
        }

        Ok(Self {
            peers: Arc::new(Mutex::new(HashMap::new())),
            peer_addrs,
            config,
            world_size,
        })
    }

    /// Get or create a `PeerClient` for the given rank.
    async fn get_peer(&self, rank: usize) -> DistributedResult<PeerClient> {
        {
            let peers = self.peers.lock().await;
            if let Some(client) = peers.get(&rank) {
                return Ok(client.clone());
            }
        }

        let uri = self
            .peer_addrs
            .get(&rank)
            .ok_or_else(|| {
                DistributedError::TransportError(format!("no address for peer rank {rank}"))
            })?
            .clone();

        let client = PeerClient::connect(&uri, &self.config).await?;

        {
            let mut peers = self.peers.lock().await;
            // Only insert if not already present (concurrent callers).
            peers.entry(rank).or_insert_with(|| client.clone());
        }

        Ok(client)
    }

    /// Send a payload to a remote node via gRPC.
    pub async fn send_message(
        &self,
        to_rank: usize,
        from_rank: usize,
        tag: MessageTag,
        payload: Vec<u8>,
    ) -> DistributedResult<()> {
        let peer = self.get_peer(to_rank).await?;
        let proto_tag = crate_tag_to_proto_tag(tag);

        let request = SendMessageRequest {
            from_rank: from_rank as u32,
            to_rank: to_rank as u32,
            tag: Some(proto_tag),
            payload,
        };

        let response = peer
            .with_retry(move |mut client| {
                let req = Request::new(request.clone());
                async move { client.send_message(req).await }
            })
            .await?
            .into_inner();

        if response.ok {
            Ok(())
        } else {
            Err(DistributedError::TransportError(format!(
                "remote send failed: {}",
                response.error
            )))
        }
    }

    /// Receive a payload from a remote node via gRPC.
    pub async fn recv_message(
        &self,
        from_rank: usize,
        to_rank: usize,
        tag: MessageTag,
    ) -> DistributedResult<Vec<u8>> {
        let peer = self.get_peer(from_rank).await?;
        let proto_tag = crate_tag_to_proto_tag(tag);

        let request = RecvMessageRequest {
            to_rank: to_rank as u32,
            from_rank: from_rank as u32,
            tag: Some(proto_tag),
            timeout_ms: self.config.timeout.as_millis() as u64,
        };

        let response = peer
            .with_retry(move |mut client| {
                let req = Request::new(request.clone());
                async move { client.recv_message(req).await }
            })
            .await?
            .into_inner();

        if response.ok {
            Ok(response.payload)
        } else {
            Err(DistributedError::MissingMessage { from_rank, to_rank })
        }
    }

    /// Execute a barrier with a remote node.
    pub async fn barrier(&self, to_rank: usize, rank: usize, tag: MessageTag) -> DistributedResult<()> {
        let peer = self.get_peer(to_rank).await?;
        let proto_tag = crate_tag_to_proto_tag(tag);

        let request = BarrierRequest {
            rank: rank as u32,
            tag: Some(proto_tag),
            world_size: self.world_size as u32,
        };

        let response = peer
            .with_retry(move |mut client| {
                let req = Request::new(request.clone());
                async move { client.barrier(req).await }
            })
            .await?
            .into_inner();

        if response.ok {
            Ok(())
        } else {
            Err(DistributedError::TransportError(format!(
                "remote barrier failed: {}",
                response.error
            )))
        }
    }

    /// Check the health of a remote node.
    pub async fn health_check(&self, rank: usize) -> DistributedResult<HealthCheckResponse> {
        let peer = self.get_peer(rank).await?;
        let request = HealthCheckRequest {};

        let response = peer
            .with_retry(move |mut client| {
                let req = Request::new(request.clone());
                async move { client.health_check(req).await }
            })
            .await?
            .into_inner();

        Ok(response)
    }

    /// Check health of all peer nodes.
    pub async fn health_check_all(&self) -> HashMap<usize, DistributedResult<HealthCheckResponse>> {
        let mut results = HashMap::new();
        for rank in self.peer_addrs.keys() {
            results.insert(*rank, self.health_check(*rank).await);
        }
        results
    }
}
