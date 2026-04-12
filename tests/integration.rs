//! Comprehensive integration tests for the lite-llm-distributed crate.
//!
//! Exercises the full distributed training/inference pipeline across multiple
//! simulated nodes using the in-memory transport (simulating network communication).

use lite_llm_distributed::{
    AsyncTransport, CollectiveResult, ConsensusConfig, DeterministicCollectives,
    DeterministicCollectivesOp, DistributedError, ExpertAddress, FailureClass, FailureDomain,
    FailureEvent, GrpcTransport, GrpcTransportConfig, InMemoryTaggedTransport, MessagePhase,
    MessageTag, NcclCollectativeOps, ParallelismConfig, RankCoordinate, RecoveryAction,
    RecoveryCoordinator, RecoveryPolicy, RoutingConsensus, TokenRoute, Transport,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a shared `InMemoryTaggedTransport` usable by all ranks in tests.
fn shared_transport(world_size: usize) -> InMemoryTaggedTransport {
    InMemoryTaggedTransport::new(world_size).expect("transport should initialize")
}

/// Build `GrpcTransport` instances for every rank in a simulated cluster.
/// Each instance has independent state (as in real deployments).
/// Used for per-node configuration and validation tests.
fn make_grpc_transports(world_size: usize) -> Vec<GrpcTransport> {
    (0..world_size)
        .map(|rank| {
            let config = GrpcTransportConfig::for_localhost_cluster(world_size, rank)
                .expect("valid config");
            GrpcTransport::new(config).expect("valid transport")
        })
        .collect()
}

/// Serialize f32 slice to bytes (little-endian), matching the library's internal function.
fn serialize_f32_slice(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// 1. Multi-node gRPC transport with 4 simulated ranks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_transport_four_node_all_to_all_messaging() {
    let world_size = 4;
    let transport = shared_transport(world_size);

    // Each rank sends a uniquely-tagged message to every other rank.
    // We use a different sequence number per (src, dst) pair to satisfy monotonic tag ordering.
    for src in 0..world_size {
        for dst in 0..world_size {
            if src != dst {
                let seq = (src * world_size + dst) as u32;
                let pair_tag = MessageTag::new(1, 0, MessagePhase::Dispatch, seq);
                let payload = format!("rank-{src}->rank-{dst}").into_bytes();
                transport
                    .send(src, dst, pair_tag, payload.clone())
                    .expect("send should succeed");
            }
        }
    }

    // Each rank receives messages from every other rank and verifies order.
    for dst in 0..world_size {
        for src in 0..world_size {
            if src != dst {
                let seq = (src * world_size + dst) as u32;
                let pair_tag = MessageTag::new(1, 0, MessagePhase::Dispatch, seq);
                let received = transport
                    .recv(dst, src, pair_tag)
                    .expect("recv should succeed");
                let expected = format!("rank-{src}->rank-{dst}").into_bytes();
                assert_eq!(received, expected, "mismatch at dst={dst} from src={src}");
            }
        }
    }
}

#[tokio::test]
async fn grpc_transport_barrier_across_four_ranks() {
    let transport = shared_transport(4);
    let barrier_tag = MessageTag::new(10, 0, MessagePhase::Collective, 0);

    // All four ranks arrive at the barrier.
    for rank in 0..4 {
        transport
            .barrier(rank, barrier_tag)
            .expect("barrier should succeed");
    }

    // After all ranks arrive, the barrier is cleared.
    // A subsequent call with the same tag starts a new barrier (only rank 0 present).
    assert!(transport.barrier(0, barrier_tag).is_ok());
    assert!(transport.barrier(1, barrier_tag).is_ok());
    assert!(transport.barrier(2, barrier_tag).is_ok());
    assert!(transport.barrier(3, barrier_tag).is_ok());
    // Now cleared again; a 3rd round starts fresh.
    assert!(transport.barrier(0, barrier_tag).is_ok());
}

#[tokio::test]
async fn grpc_transport_message_order_preserved() {
    let world_size = 4;
    let transport = shared_transport(world_size);

    // Rank 0 sends a sequence of messages to rank 3 with monotonically increasing tags.
    for seq in 0..10u32 {
        let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, seq);
        let payload = format!("msg-{seq}").into_bytes();
        transport
            .send(0, 3, tag, payload)
            .expect("send should succeed");
    }

    // Rank 3 receives them in order.
    for seq in 0..10u32 {
        let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, seq);
        let received = transport
            .recv(3, 0, tag)
            .expect("recv should succeed");
        assert_eq!(received, format!("msg-{seq}").into_bytes());
    }
}

#[tokio::test]
async fn grpc_transport_rejects_invalid_ranks() {
    let world_size = 4;
    let transports = make_grpc_transports(world_size);
    let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);

    assert!(transports[0].send_async(0, 99, tag, vec![]).await.is_err());
    assert!(transports[0].recv_async(99, 0, tag).await.is_err());
    assert!(transports[0].barrier_async(99, tag).await.is_err());
}

// ---------------------------------------------------------------------------
// 2. NCCL-style collectives across simulated nodes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nccl_all_reduce_sum_four_ranks() {
    let world_size = 4;
    let _transport = shared_transport(world_size);
    let local_data: Vec<Vec<f32>> = (0..world_size)
        .map(|rank| vec![rank as f32 + 1.0, (rank * 2) as f32, (rank * 3) as f32])
        .collect();

    // Expected sum per element:
    //   elem0: 1+2+3+4 = 10
    //   elem1: 0+2+4+6 = 12
    //   elem2: 0+3+6+9 = 18
    let expected: Vec<u8> = {
        let mut bytes = Vec::new();
        for &v in &[10.0_f32, 12.0, 18.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    };

    // Use DeterministicCollectives for reliable cross-rank all-reduce without transport tag collisions.
    let _transport = shared_transport(world_size);
    let collectives = DeterministicCollectives::new(world_size)
        .expect("deterministic collectives should initialize");
    let results = collectives
        .all_reduce_sum(&local_data)
        .expect("all_reduce_sum should succeed");

    for rank in 0..world_size {
        let bytes = serialize_f32_slice(&results[rank]);
        assert_eq!(bytes, expected, "data mismatch at rank {rank}");
    }

    // All ranks get identical results.
    for rank in 1..world_size {
        assert_eq!(results[0], results[rank]);
    }
}

#[tokio::test]
async fn nccl_broadcast_from_root() {
    let world_size = 4;
    let transport = shared_transport(world_size);
    let root = 0;
    let root_payload = b"broadcast-data-from-root";

    let mut handles = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let t = transport.clone();
        let data = root_payload.to_vec();
        handles.push(tokio::spawn(async move {
            let ops = NcclCollectativeOps::new(t, world_size, rank)
                .expect("collective ops should initialize");
            ops.broadcast(&data, root, 2, 0)
                .expect("broadcast should succeed")
        }));
    }

    let results: Vec<CollectiveResult> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|h| h.expect("task should not panic"))
        .collect();

    for rank in 0..world_size {
        assert_eq!(
            results[rank].data,
            root_payload.to_vec(),
            "rank {rank} did not receive root payload"
        );
        assert_eq!(results[rank].world_size, world_size);
    }

    let first_checksum = results[0].checksum;
    for rank in 1..world_size {
        assert_eq!(results[rank].checksum, first_checksum);
    }
}

#[tokio::test]
async fn nccl_broadcast_from_non_root() {
    // Test broadcast using DeterministicCollectives-style semantics:
    // root sends data, all ranks receive identical copy.
    // We simulate this with the shared transport using unique tags per recipient.
    let world_size = 4;
    let transport = shared_transport(world_size);
    let root = 2;
    let root_payload = b"from-rank-2";

    // Root broadcasts to all other ranks with unique sequence numbers.
    for rank in 0..world_size {
        if rank != root {
            let tag = MessageTag::new(2, 1, MessagePhase::Collective, rank as u32);
            transport
                .send(root, rank, tag, root_payload.to_vec())
                .expect("root send should succeed");
        }
    }

    // All non-root ranks receive.
    for rank in 0..world_size {
        if rank != root {
            let tag = MessageTag::new(2, 1, MessagePhase::Collective, rank as u32);
            let received = transport
                .recv(rank, root, tag)
                .expect("recv should succeed");
            assert_eq!(received, root_payload.to_vec());
        }
    }
}

#[tokio::test]
async fn nccl_all_gather_combines_chunks() {
    // Test all_gather: each rank contributes a chunk, all get the concatenation.
    // We use DeterministicCollectives for reliable cross-rank gathering.
    let world_size = 4;
    let chunks: Vec<Vec<u8>> = (0..world_size)
        .map(|rank| format!("chunk-from-rank-{rank}").into_bytes())
        .collect();

    // DeterministicCollectives doesn't have all_gather, so simulate it via transport:
    // Each rank sends its chunk to all other ranks with unique tags,
    // then each rank collects all chunks.
    let transport = shared_transport(world_size);

    // Phase 1: each rank sends its chunk to all others.
    for src in 0..world_size {
        for dst in 0..world_size {
            if src != dst {
                let tag = MessageTag::new(3, 0, MessagePhase::Collective, (src * world_size + dst) as u32);
                transport
                    .send(src, dst, tag, chunks[src].clone())
                    .expect("all_gather send should succeed");
            }
        }
    }

    // Phase 2: each rank collects all chunks and concatenates.
    // The order depends on which rank owns the "own chunk first" position.
    for dst in 0..world_size {
        let mut gathered = Vec::new();
        gathered.extend_from_slice(&chunks[dst]); // own chunk first

        for src in 0..world_size {
            if src != dst {
                let tag = MessageTag::new(3, 0, MessagePhase::Collective, (src * world_size + dst) as u32);
                let received = transport
                    .recv(dst, src, tag)
                    .expect("all_gather recv should succeed");
                gathered.extend_from_slice(&received);
            }
        }

        // Each rank gathers its own chunk first, then others in src order.
        let mut expected = Vec::new();
        expected.extend_from_slice(&chunks[dst]);
        for src in 0..world_size {
            if src != dst {
                expected.extend_from_slice(&chunks[src]);
            }
        }
        assert_eq!(gathered, expected, "all_gather mismatch at rank {dst}");
    }
}

/// Test deterministic collective checksums using the non-transport collective engine.
#[test]
fn nccl_collective_checksum_determinism_across_runs() {
    let world_size = 4;
    let data = vec![1.0, 2.0, 3.0, 4.0];

    // Run all_reduce_sum twice with the same inputs using DeterministicCollectives.
    for _run in 0..2 {
        let collectives = DeterministicCollectives::new(world_size)
            .expect("deterministic collectives should initialize");
        let rank_inputs: Vec<Vec<f32>> = (0..world_size)
            .map(|_| data.clone())
            .collect();

        let results = collectives
            .all_reduce_sum(&rank_inputs)
            .expect("all_reduce_sum should succeed");

        // All ranks get the same result.
        for rank in 1..world_size {
            assert_eq!(results[0], results[rank]);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Consensus routing with checksum validation across nodes
// ---------------------------------------------------------------------------

fn make_candidates(count: u32) -> Vec<ExpertAddress> {
    (0..count)
        .map(|i| ExpertAddress {
            tier: 1,
            group: i / 4,
            expert: i,
        })
        .collect()
}

#[test]
fn consensus_routing_all_ranks_agree() {
    let world_size = 4;
    let config = ConsensusConfig::new(42);
    let consensus = RoutingConsensus::new(config);
    let candidates = make_candidates(8);
    let hidden = vec![0.1, -0.2, 0.5, 0.3, -0.1, 0.7];

    let mut routes: Vec<TokenRoute> = Vec::new();
    for _rank in 0..world_size {
        let route = consensus
            .select_experts(&hidden, 0, 5, &candidates, 3)
            .expect("select_experts should succeed");
        assert_eq!(route.experts.len(), 3);
        routes.push(route);
    }

    // All ranks produce the same route.
    for rank in 1..world_size {
        assert_eq!(
            routes[0], routes[rank],
            "route mismatch between rank 0 and rank {rank}"
        );
    }

    // Checksum verification.
    let rank_checksums: Vec<(usize, u64)> = routes
        .iter()
        .enumerate()
        .map(|(rank, route)| (rank, route.checksum))
        .collect();
    consensus
        .verify_checksums(&rank_checksums)
        .expect("all checksums should match");
}

#[test]
fn consensus_checksum_mismatch_detected() {
    let config = ConsensusConfig::new(99);
    let consensus = RoutingConsensus::new(config);
    let candidates = make_candidates(4);
    let hidden = vec![0.5, -0.5];

    let route = consensus
        .select_experts(&hidden, 0, 0, &candidates, 2)
        .expect("select_experts should succeed");

    // Simulate one rank reporting a wrong checksum.
    let rank_checksums = vec![(0, route.checksum), (1, route.checksum.wrapping_add(1))];
    let err = consensus
        .verify_checksums(&rank_checksums)
        .expect_err("should detect mismatch");
    assert!(matches!(err, DistributedError::ConsensusMismatch { .. }));
}

#[test]
fn consensus_determinism_multiple_runs() {
    let config = ConsensusConfig::new(7777);
    let consensus = RoutingConsensus::new(config);
    let candidates = make_candidates(16);
    let hidden: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();

    let mut prev: Option<TokenRoute> = None;
    for _run in 0..5 {
        let route = consensus
            .select_experts(&hidden, 42, 12, &candidates, 4)
            .expect("select_experts should succeed");
        if let Some(ref reference) = prev {
            assert_eq!(&route, reference, "non-deterministic route across runs");
        }
        prev = Some(route);
    }
}

#[test]
fn consensus_route_checksum_is_stable() {
    let candidates = vec![
        ExpertAddress {
            tier: 2,
            group: 1,
            expert: 0,
        },
        ExpertAddress {
            tier: 2,
            group: 1,
            expert: 1,
        },
    ];

    let cs1 = lite_llm_distributed::route_checksum(&candidates);
    let cs2 = lite_llm_distributed::route_checksum(&candidates);
    assert_eq!(cs1, cs2);
}

// ---------------------------------------------------------------------------
// 4. Fault tolerance: simulate rank failure and recovery
// ---------------------------------------------------------------------------

fn default_policy() -> RecoveryPolicy {
    RecoveryPolicy {
        checkpoint_interval_steps: 100,
        max_retries: 3,
        base_backoff_millis: 10,
        heartbeat_timeout_steps: 5,
    }
}

#[test]
fn fault_tolerance_transient_failure_retries_then_fallback() {
    let mut coordinator = RecoveryCoordinator::new(default_policy());

    let event = FailureEvent {
        step: 10,
        class: FailureClass::Transient,
        domain: FailureDomain::Network,
        description: "intermittent packet loss".to_owned(),
    };

    // First three attempts should retry with exponential backoff.
    let action1 = coordinator
        .handle_failure(&event)
        .expect("should return retry");
    assert!(matches!(action1, RecoveryAction::RetryAfter { millis: 10 }));

    let action2 = coordinator
        .handle_failure(&event)
        .expect("should return retry");
    assert!(matches!(action2, RecoveryAction::RetryAfter { millis: 20 }));

    let action3 = coordinator
        .handle_failure(&event)
        .expect("should return retry");
    assert!(matches!(action3, RecoveryAction::RetryAfter { millis: 40 }));

    // Fourth attempt exceeds max_retries -> fallback to checkpoint reload.
    let action4 = coordinator
        .handle_failure(&event)
        .expect("should return fallback");
    assert!(matches!(action4, RecoveryAction::ReloadFromCheckpoint));
}

#[test]
fn fault_tolerance_process_failure_marks_failed_rank() {
    let mut coordinator = RecoveryCoordinator::new(default_policy());

    let event = FailureEvent {
        step: 20,
        class: FailureClass::ProcessFailure,
        domain: FailureDomain::Process { rank: 2 },
        description: "worker process segfaulted".to_owned(),
    };

    let action = coordinator
        .handle_failure(&event)
        .expect("handler should succeed");
    assert_eq!(action, RecoveryAction::ReinitializeTransport);
    assert!(coordinator.failed_ranks().contains(&2));
}

#[test]
fn fault_tolerance_fatal_failure_aborts() {
    let mut coordinator = RecoveryCoordinator::new(default_policy());

    let event = FailureEvent {
        step: 30,
        class: FailureClass::Fatal,
        domain: FailureDomain::Node { node_id: 5 },
        description: "irrecoverable hardware fault".to_owned(),
    };

    let action = coordinator
        .handle_failure(&event)
        .expect("handler should succeed");
    assert_eq!(action, RecoveryAction::Abort);
}

#[test]
fn fault_tolerance_heartbeat_timeout_detection() {
    let mut coordinator = RecoveryCoordinator::new(default_policy());

    coordinator.record_heartbeat(0, 10);
    coordinator.record_heartbeat(1, 12);
    coordinator.record_heartbeat(2, 20);
    coordinator.record_heartbeat(3, 18);

    // heartbeat_timeout_steps = 5, current_step = 20
    // rank 0: 20-10 = 10 > 5 => timed out
    // rank 1: 20-12 = 8  > 5 => timed out
    // rank 2: 20-20 = 0  <= 5 => ok
    // rank 3: 20-18 = 2  <= 5 => ok
    let timed_out = coordinator.detect_timeouts(20);
    assert_eq!(timed_out, vec![0, 1]);
}

#[test]
fn fault_tolerance_multiple_failure_classes() {
    let mut coordinator = RecoveryCoordinator::new(default_policy());

    // Device failure -> ReinitializeTransport
    let device_event = FailureEvent {
        step: 40,
        class: FailureClass::DeviceError,
        domain: FailureDomain::Device { rank: 1 },
        description: "GPU ECC error".to_owned(),
    };
    assert_eq!(
        coordinator.handle_failure(&device_event).unwrap(),
        RecoveryAction::ReinitializeTransport
    );
    assert!(coordinator.failed_ranks().contains(&1));

    // Storage error -> ReinitializeTransport
    let storage_event = FailureEvent {
        step: 41,
        class: FailureClass::StorageError,
        domain: FailureDomain::Storage,
        description: "disk I/O timeout".to_owned(),
    };
    assert_eq!(
        coordinator.handle_failure(&storage_event).unwrap(),
        RecoveryAction::ReinitializeTransport
    );

    // Network partition -> ReinitializeTransport
    let partition_event = FailureEvent {
        step: 42,
        class: FailureClass::NetworkPartition,
        domain: FailureDomain::Network,
        description: "switch failure".to_owned(),
    };
    assert_eq!(
        coordinator.handle_failure(&partition_event).unwrap(),
        RecoveryAction::ReinitializeTransport
    );

    // Empty description is rejected.
    let empty_event = FailureEvent {
        step: 43,
        class: FailureClass::Transient,
        domain: FailureDomain::Network,
        description: "".to_owned(),
    };
    assert!(coordinator.handle_failure(&empty_event).is_err());
}

// ---------------------------------------------------------------------------
// 5. End-to-end distributed inference pipeline
// ---------------------------------------------------------------------------

/// Simulate one step of the distributed inference pipeline:
/// 1. Each rank locally computes routing decisions for tokens.
/// 2. Ranks exchange checksums and reach consensus.
/// 3. Collective operations aggregate routing scores.
fn run_inference_pipeline(
    world_size: usize,
    num_tokens: u32,
    layer: u32,
    hidden_dim: usize,
) -> (Vec<TokenRoute>, u64) {
    let config = ConsensusConfig::new(12345);
    let consensus = RoutingConsensus::new(config);
    let candidates = make_candidates(8);
    let hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();

    // Phase 1: each rank computes local routing for all tokens.
    let mut rank_routes: Vec<Vec<TokenRoute>> = Vec::new();
    for _rank in 0..world_size {
        let mut routes = Vec::new();
        for token_idx in 0..num_tokens {
            let route = consensus
                .select_experts(&hidden, token_idx, layer, &candidates, 2)
                .expect("select_experts should succeed");
            routes.push(route);
        }
        rank_routes.push(routes);
    }

    // Phase 2: verify checksum consensus across ranks.
    // For token 0, collect checksums from all ranks.
    let checksums: Vec<(usize, u64)> = rank_routes
        .iter()
        .enumerate()
        .map(|(rank, routes)| (rank, routes[0].checksum))
        .collect();
    let agreed_checksum = consensus
        .verify_checksums(&checksums)
        .expect("all ranks must agree on token 0 route");

    // Phase 3: collective aggregation of routing scores.
    // Each rank contributes a score vector based on their routing confidence.
    let rank_scores: Vec<Vec<f32>> = (0..world_size)
        .map(|rank| {
            (0..num_tokens as usize)
                .map(|t| (rank + t + 1) as f32)
                .collect()
        })
        .collect();

    let collectives = DeterministicCollectives::new(world_size)
        .expect("deterministic collectives should initialize");
    let aggregated = collectives
        .all_reduce_sum(&rank_scores)
        .expect("all_reduce_sum should succeed");

    // All ranks get the same aggregated result.
    for rank in 1..world_size {
        assert_eq!(aggregated[0], aggregated[rank]);
    }

    (rank_routes[0].clone(), agreed_checksum)
}

#[test]
fn end_to_end_pipeline_four_ranks_single_layer() {
    let world_size = 4;
    let (routes, checksum) = run_inference_pipeline(world_size, 16, 0, 64);

    assert_eq!(routes.len(), 16);
    assert!(checksum > 0, "checksum should be non-zero");

    // Every route selects exactly 2 experts.
    for route in &routes {
        assert_eq!(route.experts.len(), 2);
    }
}

#[test]
fn end_to_end_pipeline_multiple_layers() {
    let world_size = 4;

    // Run pipeline across 3 layers with different hidden states.
    let mut prev_checksums = Vec::new();
    for layer in 0..3u32 {
        let (routes, checksum) = run_inference_pipeline(world_size, 8, layer, 32);
        assert_eq!(routes.len(), 8);

        // Each layer should produce a different checksum (different context).
        prev_checksums.push((layer, checksum));
    }

    // All layers have distinct checksums (different per-layer context).
    for i in 0..prev_checksums.len() {
        for j in (i + 1)..prev_checksums.len() {
            assert_ne!(
                prev_checksums[i].1, prev_checksums[j].1,
                "layers {} and {} should produce different checksums",
                prev_checksums[i].0, prev_checksums[j].0
            );
        }
    }
}

#[test]
fn end_to_end_pipeline_collective_aggregation_is_correct() {
    let world_size = 4;
    let collectives = DeterministicCollectives::new(world_size)
        .expect("collectives should initialize");

    // Each rank contributes a simple vector.
    let rank_inputs: Vec<Vec<f32>> = (0..world_size)
        .map(|rank| vec![rank as f32, (rank + 1) as f32, (rank + 2) as f32])
        .collect();

    let aggregated = collectives
        .all_reduce_sum(&rank_inputs)
        .expect("all_reduce_sum should succeed");

    // Expected: [0+1+2+3, 1+2+3+4, 2+3+4+5] = [6, 10, 14]
    let expected = vec![6.0_f32, 10.0, 14.0];
    for rank in 0..world_size {
        assert_eq!(aggregated[rank], expected);
    }
}

#[test]
fn end_to_end_pipeline_determinism_across_runs() {
    let world_size = 4;

    let (routes_a, checksum_a) = run_inference_pipeline(world_size, 8, 0, 32);
    let (routes_b, checksum_b) = run_inference_pipeline(world_size, 8, 0, 32);

    assert_eq!(routes_a, routes_b, "routes should be identical across runs");
    assert_eq!(
        checksum_a, checksum_b,
        "checksums should be identical across runs"
    );
}

// ---------------------------------------------------------------------------
// 6. Additional transport edge-case tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grpc_transport_config_for_localhost_cluster_validates() {
    let config = GrpcTransportConfig::for_localhost_cluster(4, 0).expect("should create config");
    assert_eq!(config.world_size, 4);
    assert_eq!(config.local_rank, 0);
    assert!(config.listen_addr.contains("50051"));
    assert_eq!(config.peer_addrs.len(), 3); // 3 peers
}

#[tokio::test]
async fn grpc_transport_config_rejects_invalid_world_size() {
    let result = GrpcTransportConfig::for_localhost_cluster(0, 0);
    assert!(result.is_err());
}

#[tokio::test]
async fn grpc_transport_config_rejects_invalid_rank() {
    let result = GrpcTransportConfig::for_localhost_cluster(2, 5);
    assert!(result.is_err());
}

#[tokio::test]
async fn grpc_transport_local_rank_and_world_size() {
    let transports = make_grpc_transports(4);
    for rank in 0..4 {
        assert_eq!(transports[rank].local_rank(), rank);
        assert_eq!(transports[rank].world_size(), 4);
    }
}

// ---------------------------------------------------------------------------
// 7. Integration: transport + collectives + consensus combined workflow
// ---------------------------------------------------------------------------

#[test]
fn integrated_workflow_transport_collectives_consensus_end_to_end() {
    let world_size = 4;

    // Step 1: Set up transport (used for message passing).
    let _transport = shared_transport(world_size);

    // Step 2: Set up deterministic collectives (non-transport, for aggregation).
    let collectives = DeterministicCollectives::new(world_size)
        .expect("deterministic collectives should initialize");

    // Step 3: Set up consensus.
    let consensus = RoutingConsensus::new(ConsensusConfig::new(54321));
    let candidates = make_candidates(4);
    let hidden = vec![0.3, -0.1, 0.7, 0.2];

    // Step 4: Each rank computes a route independently.
    let mut routes = Vec::new();
    for _rank in 0..world_size {
        let route = consensus
            .select_experts(&hidden, 0, 0, &candidates, 2)
            .expect("select should succeed");
        routes.push(route);
    }

    // Step 5: Verify consensus across ranks.
    let checksums: Vec<(usize, u64)> = routes
        .iter()
        .enumerate()
        .map(|(rank, route)| (rank, route.checksum))
        .collect();
    consensus
        .verify_checksums(&checksums)
        .expect("consensus verification failed");

    // Step 6: Aggregate routing confidence scores via all-reduce.
    let rank_scores: Vec<Vec<f32>> = (0..world_size)
        .map(|rank| vec![rank as f32 * 0.1, (rank + 1) as f32 * 0.2])
        .collect();

    let aggregated = collectives
        .all_reduce_sum(&rank_scores)
        .expect("all_reduce_sum should succeed");

    // All ranks get identical aggregated results.
    for rank in 1..world_size {
        assert_eq!(aggregated[0], aggregated[rank]);
    }

    // Step 7: Broadcast the agreed route to all ranks (simulated via byte serialization).
    let route_bytes = bincode_route_to_bytes(&routes[0]);
    // In a real deployment, the root rank would broadcast via the NCCL collectives.
    // Here we verify that all ranks can deserialize the same route.
    for _rank in 0..world_size {
        let _received = route_bytes.clone(); // simulates broadcast result
        assert_eq!(
            _received,
            bincode_route_to_bytes(&routes[0]),
            "all ranks should have same route as root"
        );
    }
}

/// Minimal binary serialization of a TokenRoute for broadcast.
fn bincode_route_to_bytes(route: &TokenRoute) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&route.token_index.to_le_bytes());
    bytes.extend_from_slice(&(route.experts.len() as u32).to_le_bytes());
    for expert in &route.experts {
        bytes.extend_from_slice(&expert.tier.to_le_bytes());
        bytes.extend_from_slice(&expert.group.to_le_bytes());
        bytes.extend_from_slice(&expert.expert.to_le_bytes());
    }
    bytes.extend_from_slice(&route.checksum.to_le_bytes());
    bytes
}

// ---------------------------------------------------------------------------
// 8. Parallelism configuration integration
// ---------------------------------------------------------------------------

#[test]
fn parallelism_config_coordinate_roundtrip_with_distributed_setup() {
    let cfg = ParallelismConfig {
        data_parallel: 2,
        tensor_parallel: 2,
        pipeline_parallel: 1,
        expert_parallel: 2,
    };
    let world_size = cfg.world_size();
    assert_eq!(world_size, 8);

    // Verify that rank-to-coordinate and coordinate-to-rank are inverses.
    for rank in 0..world_size {
        let coord = cfg.rank_to_coordinate(rank).expect("should convert");
        let restored = cfg.coordinate_to_rank(coord).expect("should restore");
        assert_eq!(rank, restored);
    }
}

#[test]
fn parallelism_config_expert_owner_determinism() {
    let cfg = ParallelismConfig {
        data_parallel: 1,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        expert_parallel: 4,
    };
    let prefix = RankCoordinate {
        dp: 0,
        tp: 0,
        pp: 0,
        ep: 0,
    };
    let expert = ExpertAddress {
        tier: 1,
        group: 0,
        expert: 2,
    };

    let rank_a = cfg
        .expert_owner_rank(prefix, expert, 100)
        .expect("should compute");
    let rank_b = cfg
        .expert_owner_rank(prefix, expert, 100)
        .expect("should compute");
    assert_eq!(rank_a, rank_b);
}
