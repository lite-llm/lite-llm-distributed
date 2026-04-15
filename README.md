# lite-llm-distributed

Distributed execution crate for Lite LLM (`SPEC-011` to `SPEC-020`).

## Overview
Implements deterministic distributed primitives for multi-node LLM execution, including topology mapping, consensus, collectives, gRPC transport, and fault tolerance.

This crate provides the complete distributed execution stack: DP/TP/PP/EP topology shapes and rank/coordinate conversion, routing consensus checksums with deterministic agreement, NCCL-style collective operations (all-reduce, all-gather, broadcast, reduce), a tagged transport abstraction with gRPC backend using tokio + tonic, failure classification with deterministic recovery actions, and monotonic message ordering with tag violation detection.

## Features

### Feature Flag: `default` (empty)
No optional features enabled by default. The core transport abstraction is always available.

### Feature Flag: `grpc-transport` (optional)
Enables tonic/prost dependencies and the full async gRPC transport stack (`GrpcTransport`, `GrpcTransportClient`, `GrpcServiceImpl`). Required for real network communication between cluster nodes.

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1 | Async runtime for network I/O |
| async-trait | 0.1 | Async trait support |
| tonic | 0.12 | gRPC framework for node-to-node communication |
| prost | 0.13 | Protocol buffer serialization |
| bytes | 1 | Message buffer types |
| tower | 0.5 | Service tower for reconnection |
| tokio-retry | 0.3 | Exponential backoff retry logic |
| tracing | 0.1 | Logging and tracing |

## Key Modules
- `parallelism` — DP/TP/PP/EP topology shapes, rank/coordinate conversion
- `consensus` — routing consensus checksums and deterministic agreement
- `collectives` — deterministic all-reduce, all-gather, broadcast, reduce
- `collective_ops` — NCCL-style collective operations with checksum verification
- `transport` — tagged transport abstraction and in-memory backend
- `grpc_transport` — async gRPC transport with `AsyncTransport` trait, `GrpcTransport`, `GrpcTransportConfig`
- `grpc_service` — gRPC service implementation with `TransportState` for message queuing
- `grpc_client` — gRPC client pool with connection pooling and retry logic
- `fault_tolerance` — failure classification, recovery policies, coordinator
- `error` — distributed error model with topology and tag violation errors

## Public API
### Core Types
- `GrpcTransport` — gRPC-backed transport implementing `AsyncTransport` and sync `Transport`
- `GrpcTransportConfig` — endpoint configuration with listen address, peer map, world size, rank
- `AsyncTransport` — async interface for `send_async`, `recv_async`, `barrier_async`
- `TransportState` — shared message queue and barrier state for gRPC server
- `GrpcTransportClient` — client pool with lazy connections and retry logic
- `GrpcServiceImpl` — gRPC service handling SendMessage, RecvMessage, Barrier, HealthCheck RPCs
- `DeterministicCollectives` — ring all-reduce implementation
- `CollectiveOps` — NCCL-style all-reduce, broadcast, all-gather, reduce
- `RoutingConsensus` — checksum-based route selection consensus
- `RecoveryCoordinator` — failure classification and recovery action dispatcher
- `ParallelismConfig` — DP/TP/PP/EP configuration

### Core Functions
- `route_checksum()` — compute deterministic checksum for route validation

### Traits
- `AsyncTransport` — async transport interface for distributed message passing
- `Transport` — sync transport interface (backward compatible)
- `TransportBackend` — backend type enumeration (Quic, etc.)

## Quick Start
```rust
use lite_llm_distributed::{
    GrpcTransport, GrpcTransportConfig, AsyncTransport,
    MessageTag, MessagePhase, Transport,
};

// Create a gRPC transport for a 4-node localhost cluster
let config = GrpcTransportConfig::for_localhost_cluster(4, 0)
    .expect("valid config");
let transport = GrpcTransport::new(config).expect("valid transport");

// Send a message to another rank
let tag = MessageTag::new(1, 0, MessagePhase::Dispatch, 0);
transport.send(0, 1, tag.clone(), b"hello".to_vec())
    .expect("send should succeed");

// Receive the message
let payload = transport.recv(1, 0, tag)
    .expect("receive should succeed");
assert_eq!(payload, b"hello");
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate implements the distributed execution layer for the lite-llm platform. The gRPC transport (`grpc_transport` module) enables real network communication between cluster nodes, replacing the in-memory tagged transport used in single-process testing. Collective operations and consensus mechanisms ensure deterministic behavior across distributed ranks, while fault tolerance primitives map failure classes to recovery actions. The crate integrates with `lite-llm-storage` for distributed checkpointing and with `lite-llm` orchestrator for cluster bootstrap.

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
