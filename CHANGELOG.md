# Changelog

All notable changes to `lite-llm-distributed` are documented in this file.

## [0.1.0] - 2026-03-01
### Added
- DP/TP/PP/EP topology modeling and deterministic rank/coordinate conversion.
- Deterministic collective operations including all-to-all with fixed global order.
- Routing consensus checksum validation primitives.
- Tagged transport abstraction with monotonic tag-order enforcement.
- Failure domain modeling and deterministic recovery coordinator actions.
- Multi-rank determinism tests for topology, collectives, consensus, and recovery logic.
