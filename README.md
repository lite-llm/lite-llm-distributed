# lite-llm-distributed

Distributed execution crate for Lite LLM (`SPEC-011` to `SPEC-020`).

## Scope
Implements deterministic distributed primitives:

- DP/TP/PP/EP topology mapping and ownership rules
- routing consensus checksums and agreement
- deterministic collective operations and all-to-all ordering
- tagged transport abstraction and monotonic message ordering
- failure classification and deterministic recovery actions

## Modules
- `src/parallelism.rs`: topology shapes and rank/coordinate conversion
- `src/consensus.rs`: route selection consensus and checksum validation
- `src/collectives.rs`: deterministic all-reduce/all-to-all
- `src/transport.rs`: transport interface and in-memory tagged backend
- `src/fault_tolerance.rs`: failure domains, policies, and coordinator
- `src/error.rs`: distributed error model

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- System docs: `../lite-llm-docs/README.md`
- Crate map: `../lite-llm-docs/reference/crate-catalog.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
