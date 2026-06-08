# lethe

Workspace bootstrap for the Lethe program.

## Crates

- `lethe-core`: pure domain layer
- `lethe-substrates`: substrate adapters
- `lethe-model`: machine and experiment models
- `lethe-diagnostic`: diagnostic services
- `lethe-cli`: binary entrypoint

## Local preflight

```bash
cargo build --workspace
cargo test --workspace
cargo l
cargo audit
```

## Reproducibility check

The CI workflow runs a fixed-seed micro-experiment on x86_64 and arm64, uploads both
binary outputs, and fails if bytes differ.

Run it locally:

```bash
cargo run -p lethe-cli -- repro --seed 424242 --steps 1024 --output repro.bin
```

Repeat on another architecture and compare:

```bash
cmp -s repro-x86.bin repro-arm.bin
```
