# Helix

Helix is a proving pipeline for R1CS instances built around multilinear extensions (MLEs) and a BaseFold/FRI-like folding layer backed by Merkle commitments.

`examples/helix.rs` is an end-to-end driver that exercises the core protocol implementation in `src/protocols/batch_sumcheck.rs`.

## Getting started

This crate uses the pinned nightly toolchain in `rust-toolchain.toml`.

```bash
cargo build --release
cargo test
cargo run --release --example helix
```

For profiling-oriented runs:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example helix
```

## Example flow (`examples/helix.rs`)

At a high level, the example:

1. Builds a Poseidon2 R1CS instance (`build_default_poseidon2_instance`).
2. Generates a batch witness matrix for many Poseidon2 executions (`build_poseidon2_witness_matrix_from_states`).
3. Treats the (transposed) witness matrix as an MLE `z(x)` (`MLE::new(...)`).
4. Commits to `z` via `BatchSumCheckProof::commit_skip(...)`.
5. Produces a proof with `BatchSumCheckProof::prove(...)`, deriving challenges via `Challenger` (Fiat–Shamir).
6. Verifies via `BatchSumCheckProof::verify(...)` and checks the transcript-derived challenges match.

The minimal API surface you’ll see in the example is:

```rust
use helix::{
    BaseFoldConfig, BatchSumCheckProof, Challenger, Fp, MLE,
    build_default_poseidon2_instance, build_poseidon2_witness_matrix_from_states,
};
```

## Code reading guide

The repo is a single crate; the main modules are:

- `examples/helix.rs`: end-to-end driver
- `src/protocols/batch_sumcheck.rs`: core protocol implementation (commit/prove/verify)
- `src/r1cs/`: sparse R1CS representation and Poseidon2 constraints (`src/r1cs/poseidon2.rs`)
- `src/poly/`: MLE utilities, equality polynomial helpers, and sparse MLE operations
- `src/transcript/`: Fiat–Shamir transcript (`Challenger`)
- `src/pcs/`: BaseFold-style encoding + Merkle commitments (placeholder PCS)
- `src/merkle/`: Merkle tree implementation (BLAKE3-backed)
- `src/field/`: BabyBear base field `Fp` and extension field `Fp4`

## Invariants and layout expectations

- `BatchSumCheckProof::commit_skip` requires the committed MLE length to be a power of two.
- Witnesses are expected to be column-aligned with the constraint matrices; the Poseidon2 helpers produce the expected layout for the example.
- Folding / DFT helpers assume power-of-two domains; prefer power-of-two R1CS dimensions when changing or adding circuits.

## Configuration

`BaseFoldConfig` controls the commitment/folding knobs used by the protocol:

- `queries`: number of Merkle openings used for consistency checks
- `rate`: Reed–Solomon expansion factor (must be a power of two)
- `round_skip`: skip early commitment rounds (performance knob)
- `early_stopping_threshold`: stop folding early and send a larger tail (performance knob)
- `enable_parallel`: enable/disable parallel folding paths where supported

## Development

```bash
cargo clippy --all-targets --all-features
cargo test
cargo build --release
```

Benchmarks live under `benches/` (see `Cargo.toml` for names).

## License

MIT (see `LICENSE`).

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`cargo test`)
5. Run benchmarks if performance-critical (`cargo bench`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow Rust 2024 edition conventions
- Add tests for new functionality
- Update documentation and examples
- Ensure `cargo clippy` passes
- Format code with `cargo fmt`

## Acknowledgments

Built with the [Plonky3](https://github.com/Plonky3/Plonky3) field arithmetic library and inspired by the Spartan and BaseFold papers.

## Contact

[Your contact information or links to discussions/issues]

---

**Note**: This is research-grade software. Use in production at your own risk and conduct thorough security audits.
