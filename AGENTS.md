# Repository Guidelines

## Project Structure & Module Organization

Helix is a single nightly Rust crate (`Cargo.toml`, `rust-toolchain.toml`).

Core modules:

- `src/helix/`: prover/verifier wiring, R1CS (`src/helix/r1cs/`), and sum-check (`src/helix/sumcheck/`).
- `src/pcs/`: placeholder PCS + FFT helpers (not production-sound).
- `src/utils/`: BabyBear/Fp4 field types, Fiat–Shamir `Challenger`, Merkle trees, and polynomial utilities.

Keep witnesses column-aligned with constraint matrices, and prefer power-of-two R1CS dimensions where required by sum-check/FFT code.

## Build, Test, and Development Commands

- `cargo build --release`: optimized build; catches nightly-only breakages early.
- `cargo test`: runs unit + property tests (`proptest`) under `#[cfg(test)]`.
- `cargo clippy --all-targets --all-features`: lints all targets/features; fix warnings before review.
- `RUSTFLAGS="-C target-cpu=native" cargo run --release --example helix`: performance tracking.

## Coding Style & Naming Conventions

- Rust defaults: 4-space indentation, `snake_case` fns/vars, `CamelCase` types, `UPPER_SNAKE_CASE` consts.
- Prefer explicit module paths and keep public APIs near the top of each module; keep helpers `pub(crate)`/private.
- Add short, plain-English comments for non-obvious math/transcript steps.

## Testing Guidelines

- Put unit tests next to implementation in `mod tests` guarded by `#[cfg(test)]`; name tests `*_test`.
- Prefer `proptest` for property-heavy logic (e.g., polynomial identities, `Challenger` sequencing, size/edge cases).
- Re-run relevant benches when touching FFTs, sum-check, or field arithmetic to catch performance regressions.

## Commit & Pull Request Guidelines

- Commit messages in this repo are usually short, imperative, and lowercase (e.g., `refactor`, `verifier updated`); scoped Conventional Commits like `feat(spartan): ...` are also acceptable.
- PRs should describe the problem, the fix, and the impacted modules (e.g., `src/helix/sumcheck/`), and include `cargo test` output. Attach benchmark notes when changing FFT/sum-check/field arithmetic.

## Security & Configuration Tips

- The PCS layer in `src/pcs/` is a placeholder; do not claim cryptographic soundness without a real commitment scheme.
- Keep transcript behavior deterministic and use the pinned nightly toolchain from `rust-toolchain.toml`.
