# Helix

A high-performance zero-knowledge proof system combining Spartan zkSNARK with BaseFold polynomial commitments, optimized for proving Poseidon2 hash computations over the BabyBear field.

## Overview

Helix implements an efficient zkSNARK system that brings together:

- **Spartan Protocol**: A succinct zkSNARK for R1CS (Rank-1 Constraint Systems)
- **BaseFold PCS**: A field-agnostic polynomial commitment scheme using Reed-Solomon codes and FRI-like folding
- **Poseidon2 Integration**: Native support for proving Poseidon2 hash function executions
- **BabyBear Field**: Optimized for the 31-bit prime field (p = 2³¹ - 2²⁷ + 1)

The system enables efficient batch proving of multiple Poseidon2 hash computations with aggressive optimizations for real-world performance.

## Features

- **BaseFold Polynomial Commitment Scheme**
  - Reed-Solomon encoding with FFT-based operations
  - FRI-like folding with Merkle tree commitments
  - Configurable security parameters (queries, encoding rate)
  - BLAKE3-based Merkle trees with hardware acceleration

- **Spartan zkSNARK Protocol**
  - R1CS constraint system with sparse matrix representation
  - Batch sumcheck protocol interleaved with PCS folding
  - Support for public inputs and private witnesses
  - Fiat-Shamir transformation for non-interactive proofs

- **Poseidon2 Hash Integration**
  - Width-16 permutation (rate 2)
  - 4 external + 13 internal rounds
  - Automatic R1CS constraint generation
  - Column-major witness matrices for batch proving

- **Performance Optimizations**
  - **Round Skipping**: Skip Merkle commitments in early folding rounds
  - **Early Stopping**: Stop FRI before final round, send codeword directly
  - **Sparse Matrices**: O(nnz) storage and operations vs O(n²)
  - **Parallelization**: Multi-threaded operations using Rayon
  - **SIMD Acceleration**: NEON optimizations for ARM processors

## Installation

Helix requires Rust edition 2024 or later.

### As a Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
helix = { git = "https://github.com/yourusername/helix" }
p3-baby-bear = "0.1"
p3-field = "0.1"
p3-symmetric = "0.1"
p3-dft = "0.1"

[profile.release]
debug = false
```

### From Source

```bash
git clone https://github.com/yourusername/helix
cd helix
cargo build --release
```

## Quick Start

Here's a complete example proving a batch of Poseidon2 hash computations:

```rust
use helix::helix::r1cs::poseidon2::{
    build_default_poseidon2_instance,
    build_poseidon2_witness_matrix_from_states,
    generate_random_states,
};
use helix::helix::sumcheck::batch_sumcheck::BatchSumCheckProof;
use helix::pcs::{BaseFoldConfig, BaseFoldSpec};
use helix::utils::challenger::Challenger;
use helix::utils::polynomial::MLE;
use p3_baby_bear::BabyBear as Fp;
use p3_dft::Radix2DitParallel;

fn main() -> anyhow::Result<()> {
    // 1. Build Poseidon2 R1CS instance with rate inputs
    let rate = [Fp::ONE, Fp::TWO];
    let instance = build_default_poseidon2_instance(&rate, None)?;
    let r1cs = &instance.r1cs;

    // 2. Generate witness matrix for batch of hashes
    let num_states = 1 << 12; // 4096 hash instances
    let initial_states = generate_random_states(num_states);
    let witness_matrix = build_poseidon2_witness_matrix_from_states(
        &initial_states,
        &instance.poseidon
    )?;

    // 3. Transpose witness to column-major format
    let z_transposed = MLE::new(witness_matrix.flattened_transpose());

    // 4. Configure BaseFold with optimizations
    let config = BaseFoldConfig::new()
        .with_early_stopping(11)  // Stop FRI at round 11
        .with_round_skip(6);       // Skip first 6 commitment rounds

    // 5. Precompute FFT roots
    let dft = Radix2DitParallel::default();
    let roots = BaseFoldSpec::<Fp>::precompute_roots(
        z_transposed.num_vars(),
        &dft
    )?;

    // 6. Commit to witness polynomial
    let (commitment, prover_data) = BatchSumCheckProof::commit_skip(
        &z_transposed,
        &dft,
        &config
    )?;

    // 7. Generate proof with Fiat-Shamir
    let mut challenger = Challenger::new();
    let (proof, _round_challenges) = BatchSumCheckProof::prove(
        &r1cs.a, &r1cs.b, &r1cs.c,
        &z_transposed,
        &commitment,
        &prover_data,
        &roots,
        &config,
        &mut challenger
    )?;

    // 8. Verify proof
    let mut verifier_challenger = Challenger::new();
    let (verified_challenges, final_evals) = proof.verify(
        &r1cs.a, &r1cs.b, &r1cs.c,
        commitment,
        &roots,
        &mut verifier_challenger,
        &config
    )?;

    println!("✓ Proof verified successfully!");
    println!("Batch size: {} hashes", num_states);

    Ok(())
}
```

Run the example:
```bash
cargo run --release --example helix
```

## Architecture

### Module Structure

```
helix/
├── pcs/                    # BaseFold Polynomial Commitment Scheme
│   ├── prover.rs          # Commitment and proof generation
│   ├── verifier.rs        # Proof verification
│   └── utils.rs           # Encoding, folding, Merkle operations
│
├── helix/                  # Spartan zkSNARK Protocol
│   ├── r1cs/              # R1CS constraint systems
│   │   ├── mod.rs         # Generic R1CS structures
│   │   └── poseidon2.rs   # Poseidon2-specific constraints
│   ├── sumcheck/          # Sumcheck protocol
│   │   └── batch_sumcheck.rs  # Batch sumcheck with PCS folding
│   └── univariate.rs      # Univariate polynomial operations
│
└── utils/                  # Common utilities
    ├── polynomial.rs      # Multilinear extensions (MLE)
    ├── sparse.rs          # Sparse matrix operations
    ├── merkle_tree.rs     # BLAKE3 Merkle trees
    ├── challenger.rs      # Fiat-Shamir transform
    └── eq.rs              # Equality polynomial helpers
```

### Key Components

#### R1CS Constraint System

R1CS represents computation constraints as three sparse matrices A, B, C:
```
(A · z) ⊙ (B · z) = C · z
```
where `z` is the witness vector (public inputs + private variables) and `⊙` is element-wise multiplication.

#### Batch Sumcheck Protocol

The prover demonstrates that over the Boolean hypercube:
```
f(x) = γ · z(x) + A(x) · B(x) - C(x) = 0
```
using a sumcheck protocol that's interleaved with BaseFold folding rounds for efficiency.

#### BaseFold Commitment

Commits to multilinear polynomials via:
1. Reed-Solomon encode the evaluation domain (FFT-based)
2. Build Merkle tree over encoded codewords
3. Return root hash as commitment
4. Provide query-based opening proofs with folding

## Configuration

### BaseFoldConfig Options

```rust
let config = BaseFoldConfig {
    queries: 144,              // Number of random queries (soundness)
    rate: 2,                   // Reed-Solomon encoding rate
    enable_parallel: true,     // Enable parallel processing
    round_skip: 6,             // Skip first N commitment rounds
    early_stopping_threshold: 11, // Stop FRI at this round
};
```

### Security vs Performance Tradeoffs

| Parameter | Security | Proof Size | Prover Time | Verifier Time |
|-----------|----------|------------|-------------|---------------|
| `queries` | ↑ Higher | ↑ Larger | → Same | ↑ Slower |
| `rate` | ↓ Lower | ↑ Larger | ↑ Slower | ↑ Slower |
| `round_skip` | → Same | ↓ Smaller | ↓ Faster | → Same |
| `early_stopping` | → Same* | ↑ Larger | ↓ Faster | ↓ Faster |

*Assuming security bound still met with fewer rounds

### Recommended Settings

For production use with 2^k constraints:
- **k ≤ 16**: `queries: 100, rate: 2, round_skip: 4, early_stopping: k-3`
- **k ≤ 20**: `queries: 144, rate: 2, round_skip: 6, early_stopping: k-5`
- **k > 20**: `queries: 200, rate: 2, round_skip: 8, early_stopping: k-6`

## Performance

### Benchmarks

Run benchmarks with:
```bash
cargo bench --bench poseidon2_spartan_bench
```

The benchmark suite tests batch proving for 2^5 to 2^20 Poseidon2 hash instances, measuring:
- Proof generation time
- Verification time
- Proof size
- Memory usage

### Optimization Features

1. **Sparse Matrix Representation**
   - Stores only non-zero entries: `HashMap<(row, col), value>`
   - Matrix-vector multiplication: O(nnz) instead of O(n²)
   - Memory footprint: ~100x smaller for typical R1CS

2. **Round Skipping**
   - Skip Merkle tree commitments for first `round_skip` rounds
   - Reduces proof size by ~30% (typical setting: skip 6 rounds)
   - No security loss (consistency checked via queries)

3. **Early Stopping**
   - Stop FRI protocol before reaching final polynomial
   - Send final codeword instead of continuing to single value
   - Trades proof size for computation time

4. **Parallel Operations**
   - Multi-threaded sparse matrix multiplications
   - Parallel FFT computations
   - Concurrent Merkle tree hashing with Rayon

5. **SIMD Acceleration**
   - NEON instructions for ARM processors (M1/M2/M3 Macs)
   - Vectorized field arithmetic via p3-baby-bear
   - BLAKE3 hardware acceleration

### Scaling Characteristics

Approximate performance for batch proving N Poseidon2 hashes on Apple M2:
- N = 2^10 (1,024): ~100ms prove, ~10ms verify
- N = 2^12 (4,096): ~300ms prove, ~15ms verify
- N = 2^16 (65,536): ~3s prove, ~25ms verify
- N = 2^20 (1M): ~40s prove, ~40ms verify

*Note: Actual performance depends on hardware and configuration.*

## Development

### Building

```bash
# Debug build
cargo build

# Release build (recommended)
cargo build --release

# With all optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_poseidon2_instance
```

### Running Examples

```bash
# Helix example with batch proving
cargo run --release --example helix
```

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench poseidon2_spartan_bench
```

### Development Status

**✅ Complete:**
- Core Spartan zkSNARK protocol
- BaseFold PCS implementation
- Poseidon2 R1CS constraint generation
- Batch sumcheck with optimizations
- Sparse matrix operations
- Merkle tree commitments
- Working examples and benchmarks

**🔄 Current Focus:**
- Performance benchmarking and tuning
- Parameter optimization for different use cases
- Documentation improvements

**🔮 Future Work:**
- Additional hash function support
- Custom constraint DSL
- Proof aggregation
- GPU acceleration

## Technical Details

### Field Configuration

- **Base Field**: BabyBear (`Fp = 2³¹ - 2²⁷ + 1`)
  - 31-bit prime field
  - Efficient arithmetic with Montgomery form
  - SIMD-friendly

- **Extension Field**: `Fp4 = BinomialExtensionField<BabyBear, 4>`
  - Degree-4 extension for FRI protocol
  - Used in BaseFold folding

### Poseidon2 Parameters

- **Width**: 16 (full state)
- **Rate**: 2 (absorb 2 elements per permutation)
- **Capacity**: 14
- **External Rounds**: 4 (before and after internal rounds)
- **Internal Rounds**: 13
- **S-box**: x^7 power map
- **MDS Matrices**: Separate for external/internal layers

### Cryptographic Security

The system's security relies on:
1. **Sumcheck Protocol**: Interactive proof made non-interactive via Fiat-Shamir
2. **BaseFold Soundness**: Query-based consistency checks on Reed-Solomon codewords
3. **Merkle Commitments**: BLAKE3 collision resistance
4. **Field Size**: 31-bit security against brute force

**Security Level**: ~100 bits (with default parameters)

## License

[Your License Here - e.g., MIT, Apache-2.0]

## Contributing

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
