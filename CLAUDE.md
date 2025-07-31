# Claude Context: Deep-FRI Project

## Project Overview
Deep-FRI is a Rust cryptographic library implementing zero-knowledge proof primitives. It focuses on multilinear extensions (MLEs), polynomial commitments, and Merkle tree constructions for efficient zero-knowledge proof systems.

## Architecture & Structure

### Module Organization
```
src/
├── lib.rs                 # Root module (pub mod prover; pub mod utils;)
├── prover.rs             # WHIR commitment scheme
└── utils/                # Core cryptographic primitives
    ├── mod.rs           # Type aliases: Fp=BabyBear, Fp4=BinomialExtensionField<BabyBear,4>
    ├── challenger.rs    # Fiat-Shamir challenger with BLAKE3
    ├── eq.rs           # Equality polynomial computations
    ├── merkle_tree.rs  # BLAKE3-based Merkle trees
    └── polynomial.rs   # Multilinear extension evaluations
```

### Key Types & Aliases
- `Fp` = `BabyBear` (base field)
- `Fp4` = `BinomialExtensionField<BabyBear, 4>` (4-degree extension)

## Core Components

### 1. WHIR Commitment (`prover.rs`)
- **Struct**: `WHIRCommitment { rate: usize, merkle_tree: MerkleTree, code: Vec<BabyBear> }`
- **Key Method**: `commit(&mut self, domain_evals: &[BabyBear], root_table: Option<&[Vec<BabyBear>]>)`
- Uses FFT operations: `BabyBear::forward_fft()` and `BabyBear::roots_of_unity_table()`

### 2. Equality Polynomials (`utils/eq.rs`)
- **Struct**: `EqEvals<'a> { point: &'a [Fp4], coeffs: Vec<Fp4>, n_vars: usize }`
- **Key Method**: `gen_from_point(point: &'a [Fp4])` - generates coefficients via tensor product expansion
- **Algorithm**: Iterative tensor product construction for equality check polynomials over hypercubes
- **Test Coverage**: Single variable and empty point cases

### 3. Multilinear Extensions (`utils/polynomial.rs`)
- **Struct**: `MLE { coeffs: Vec<BabyBear> }`
- **Constraint**: Coefficient length must be power of two
- **Key Method**: `evaluate(&self, point: &[Fp4]) -> Fp4` - uses EqEvals for efficient evaluation
- **Integration**: Relies on equality polynomials for evaluation

### 4. Merkle Trees (`utils/merkle_tree.rs`)
- **Struct**: `MerkleTree { root: [u8; 32], nodes: Vec<[u8; 32]>, depth: u32 }`
- **Hash Function**: BLAKE3 exclusively
- **Constraint**: Leaf count must be power of two
- **Methods**: `new()`, `get_path()`, `verify_path()`
- **Storage**: Flat vector with leaves first, then layers up to root

### 5. Challenger (`utils/challenger.rs`)
- **Struct**: `Challenger { state: Hasher, round: usize }`
- **Purpose**: Fiat-Shamir transformations for interactive proof protocols
- **Hash Function**: BLAKE3 Hasher
- **Methods**: `observe_field_elem()`, `observe_field_elems()`, `get_challenge()`

## Dependencies & Build
- **p3-baby-bear**: BabyBear finite field implementation
- **p3-field**: Generic field arithmetic traits
- **blake3**: Fast cryptographic hash function (used throughout)
- **anyhow**: Error handling
- **serde**: Serialization framework

**Build Commands**:
- `cargo build --release`
- `cargo test`

## Security & Design Principles
- Power-of-two requirements for cryptographic security
- BLAKE3 hash function used consistently
- Constant-time field operations where supported
- Extension field arithmetic over BabyBear base field
- Memory-efficient polynomial evaluation algorithms

## Current State
- Modified file: `src/utils/eq.rs` (git status shows modifications)
- Recent commits focus on polynomial evaluation and equality polynomial generation
- Comprehensive test coverage for core components
- Well-documented API with usage examples in README

## Development Notes
- All cryptographic operations use BLAKE3
- Power-of-two constraints are enforced throughout
- Extension field support is central to the design
- Test suite includes correctness verification for all major components
- Code follows Rust 2024 edition conventions

## Claude Memories

### Project Insights
- Take a look at the Plonky3 documentation