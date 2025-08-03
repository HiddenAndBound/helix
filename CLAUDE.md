# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-FRI is a Rust implementation of cryptographic primitives for zero-knowledge proof systems. The codebase implements Spartan-WHIR-Twist&Shout protocols with a focus on multilinear extensions (MLEs), polynomial commitments, and Merkle tree constructions over the BabyBear finite field.

## Commands

### Building
```bash
cargo build --release
```

### Testing
```bash
cargo test
```

### Development Commands
```bash
# Check compilation without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

## Architecture

### Field Operations
- Uses BabyBear finite field (p = 2^31 - 2^27 + 1) as the base field
- Extension field arithmetic with 4-degree binomial extensions (Fp4)
- Type aliases: `Fp` (BabyBear), `Fp4` (4-degree extension)

### Core Module Structure
```
src/
├── lib.rs              # Main exports
├── spartan/            # Spartan zkSNARK implementation
│   ├── mod.rs
│   ├── prover.rs
│   └── sparse.rs
└── utils/              # Cryptographic primitives
    ├── challenger.rs   # Fiat-Shamir transformation
    ├── eq.rs          # Equality polynomials
    ├── merkle_tree.rs # Cryptographic commitment trees
    ├── mod.rs
    └── polynomial.rs  # Multilinear extensions (MLEs)
```

### Key Components

**Multilinear Extensions (MLEs)**: Core polynomial representation supporting efficient evaluation over extension fields. Located in `src/utils/polynomial.rs`.

**WHIR Commitments**: Polynomial commitment scheme with configurable rates, implemented in `src/spartan/prover.rs` via `WHIRCommitment`.

**Merkle Trees**: Use BLAKE3 hash function for secure data commitment with path verification in `src/utils/merkle_tree.rs`.

**Equality Polynomials**: Efficient computation using tensor product expansions in `src/utils/eq.rs`.

### Dependencies
- `p3-baby-bear`: BabyBear finite field implementation
- `p3-field`: Generic field arithmetic traits
- `blake3`: Cryptographic hash function
- `anyhow`: Error handling
- `serde`: Serialization framework

## Implementation Guidelines

### Working with Existing Code
The codebase has sparse matrix implementations and polynomial operations that should be preserved. The `sumcheck/` module mentioned in documentation has been removed, but existing utility functions should be maintained.

### Extension Areas
Based on the implementation plan, future development focuses on:
- Enhanced field operations with Montgomery form
- Complete WHIR polynomial commitment scheme
- Full Spartan zkSNARK integration
- Twist & Shout memory checking protocols

### Security Considerations
- All cryptographic operations use BLAKE3
- Merkle trees require power-of-two leaf counts
- Field operations are designed for constant-time execution where supported