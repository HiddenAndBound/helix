# Helix

A Rust implementation of cryptographic primitives for zero-knowledge proof systems, focusing on multilinear extensions (MLEs), polynomial commitments, and Merkle tree constructions.

## Overview

Helix provides core cryptographic building blocks for constructing efficient zero-knowledge proof systems. The library implements:

- **Multilinear Extensions (MLEs)**: Polynomial representations supporting efficient evaluation over extension fields
- **WHIR Commitment Scheme**: A polynomial commitment protocol with configurable rates  
- **Merkle Trees**: Cryptographic hash trees with BLAKE3 for secure data commitment
- **Equality Polynomials**: Efficient computation of equality check polynomials over hypercubes
- **Challenger**: Fiat-Shamir transformation utilities for interactive proof protocols

## Features

- Built on the BabyBear finite field (p3-baby-bear)
- Extension field arithmetic with 4-degree binomial extensions
- BLAKE3 hash function integration for cryptographic security
- Comprehensive test coverage with correctness verification
- Memory-efficient polynomial evaluation algorithms

## Dependencies

- `p3-baby-bear`: BabyBear finite field implementation
- `p3-field`: Generic field arithmetic traits and utilities
- `blake3`: Fast cryptographic hash function
- `anyhow`: Error handling utilities
- `serde`: Serialization framework

## Core Components

### Multilinear Extensions (`src/utils/polynomial.rs`)

The `MLE` struct represents multilinear polynomials that can be efficiently evaluated at points in extension fields:

```rust
let coeffs = vec![BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4)];
let mle = MLE::new(coeffs);
let point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
let result = mle.evaluate(&point);
```

### WHIR Commitments (`src/prover.rs`)

The `WHIRCommitment` struct implements a polynomial commitment scheme:

```rust
let code = vec![BabyBear::new(1), BabyBear::new(2)];
let merkle_tree = MerkleTree::new(code.clone())?;
let mut commitment = WHIRCommitment::new(code, merkle_tree, 2);
commitment.commit(&domain_evals, None)?;
```

### Merkle Trees (`src/utils/merkle_tree.rs`)

Secure commitment to vectors of field elements with path verification:

```rust
let leaves = vec![BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4)];
let tree = MerkleTree::new(leaves)?;
let path = tree.get_path(0);
tree.verify_path(0, path)?;
```

### Equality Polynomials (`src/utils/eq.rs`)

Efficient computation of equality check polynomials using tensor product expansions:

```rust
let point = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
let eq_evals = EqEvals::gen_from_point(&point);
// eq_evals.coeffs contains the multilinear extension coefficients
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

The test suite includes comprehensive correctness verification for:
- Multilinear extension evaluation algorithms
- Equality polynomial tensor product expansions  
- Merkle tree construction and path verification
- Field arithmetic operations

## Architecture

The library is organized into two main modules:

- `prover`: High-level proof system components and commitment schemes
- `utils`: Low-level cryptographic primitives and field operations
  - `polynomial`: Multilinear extension implementations
  - `eq`: Equality polynomial computations
  - `merkle_tree`: Cryptographic commitment trees
  - `challenger`: Fiat-Shamir challenge generation

## Field Configuration

The library uses type aliases for consistent field usage:

- `Fp`: BabyBear base field
- `Fp4`: 4-degree extension field over BabyBear

## Security

- All cryptographic operations use the BLAKE3 hash function
- Merkle trees require power-of-two leaf counts for security
- Field operations are constant-time where supported by underlying implementations

## License

This project uses standard Rust edition 2024 conventions and dependencies with permissive licenses.