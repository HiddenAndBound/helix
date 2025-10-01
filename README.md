# Helix

A work-in-progress Rust implementation of the Spartan zkSNARK protocol for proving satisfiability of Rank-1 Constraint Systems (R1CS). Built on multilinear extensions (MLEs), sum-check protocols, and efficient sparse polynomial operations.

## Overview

Helix implements the Spartan zkSNARK protocol for proving satisfiability of Rank-1 Constraint Systems (R1CS). The library provides:

- **R1CS Constraint Systems**: Complete implementation with sparse matrix representation
- **Sum-Check Protocols**: Outer and inner sum-check with Fiat-Shamir transformation
- **Sparse Multilinear Extensions**: Memory-efficient O(nnz) polynomial operations
- **Spartan Proof Structure**: Two-phase proving for R1CS satisfaction (with placeholder commitments)
- **Multilinear Extensions (MLEs)**: Polynomial representations over extension fields
- **Equality Polynomials**: Tensor product expansions for hypercube evaluations
- **Supporting Utilities**: Challenger, Merkle trees, and field arithmetic

## Current Features

- **Foundation**: Built on the BabyBear finite field (p3-baby-bear) with Fp4 extensions
- **R1CS Systems**: Complete constraint system with sparse matrix operations
- **Sum-Check Protocols**: Outer and inner protocols for constraint satisfaction proving
- **Sparse Polynomials**: Memory-efficient O(nnz) operations instead of O(n²) dense representation
- **Cryptographic Utilities**: BLAKE3-based Merkle trees and Fiat-Shamir challenger
- **Comprehensive Testing**: Unit tests with mathematical correctness verification

## Planned Features

- **Polynomial Commitment Schemes**: Real cryptographic commitments (currently using placeholder)
- **Zero-Knowledge**: Privacy-preserving proofs with witness hiding
- **Performance Optimizations**: Parallelization and hardware acceleration
- **Complete Verification**: Standalone verifier with public input handling

## Dependencies

- `p3-baby-bear`: BabyBear finite field implementation  
- `p3-field`: Generic field arithmetic traits and utilities
- `p3-monty-31`: Montgomery arithmetic optimizations
- `blake3`: Fast cryptographic hash function
- `serde` & `serde_json`: Serialization framework
- `anyhow`: Error handling utilities

### Development Dependencies
- `proptest`: Property-based testing framework

## Core Components

### R1CS Constraint Systems (`src/spartan/r1cs.rs`)

Create and verify Rank-1 Constraint Systems with sparse matrix representation:

```rust
use helix::spartan::{R1CS, R1CSInstance, Witness};

// Create a test R1CS instance: x * y = z
let (r1cs, witness) = R1CS::simple_test_instance()?;
let instance = R1CSInstance::new(r1cs, witness)?;

// Verify constraints are satisfied
assert!(instance.verify()?);
```

### Poseidon2 Witness Generation (`src/spartan/r1cs/poseidon2.rs`)

Produce R1CS-compatible witnesses for the 16-lane Poseidon2 permutation. The module offers two
paths depending on whether you start from absorbed rate/capacity seeds or from fully materialised
public states:

```rust
use helix::spartan::{
    Poseidon2ColumnSeed, R1CSInstance, build_default_poseidon2_witness_matrix_from_states,
    build_poseidon2_instance, build_poseidon2_witness_matrix,
};
use p3_baby_bear::BabyBear;

// 1) Seed-based builder (rate + optional capacity lanes)
let poseidon = p3_baby_bear::default_babybear_poseidon2_16();
let seeds = vec![
    Poseidon2ColumnSeed {
        rate: vec![BabyBear::ONE, BabyBear::TWO],
        capacity: None,
    },
    Poseidon2ColumnSeed {
        rate: vec![BabyBear::from_int(5)],
        capacity: Some([BabyBear::ZERO; 14]),
    },
];
let seed_matrix = build_poseidon2_witness_matrix(&seeds, &poseidon)?;

// 2) Multi-state builder (each state is the 16-lane public input for one permutation column)
let mut s0 = [BabyBear::ZERO; 16];
s0[0] = BabyBear::ONE;
let mut s1 = [BabyBear::ZERO; 16];
s1[0] = BabyBear::from_int(5);
let states = vec![s0, s1];
let matrix = build_default_poseidon2_witness_matrix_from_states(&states)?;

assert_eq!(matrix.num_columns, 2);
assert_eq!(matrix.num_public_inputs, 16);
assert_eq!(matrix.assignments.len(), matrix.column_len * matrix.num_columns);

// Turn a specific column back into an R1CS instance for proving/verifying
let column0 = matrix.column_witness(0).expect("column 0 should exist");
let mut capacity0 = [BabyBear::ZERO; 14];
capacity0.copy_from_slice(&states[0][2..]);
let template = build_poseidon2_instance(&states[0][..2], Some(&capacity0), &poseidon)?;
let instance = R1CSInstance::new(template.r1cs.clone(), column0)?;
assert!(instance.verify()?);
```

Key facts about the multi-state API:

- Each column shares the exact same layout (public inputs at the first 16 lanes, identical wiring).
- Columns are padded to a power-of-two `column_len` and stored in column-major order inside
  `assignments`.
- `final_states[i]` and `digests[i]` cache the post-permutation result and rate digest for each
  column so that callers can reconstitute a full `Poseidon2Witness` when needed.
- The builders enforce consistent layout/length invariants and return
  `SparseError::ValidationError` if they are violated (including empty input sets).

### Spartan Proof Generation (`src/spartan/prover.rs`)

Generate proofs for R1CS satisfaction using sum-check protocols:

```rust
use helix::spartan::SpartanProof;
use helix::challenger::Challenger;

let instance = R1CSInstance::simple_test()?;
let mut challenger = Challenger::new();

// Generate Spartan proof (uses dummy commitments currently)
let proof = SpartanProof::prove(instance, &mut challenger);

// Verify the proof structure
let mut verifier = Challenger::new(); 
proof.verify(&mut verifier);
```

### Sum-Check Protocols (`src/spartan/sumcheck.rs`)

Interactive proof protocols for polynomial constraints:

```rust
use helix::spartan::sumcheck::{OuterSumCheckProof, InnerSumCheckProof};

// Outer sum-check proves R1CS constraint satisfaction
// Inner sum-check verifies evaluation claims
// Both use Fiat-Shamir for non-interactivity
```

### Sparse Multilinear Extensions (`src/spartan/sparse.rs`)

Memory-efficient sparse polynomial operations:

```rust
use helix::spartan::sparse::SparseMLE;

let sparse_mle = SparseMLE::new(sparse_coefficients)?;

// Bind first half of variables for sum-check
let bound_mle = sparse_mle.bind_first_half_variables(&eq_evals)?;
```

### Multilinear Extensions (`src/utils/polynomial.rs`)

Dense multilinear polynomials with evaluation and folding:

```rust
use helix::utils::polynomial::MLE;

let coeffs = vec![BabyBear::new(1), BabyBear::new(2), BabyBear::new(3), BabyBear::new(4)];
let mle = MLE::new(coeffs);
let point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
let result = mle.evaluate(&point);
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
- R1CS constraint satisfaction and witness validation
- Sum-check protocol correctness (outer and inner phases)
- Sparse polynomial operations and matrix bindings
- Multilinear extension evaluation and folding algorithms
- Equality polynomial tensor product expansions  
- Merkle tree construction and path verification
- Field arithmetic operations

## Implementation Status

This implementation provides a solid foundation for the Spartan zkSNARK protocol with several key components complete:

### ✅ **Fully Implemented**
- **R1CS Constraint Systems**: Complete with sparse matrix representation and witness handling
- **Sum-Check Protocols**: Both outer (R1CS satisfaction) and inner (evaluation claims) protocols
- **Sparse Multilinear Extensions**: Memory-efficient O(nnz) polynomial operations  
- **Supporting Utilities**: MLE, equality polynomials, Merkle trees, Fiat-Shamir challenger
- **Mathematical Foundations**: BabyBear field arithmetic with Fp4 extension field support

### ⚠️ **Partially Implemented**  
- **Spartan Prover**: Core proof structure exists but uses placeholder polynomial commitments
- **Verification**: Basic verification logic present but lacks complete public input handling
- **Polynomial Commitments**: Interface defined but only dummy implementation provided

### ❌ **Not Yet Implemented**
- **Real Polynomial Commitment Scheme**: Currently uses placeholder that accepts all proofs
- **Zero-Knowledge Features**: No privacy guarantees, witness information is leaked
- **Spark Protocol**: Referenced but not implemented (would handle commitment openings)
- **Complete Verifier**: Standalone verifier module with full public input support
- **Performance Optimizations**: No parallelization or hardware-specific optimizations

### Security Notice

⚠️ **This implementation is for educational and research purposes only.** The dummy polynomial commitment scheme provides no cryptographic security guarantees. Do not use in production without implementing a real commitment scheme.

## Architecture

The library is organized into two main modules:

### `src/spartan/` - Spartan zkSNARK Protocol
- `r1cs.rs`: R1CS constraint systems and witness handling
- `prover.rs`: Spartan proof generation and verification logic  
- `sumcheck.rs`: Sum-check protocols (outer, inner, and planned spark)
- `sparse.rs`: Sparse multilinear extension operations
- `commitment.rs`: Polynomial commitment trait (dummy implementation currently)
- `univariate.rs`: Univariate polynomial utilities for sum-check rounds
- `error.rs`: Spartan-specific error types

### `src/utils/` - Supporting Cryptographic Primitives
- `polynomial.rs`: Dense multilinear extension (MLE) implementations
- `eq.rs`: Equality polynomial computations with tensor products
- `merkle_tree.rs`: BLAKE3-based Merkle tree for commitments
- `challenger.rs`: Fiat-Shamir challenge generation

## Field Configuration

The library uses type aliases for consistent field usage:

- `Fp`: BabyBear base field
- `Fp4`: 4-degree extension field over BabyBear

## Security

- All cryptographic operations use the BLAKE3 hash function
- Merkle trees require power-of-two leaf counts for security  
- Field operations are constant-time where supported by underlying implementations
- ⚠️ **Current limitation**: Dummy polynomial commitments provide no security

## Future Vision

This implementation serves as the foundation for a more comprehensive zero-knowledge proving system. The planned roadmap includes:

### **WHIR Integration**
Integration with the WHIR polynomial commitment scheme for ultra-fast verification (290-610 microseconds). WHIR's Reed-Solomon proximity testing approach would replace the current dummy commitments with cryptographically secure polynomial commitments.

### **Twist & Shout Memory Checking**  
Implementation of revolutionary memory checking protocols using one-hot addressing:
- **Twist**: Read-write memory consistency with timestamps
- **Shout**: Batch read-only lookups for instruction fetches
- Both protocols integrate naturally with sum-check, avoiding expensive permutation arguments

### **Complete System**
The ultimate goal is a fully integrated system combining:
- Spartan's transparent zkSNARK protocol (no trusted setup)
- WHIR's blazing-fast polynomial commitment verification
- Twist & Shout's efficient memory checking
- Zero-knowledge features for privacy

This combination would create a high-performance SNARK suitable for production use while maintaining the simplicity and auditability that makes these protocols attractive.

For detailed technical specifications of the complete vision, see the [comprehensive implementation guide](context/total_plan/spartan-whir-twist-shout-guide.md).

## License

This project uses standard Rust edition 2024 conventions and dependencies with permissive licenses.
