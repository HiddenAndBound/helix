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

**Multilinear Extensions (MLEs)**: Core polynomial representation supporting efficient evaluation over extension fields. Located in `src/utils/polynomial.rs`. Uses power-of-2 coefficient vectors and supports folding operations for sum-check protocols.

**Sparse Matrices (SparseMLE)**: Efficient sparse matrix representation for Spartan R1CS matrices. Located in `src/spartan/sparse.rs`. Key features:
- O(nnz) storage complexity vs O(n²) dense representation
- Power-of-2 dimension requirements for zkSNARK compatibility
- Direct MLE multiplication via `multiply_by_mle()` method
- Metadata preprocessing for sum-check protocols with timestamp tracking

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

## Testing Guidelines

### Isolated Module Testing
When compilation errors exist in other modules, use targeted testing approaches:
```bash
# Test specific module only
cargo test --lib spartan::sparse::tests

# Test with pattern matching
cargo test sparse --lib
```

### Module Isolation for Development
For parallel development, temporarily disable problematic modules in `mod.rs`:
```rust
mod error;
// mod prover;    // Temporarily disabled
mod sparse;
// mod sumcheck;  // Temporarily disabled
mod univariate;
```

### Field Element Construction
Use correct BabyBear constructors in tests:
- `BabyBear::new(value)` - Basic constructor
- `BabyBear::from_u32(value)` - From u32 (preferred in tests)
- `BabyBear::from_usize(value)` - From usize (for indices)
- **Avoid**: `BabyBear::from_canonical_u32()` (doesn't exist)

### Sparse Matrix Implementation Details

**Dimension Calculation**: Uses `(max_coordinate + 1).next_power_of_two()` to ensure power-of-2 sizing while containing all matrix indices.

**TimeStamps Arrays**: 
- `read_ts` and `final_ts` serve different purposes and have different sizes
- `read_ts[i]` = timestamp before i-th memory access (size = padded accesses)
- `final_ts[j]` = final write count for address j (size = address space)
- No requirement for equal array lengths

**Zero-Padding Strategy**: Dummy memory accesses to address 0 maintain power-of-2 memory access patterns without affecting computation results.