//! Comprehensive sum-check protocol implementation for Spartan zkSNARK.
//!
//! Provides various distinct sum-check protocols for different constraint types:
//!
//! 1. **OuterSumCheck**: Proves `f(x) = A(x)·B(x) - C(x) = 0` over Boolean hypercube
//! 2. **InnerSumCheck**: Proves `f(x) = ⟨A(x), B(x)⟩` for inner product constraints
//! 3. **CubicSumCheck**: Proves `f(x) = A(x)·B(x)·C(x)` for triple product constraints
//! 4. **BatchedCubicSumCheck**: Batched version of cubic sum-check for multiple claims
//! 5. **SparkSumCheck**: Proves `f(x) = A(x)·B(x)·C(x)` for sparse MLE evaluation
//! 6. **PCSSumCheck**: Proves polynomial evaluation claims for PCS integration
//!
//! All implementations share common optimizations:
//! - First round uses base field (Fp), later rounds use extension field (Fp4)
//! - Gruen's optimization: evaluates at points 0, 1, 2 for degree-2 polynomials
//! - Sparse representation reduces complexity from O(2^m) to O(nnz)
//!
//! ## Protocol Overview
//!
//! Each sum-check protocol follows the same high-level structure:
//! 1. **Proving Phase**: Computes polynomial evaluations and generates round proofs
//! 2. **Verification Phase**: Verifies round proofs using Fiat-Shamir challenges
//! 3. **Final Check**: Validates the final polynomial evaluation
//!
//! ## Mathematical Formulation
//!
//! Given polynomial g(x₁, ..., xₙ), prove:
//! ```
//! ∑_{w∈{0,1}ⁿ} g(w) = claimed_sum
//! ```
//!
//! Where g takes different forms depending on the specific protocol variant.

pub mod cubic;
pub mod inner;
pub mod outer;
pub mod pcs;
pub mod spark;

// Re-export all sum-check proof types
pub use cubic::{BatchedCubicSumCheckProof, CubicSumCheckProof};
pub use inner::InnerSumCheckProof;
pub use outer::OuterSumCheckProof;
pub use pcs::PCSSumCheckProof;
pub use spark::SparkSumCheckProof;

#[cfg(test)]
mod tests;
