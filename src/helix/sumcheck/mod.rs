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
//! ```ignore
//! ∑_{w∈{0,1}ⁿ} g(w) = claimed_sum
//! ```
//!
//! Where g takes different forms depending on the specific protocol variant.
pub mod batch_sumcheck;
use p3_field::{ Field, PrimeCharacteristicRing };

use crate::Fp;

fn eval_at_two<F: Field>(eval_0: F, eval_1: F) -> F {
    eval_1.double() - eval_0
}

#[inline]
fn eval_at_infinity<F: Field>(eval_0: F, eval_1: F) -> F {
    eval_1 - eval_0
}

pub fn transpose_column_major(matrix: &[Fp], rows: usize, cols: usize) -> Vec<Fp> {
    assert_eq!(rows * cols, matrix.len(), "matrix dimensions mismatch");

    let mut transposed = vec![Fp::ZERO; matrix.len()];

    for col in 0..cols {
        let start = col * rows;
        for row in 0..rows {
            transposed[row * cols + col] = matrix[start + row];
        }
    }

    transposed
}
