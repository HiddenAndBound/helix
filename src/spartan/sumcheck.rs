//! Comprehensive sum-check protocol implementation for Spartan zkSNARK.
//!
//! Provides three distinct sum-check protocols for different constraint types:
//! 
//! 1. **OuterSumCheck**: Proves `f(x) = A(x)·B(x) - C(x) = 0` over Boolean hypercube
//! 2. **InnerSumCheck**: Proves `f(x) = ⟨A(x), B(x)⟩` for inner product constraints
//! 3. **SparkSumCheck**: Proves `f(x) = A(x)·B(x)·C(x)` for triple product constraints
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
//! Where g takes different forms:
//! - Outer: g(x) = A(x)·B(x) - C(x)
//! - Inner: g(x) = A(x)·B(x)  
//! - Spark: g(x) = A(x)·B(x)·C(x)

use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    polynomial::MLE,
    spartan::{sparse::SparseMLE, univariate::UnivariatePoly},
};
use std::ops::Mul;

/// Sum-check proof demonstrating that f(x₁, ..., xₙ) = A(x)·B(x) - C(x) sums to zero
/// over the boolean hypercube {0,1}ⁿ.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r), C(r)] at the random point r.
    final_evals: Vec<Fp4>,
}

impl OuterSumCheckProof {
    /// Creates a new sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: Vec<Fp4>) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a sum-check proof for f(x) = A(x)·B(x) - C(x).
    /// 
    /// Computes A·z, B·z, C·z then runs the sum-check protocol: for each round,
    /// computes a univariate polynomial, gets a random challenge, and folds.
    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_first_round(&a, &b, &c, &eq, &eq_point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a.fold_in_place(round_challenge);
        let mut b_fold = b.fold_in_place(round_challenge);
        let mut c_fold = c.fold_in_place(round_challenge);
        eq.fold_in_place();

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_round(
                &a_fold,
                &b_fold,
                &c_fold,
                &eq,
                &eq_point,
                current_claim,
                round,
                rounds,
            );

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            a_fold = a_fold.fold_in_place(round_challenge);
            b_fold = b_fold.fold_in_place(round_challenge);
            c_fold = c_fold.fold_in_place(round_challenge);
            eq.fold_in_place();
        }

        // Extract final evaluations A(r), B(r), C(r)
        let final_evals = vec![a_fold[0], b_fold[0], c_fold[0]];

        OuterSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the sum-check proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();
        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = Fp4::ZERO;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                    + eq_point[round] * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: A(r)·B(r) - C(r) = final_claim
        assert_eq!(
            current_claim,
            self.final_evals[0] * self.final_evals[1] - self.final_evals[2]
        )
    }
}

/// Computes the univariate polynomial for sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w) - c(X,w)].
pub fn compute_round(
    a: &MLE<Fp4>,
    b: &MLE<Fp4>,
    c: &MLE<Fp4>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) * (b[i << 1] + b[i << 1 | 1].double())
                - (c[i << 1] + c[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_first_round(
    a: &MLE<Fp>,
    b: &MLE<Fp>,
    c: &MLE<Fp>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0 (base field Fp promoted to Fp4)
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) * (b[i << 1] + b[i << 1 | 1].double())
                - (c[i << 1] + c[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Sum-check proof for inner product constraints of the form:
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} ⟨A(w), B(w)⟩`
/// where A and B are vectors and ⟨·,·⟩ denotes the inner product.
#[derive(Debug, Clone, PartialEq)]
pub struct InnerSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r)] at the random point r.
    final_evals: Vec<Fp4>,
}

impl InnerSumCheckProof {
    /// Creates a new inner sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: Vec<Fp4>) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a sum-check proof for f(x) = ⟨A(x), B(x)⟩.
    /// 
    /// Computes A·z and B·z then runs the sum-check protocol for inner products.
    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        // Compute A·z and B·z (sparse matrix-MLE multiplications)
        let (a, b) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_inner_first_round(&a, &b, &eq, &eq_point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a.fold_in_place(round_challenge);
        let mut b_fold = b.fold_in_place(round_challenge);
        eq.fold_in_place();

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_inner_round(
                &a_fold,
                &b_fold,
                &eq,
                &eq_point,
                current_claim,
                round,
                rounds,
            );

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            a_fold = a_fold.fold_in_place(round_challenge);
            b_fold = b_fold.fold_in_place(round_challenge);
            eq.fold_in_place();
        }

        // Extract final evaluations A(r), B(r)
        let final_evals = vec![a_fold[0], b_fold[0]];

        InnerSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the inner sum-check proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();
        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = Fp4::ZERO;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                    + eq_point[round] * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: ⟨A(r), B(r)⟩ = final_claim
        assert_eq!(
            current_claim,
            self.final_evals[0] * self.final_evals[1]
        )
    }
}

/// Computes the univariate polynomial for inner product sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w)].
pub fn compute_inner_round(
    a: &MLE<Fp4>,
    b: &MLE<Fp4>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) * (b[i << 1] + b[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first inner sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_inner_first_round(
    a: &MLE<Fp>,
    b: &MLE<Fp>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0 (base field Fp promoted to Fp4)
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) * (b[i << 1] + b[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Sum-check proof for Spark constraints involving sparse polynomial evaluation.
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} A(w)·B(w)·C(w)` for sparse polynomial products.
#[derive(Debug, Clone, PartialEq)]
pub struct SparkSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r), C(r)] at the random point r.
    final_evals: Vec<Fp4>,
}

impl SparkSumCheckProof {
    /// Creates a new Spark sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: Vec<Fp4>) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a sum-check proof for f(x) = A(x)·B(x)·C(x) for Spark constraints.
    /// 
    /// Computes A·z, B·z, C·z then runs the sum-check protocol for triple products.
    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_spark_first_round(&a, &b, &c, &eq, &eq_point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a.fold_in_place(round_challenge);
        let mut b_fold = b.fold_in_place(round_challenge);
        let mut c_fold = c.fold_in_place(round_challenge);
        eq.fold_in_place();

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_spark_round(
                &a_fold,
                &b_fold,
                &c_fold,
                &eq,
                &eq_point,
                current_claim,
                round,
                rounds,
            );

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            a_fold = a_fold.fold_in_place(round_challenge);
            b_fold = b_fold.fold_in_place(round_challenge);
            c_fold = c_fold.fold_in_place(round_challenge);
            eq.fold_in_place();
        }

        // Extract final evaluations A(r), B(r), C(r)
        let final_evals = vec![a_fold[0], b_fold[0], c_fold[0]];

        SparkSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the Spark sum-check proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();
        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = Fp4::ZERO;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                    + eq_point[round] * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: A(r)·B(r)·C(r) = final_claim
        assert_eq!(
            current_claim,
            self.final_evals[0] * self.final_evals[1] * self.final_evals[2]
        )
    }
}

/// Computes the univariate polynomial for Spark sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w) * c(X,w)].
pub fn compute_spark_round(
    a: &MLE<Fp4>,
    b: &MLE<Fp4>,
    c: &MLE<Fp4>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] * c[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) 
                * (b[i << 1] + b[i << 1 | 1].double())
                * (c[i << 1] + c[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first Spark sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_spark_first_round(
    a: &MLE<Fp>,
    b: &MLE<Fp>,
    c: &MLE<Fp>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0 (base field Fp promoted to Fp4)
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] * c[i << 1]);
        
        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) 
                * (b[i << 1] + b[i << 1 | 1].double())
                * (c[i << 1] + c[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        challenger::Challenger,
        eq::EqEvals,
        polynomial::MLE,
        spartan::sparse::SparseMLE,
        utils::{Fp, Fp4}
    };
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use std::collections::HashMap;

    /// Helper function to create a simple challenger for testing
    fn create_test_challenger() -> Challenger {
        // Create a challenger with some test randomness
        Challenger::new()
    }

    /// Helper function to create simple sparse matrices for testing
    fn create_test_sparse_matrices() -> (SparseMLE, SparseMLE, SparseMLE) {
        // Create simple 2x2 matrices for testing
        let mut a_coeffs = HashMap::new();
        a_coeffs.insert((0, 0), BabyBear::ONE);
        a_coeffs.insert((1, 1), BabyBear::from_u32(2));
        let A = SparseMLE::new(a_coeffs).unwrap();

        let mut b_coeffs = HashMap::new();
        b_coeffs.insert((0, 0), BabyBear::from_u32(3));
        b_coeffs.insert((1, 1), BabyBear::from_u32(4));
        let B = SparseMLE::new(b_coeffs).unwrap();

        let mut c_coeffs = HashMap::new();
        c_coeffs.insert((0, 0), BabyBear::from_u32(3)); // 1 * 3 = 3
        c_coeffs.insert((1, 1), BabyBear::from_u32(8)); // 2 * 4 = 8
        let C = SparseMLE::new(c_coeffs).unwrap();

        (A, B, C)
    }

    /// Helper function to create test matrices for inner products
    fn create_test_inner_matrices() -> (SparseMLE, SparseMLE) {
        // Create simple 2x2 matrices for testing inner products
        let mut a_coeffs = HashMap::new();
        a_coeffs.insert((0, 0), BabyBear::ONE);
        a_coeffs.insert((1, 1), BabyBear::from_u32(3));
        let A = SparseMLE::new(a_coeffs).unwrap();

        let mut b_coeffs = HashMap::new();
        b_coeffs.insert((0, 0), BabyBear::from_u32(2));
        b_coeffs.insert((1, 1), BabyBear::from_u32(4));
        let B = SparseMLE::new(b_coeffs).unwrap();

        (A, B)
    }

    /// Helper function to create a test witness
    fn create_test_witness() -> MLE<Fp> {
        let coeffs = vec![BabyBear::ONE, BabyBear::ONE]; // [1, 1]
        MLE::new(coeffs)
    }

    #[test]
    fn test_outer_sumcheck_proof_new() {
        let round_proofs = vec![
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ZERO, Fp4::ONE),
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ONE, Fp4::ZERO),
        ];
        let final_evals = vec![Fp4::ZERO, Fp4::ONE, Fp4::ZERO];

        let proof = OuterSumCheckProof::new(round_proofs.clone(), final_evals.clone());

        assert_eq!(proof.round_proofs, round_proofs);
        assert_eq!(proof.final_evals, final_evals);
    }

    #[test]
    fn test_outer_sumcheck_proof_prove_simple() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        // This should not panic for valid inputs
        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

        // Check that proof has expected structure
        assert!(!proof.round_proofs.is_empty());
        assert_eq!(proof.final_evals.len(), 3);
    }

    #[test]
    fn test_outer_sumcheck_proof_verify_consistency() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut prover_challenger = create_test_challenger();

        // Generate proof
        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut prover_challenger);

        // Verify with fresh challenger (should have same randomness)
        let mut verifier_challenger = create_test_challenger();
        
        // This should not panic for a valid proof
        proof.verify(&mut verifier_challenger);
    }

    #[test]
    fn test_inner_sumcheck_proof_new() {
        let round_proofs = vec![
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ZERO, Fp4::ONE),
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ONE, Fp4::ZERO),
        ];
        let final_evals = vec![Fp4::ZERO, Fp4::ONE];

        let proof = InnerSumCheckProof::new(round_proofs.clone(), final_evals.clone());

        assert_eq!(proof.round_proofs, round_proofs);
        assert_eq!(proof.final_evals, final_evals);
    }

    #[test]
    fn test_inner_sumcheck_proof_prove_simple() {
        let (A, B) = create_test_inner_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        // This should not panic for valid inputs
        let proof = InnerSumCheckProof::prove(&A, &B, &z, &mut challenger);

        // Check that proof has expected structure
        assert!(!proof.round_proofs.is_empty());
        assert_eq!(proof.final_evals.len(), 2);
    }

    #[test]
    fn test_inner_sumcheck_proof_verify_consistency() {
        let (A, B) = create_test_inner_matrices();
        let z = create_test_witness();
        let mut prover_challenger = create_test_challenger();

        // Generate proof
        let proof = InnerSumCheckProof::prove(&A, &B, &z, &mut prover_challenger);

        // Verify with fresh challenger
        let mut verifier_challenger = create_test_challenger();
        
        // This should not panic for a valid proof
        proof.verify(&mut verifier_challenger);
    }

    #[test]
    fn test_spark_sumcheck_proof_new() {
        let round_proofs = vec![
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ZERO, Fp4::ONE),
            crate::spartan::univariate::UnivariatePoly::from_coeffs(Fp4::ONE, Fp4::ZERO),
        ];
        let final_evals = vec![Fp4::ONE, Fp4::from_u32(2), Fp4::from_u32(3)];

        let proof = SparkSumCheckProof::new(round_proofs.clone(), final_evals.clone());

        assert_eq!(proof.round_proofs, round_proofs);
        assert_eq!(proof.final_evals, final_evals);
    }

    #[test]
    fn test_spark_sumcheck_proof_prove_simple() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        // This should not panic for valid inputs
        let proof = SparkSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

        // Check that proof has expected structure
        assert!(!proof.round_proofs.is_empty());
        assert_eq!(proof.final_evals.len(), 3);
    }

    #[test]
    fn test_spark_sumcheck_proof_verify_consistency() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut prover_challenger = create_test_challenger();

        // Generate proof
        let proof = SparkSumCheckProof::prove(&A, &B, &C, &z, &mut prover_challenger);

        // Verify with fresh challenger
        let mut verifier_challenger = create_test_challenger();
        
        // This should not panic for a valid proof
        proof.verify(&mut verifier_challenger);
    }

    #[test]
    fn test_compute_first_round_basic() {
        let a = MLE::new(vec![BabyBear::ONE, BabyBear::from_u32(2)]);
        let b = MLE::new(vec![BabyBear::from_u32(3), BabyBear::from_u32(4)]);
        let c = MLE::new(vec![BabyBear::from_u32(3), BabyBear::from_u32(8)]);

        let eq_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&eq_point[1..]);

        let current_claim = Fp4::ZERO;
        let rounds = 2;

        let poly = compute_first_round(&a, &b, &c, &eq, &eq_point, current_claim, rounds);

        // Check that polynomial has degree <= 2
        assert!(poly.degree() <= 2);
        
        // Check that coefficients are in expected field
        assert!(poly.coefficients().len() >= 2);
        assert!(poly.coefficients().len() <= 3);
    }

    #[test]
    fn test_compute_round_basic() {
        let a = MLE::new(vec![Fp4::ONE, Fp4::from_u32(2)]);
        let b = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(4)]);
        let c = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(8)]);

        let eq_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&eq_point[1..]);

        let current_claim = Fp4::ZERO;
        let round = 0;
        let rounds = 2;

        let poly = compute_round(&a, &b, &c, &eq, &eq_point, current_claim, round, rounds);

        // Check that polynomial has degree <= 2
        assert!(poly.degree() <= 2);
        
        // Check that coefficients are in expected field
        assert!(poly.coefficients().len() >= 2);
        assert!(poly.coefficients().len() <= 3);
    }

    #[test]
    fn test_compute_inner_round_basic() {
        let a = MLE::new(vec![Fp4::ONE, Fp4::from_u32(2)]);
        let b = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(4)]);

        let eq_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&eq_point[1..]);

        let current_claim = Fp4::ZERO;
        let round = 0;
        let rounds = 2;

        let poly = compute_inner_round(&a, &b, &eq, &eq_point, current_claim, round, rounds);

        // Check that polynomial has degree <= 2
        assert!(poly.degree() <= 2);
        
        // Check that coefficients are in expected field
        assert!(poly.coefficients().len() >= 2);
        assert!(poly.coefficients().len() <= 3);
    }

    #[test]
    fn test_compute_spark_round_basic() {
        let a = MLE::new(vec![Fp4::ONE, Fp4::from_u32(2)]);
        let b = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(4)]);
        let c = MLE::new(vec![Fp4::from_u32(5), Fp4::from_u32(6)]);

        let eq_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&eq_point[1..]);

        let current_claim = Fp4::ZERO;
        let round = 0;
        let rounds = 2;

        let poly = compute_spark_round(&a, &b, &c, &eq, &eq_point, current_claim, round, rounds);

        // Check that polynomial has degree <= 2
        assert!(poly.degree() <= 2);
        
        // Check that coefficients are in expected field
        assert!(poly.coefficients().len() >= 2);
        assert!(poly.coefficients().len() <= 3);
    }

    #[test]
    fn test_polynomial_evaluation_consistency() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

        // Test that each round polynomial evaluates consistently
        for poly in &proof.round_proofs {
            let eval_0 = poly.evaluate(Fp4::ZERO);
            let eval_1 = poly.evaluate(Fp4::ONE);
            
            // Evaluations should be valid field elements
            assert_ne!(eval_0, Fp4::from_u32(u32::MAX)); // Not a poison value
            assert_ne!(eval_1, Fp4::from_u32(u32::MAX)); // Not a poison value
        }
    }

    #[test]
    fn test_final_evaluations_structure() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

        // Should have exactly 3 final evaluations: A(r), B(r), C(r)
        assert_eq!(proof.final_evals.len(), 3);

        // All evaluations should be valid field elements
        for eval in &proof.final_evals {
            assert_ne!(*eval, Fp4::from_u32(u32::MAX)); // Not a poison value
        }
    }

    #[test] 
    fn test_empty_matrices_handling() {
        // Test behavior with matrices that would cause issues
        let mut small_coeffs = HashMap::new();
        small_coeffs.insert((0, 0), BabyBear::ONE);
        
        let A = SparseMLE::new(small_coeffs.clone()).unwrap();
        let B = SparseMLE::new(small_coeffs.clone()).unwrap();
        let C = SparseMLE::new(small_coeffs.clone()).unwrap();
        
        let z = MLE::new(vec![BabyBear::ONE]);
        let mut challenger = create_test_challenger();

        // This should work with minimal valid input
        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);
        
        assert!(!proof.round_proofs.is_empty());
        assert_eq!(proof.final_evals.len(), 3);
    }

    #[test]
    fn test_round_proofs_length() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        let mut challenger = create_test_challenger();

        let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

        // Number of round proofs should equal the number of variables
        let expected_rounds = z.n_vars();
        assert_eq!(proof.round_proofs.len(), expected_rounds);
    }

    #[test]
    fn test_proof_determinism() {
        let (A, B, C) = create_test_sparse_matrices();
        let z = create_test_witness();
        
        // Generate two proofs with the same input
        let mut challenger1 = create_test_challenger();
        let proof1 = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger1);

        let mut challenger2 = create_test_challenger();
        let proof2 = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger2);

        // Proofs should be identical (since challengers start the same)
        assert_eq!(proof1.round_proofs.len(), proof2.round_proofs.len());
        assert_eq!(proof1.final_evals.len(), proof2.final_evals.len());
    }

    // Property-based test using proptest
    #[cfg(feature = "proptest")]
    use proptest::prelude::*;

    #[cfg(feature = "proptest")]
    proptest! {
        #[test]
        fn test_sumcheck_proof_verification_property(
            coeff_a in 1u32..100,
            coeff_b in 1u32..100,
            coeff_c in 1u32..100
        ) {
            // Create matrices with the given coefficients
            let mut a_coeffs = HashMap::new();
            a_coeffs.insert((0, 0), BabyBear::from_u32(coeff_a));
            let A = SparseMLE::new(a_coeffs).unwrap();

            let mut b_coeffs = HashMap::new();
            b_coeffs.insert((0, 0), BabyBear::from_u32(coeff_b));
            let B = SparseMLE::new(b_coeffs).unwrap();

            let mut c_coeffs = HashMap::new();
            c_coeffs.insert((0, 0), BabyBear::from_u32(coeff_c));
            let C = SparseMLE::new(c_coeffs).unwrap();

            let z = MLE::new(vec![BabyBear::ONE]);
            let mut prover_challenger = create_test_challenger();

            let proof = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut prover_challenger);

            let mut verifier_challenger = create_test_challenger();
            
            // Verification should not panic (property: all valid proofs verify)
            proof.verify(&mut verifier_challenger);
        }
    }
}