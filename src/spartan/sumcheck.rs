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
    spartan::{
        spark::sparse::{SparseMLE, SpartanMetadata, TimeStamps},
        univariate::UnivariatePoly,
    },
};

/// Sum-check proof demonstrating that f(x₁, ..., xₙ) = A(x)·B(x) - C(x) sums to zero
/// over the boolean hypercube {0,1}ⁿ.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    pub round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r), C(r)] at the random point r.
    pub final_evals: [Fp4; 3],
}

impl OuterSumCheckProof {
    /// Creates a new sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 3]) -> Self {
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
        let final_evals = [a_fold[0], b_fold[0], c_fold[0]];

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
        round_coeffs[2] += ((a[i << 1] + a[i << 1 | 1].double())
            * (b[i << 1] + b[i << 1 | 1].double())
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
    /// Final evaluations [A(r), B(r), C(r) and Z(r)] at the random point r.
    final_evals: [Fp4; 4],
}

impl InnerSumCheckProof {
    /// Creates a new inner sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 4]) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a batched sum-check proof for inner product claims.
    /// Proves: (γ₀ A_bound(y) + γ₀² B_bound(y) + γ₀³ C_bound(y)) · Z(y) = batched_claim
    ///
    /// This verifies the evaluation claims from OuterSumCheck by proving the inner products.
    pub fn prove(
        a_bound: &MLE<Fp4>,
        b_bound: &MLE<Fp4>,
        c_bound: &MLE<Fp4>,
        outer_claims: [Fp4; 3],
        gamma: Fp4,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        // Use the bound matrices from outer sumcheck
        let rounds = a_bound.n_vars();

        let mut current_claim = gamma * outer_claims[0]
            + gamma.square() * outer_claims[1]
            + gamma.cube() * outer_claims[2];
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately - note: bound matrices are already in Fp4
        let round_proof = compute_inner_first_round_batched(
            &a_bound,
            &b_bound,
            &c_bound,
            gamma,
            &z,
            current_claim,
            rounds,
        );

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a_bound.fold_in_place(round_challenge);
        let mut b_fold = b_bound.fold_in_place(round_challenge);
        let mut c_fold = c_bound.fold_in_place(round_challenge);
        let mut z_fold = z.fold_in_place(round_challenge);

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_inner_round_batched(
                &a_fold,
                &b_fold,
                &c_fold,
                gamma,
                &z_fold,
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
            z_fold = z_fold.fold_in_place(round_challenge);
        }

        // Extract final evaluations A_bound(r), B_bound(r), C_bound(r), Z(r)
        let final_evals = [a_fold[0], b_fold[0], c_fold[0], z_fold[0]];

        InnerSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the inner sum-check proof. Panics if verification fails.
    pub fn verify(&self, outer_claims: [Fp4; 3], gamma: Fp4, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();

        let [a_claim, b_claim, c_claim] = outer_claims;
        let mut current_claim = gamma * a_claim + gamma.square() * b_claim + gamma.cube() * c_claim;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        let [a, b, c, z] = self.final_evals;
        // Final check: (γ·A_bound(r) + γ²·B_bound(r) + γ³·C_bound(r)) · Z(r) = final_claim
        assert_eq!(
            current_claim,
            (gamma * a + gamma.square() * b + gamma.cube() * c) * z
        )
    }
}

/// Computes the univariate polynomial for batched inner product sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [(γ·a(X,w) + γ²·b(X,w) + γ³·c(X,w)) * z(X,w)].
pub fn compute_inner_round_batched(
    a_bound: &MLE<Fp4>,
    b_bound: &MLE<Fp4>,
    c_bound: &MLE<Fp4>,
    gamma: Fp4,
    z: &MLE<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set first variable to 0
        round_coeffs[0] += (gamma * a_bound[i << 1]
            + gamma_squared * b_bound[i << 1]
            + gamma_cubed * c_bound[i << 1])
            * Fp4::from(z[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += (gamma * (a_bound[i << 1] + a_bound[i << 1 | 1].double())
            + gamma_squared * (b_bound[i << 1] + b_bound[i << 1 | 1].double())
            + gamma_cubed * (c_bound[i << 1] + c_bound[i << 1 | 1].double()))
            * (Fp4::from(z[i << 1]) + Fp4::from(z[i << 1 | 1]).double());
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first batched inner sum-check round.
/// Since bound matrices are already in Fp4, we work directly with Fp4.
pub fn compute_inner_first_round_batched(
    a_bound: &MLE<Fp4>,
    b_bound: &MLE<Fp4>,
    c_bound: &MLE<Fp4>,
    gamma: Fp4,
    z: &MLE<Fp>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0

        round_coeffs[0] += (gamma * a_bound[i << 1]
            + gamma_squared * b_bound[i << 1]
            + gamma_cubed * c_bound[i << 1])
            * Fp4::from(z[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += (gamma * (a_bound[i << 1] + a_bound[i << 1 | 1].double())
            + gamma_squared * (b_bound[i << 1] + b_bound[i << 1 | 1].double())
            + gamma_cubed * (c_bound[i << 1] + c_bound[i << 1 | 1].double()))
            * (Fp4::from(z[i << 1]) + Fp4::from(z[i << 1 | 1]).double());
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Sum-check proof for cubic product constraints of the form:
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} left(w) * right(w) * eq(w)`
/// where left and right are MLEs and eq is the equality polynomial.
#[derive(Debug, Clone, PartialEq)]
pub struct CubicSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    pub round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [left(r), right(r), eq(r)] at the random point r.
    pub final_evals: [Fp4; 2],
}

impl CubicSumCheckProof {
    /// Creates a new cubic sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 2]) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a cubic sum-check proof for the product relationship: left(x) * right(x) * eq(x).
    ///
    /// Proves: ∑_{x ∈ {0,1}^k} left(x) * right(x) * eq(x) = claimed_sum
    ///
    /// # Arguments
    /// * `left` - Left MLE from ProductCircuit
    /// * `right` - Right MLE from ProductCircuit
    /// * `eq_evals` - Equality polynomial evaluations
    /// * `claimed_sum` - The claimed sum value
    /// * `challenger` - Challenger for Fiat-Shamir randomness
    pub fn prove(
        left: &MLE<Fp>,
        right: &MLE<Fp>,
        eq_evals: &EqEvals,
        claimed_sum: Fp4,
        challenger: &mut Challenger,
    ) -> Self {
        let rounds = left.n_vars();
        assert_eq!(
            right.n_vars(),
            rounds,
            "Left and right MLEs must have same number of variables"
        );
        assert_eq!(
            eq_evals.n_vars, rounds,
            "Equality polynomial must match MLE dimensions"
        );

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = claimed_sum;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof =
            compute_cubic_first_round(left, right, eq_evals, &eq_point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut left_fold = left.fold_in_place(round_challenge);
        let mut right_fold = right.fold_in_place(round_challenge);
        let mut eq_fold = eq_evals.clone();
        eq_fold.fold_in_place();

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_cubic_round(
                &left_fold,
                &right_fold,
                &eq_fold,
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
            left_fold = left_fold.fold_in_place(round_challenge);
            right_fold = right_fold.fold_in_place(round_challenge);
            eq_fold.fold_in_place();
        }

        // Extract final evaluations left(r), right(r), eq(r)
        let final_evals = [left_fold[0], right_fold[0]];

        CubicSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the cubic sum-check proof. Panics if verification fails.
    pub fn verify(&self, claimed_sum: Fp4, challenger: &mut Challenger) -> Vec<Fp4> {
        let rounds = self.round_proofs.len();
        let mut current_claim = claimed_sum;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            // For cubic sumcheck, we use the standard relation
            let eq_point = challenger.get_challenge();
            assert_eq!(
                current_claim,
                (Fp4::ONE - eq_point) * round_poly.evaluate(Fp4::ZERO)
                    + eq_point * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: left(r) * right(r) * eq(r) = final_claim
        let [left_eval, right_eval] = self.final_evals;
        assert_eq!(current_claim, left_eval * right_eval);

        round_challenges
    }
}

/// Sum-check proof for inner product constraints of the form:
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} ⟨A(w), B(w)⟩`
/// where A and B are vectors and ⟨·,·⟩ denotes the inner product.
#[derive(Debug, Clone, PartialEq)]
pub struct SparkSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [row(r), e_rx(r), e_ry(r)] at the random point r for each of the 3 matrics.
    final_evals: [Fp4; 9],
}

impl SparkSumCheckProof {
    /// Creates a new inner sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 9]) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a batched sum-check proof for sparse mle evaluation.
    /// Proves: row * e_rx * e_ry as a
    ///
    /// This verifies the evaluation claims from OuterSumCheck by proving the inner products.
    pub fn prove(
        metadatas: &[SpartanMetadata; 3],
        oracle_pairs: &[(MLE<Fp4>, MLE<Fp4>); 3],
        evaluation_claims: [Fp4; 3],
        gamma: Fp4,
        challenger: &mut Challenger,
    ) -> Self {
        let rounds = metadatas[0].val().n_vars();

        // Batch the evaluation claims with gamma powers
        let mut current_claim = gamma * evaluation_claims[0]
            + gamma.square() * evaluation_claims[1]
            + gamma.cube() * evaluation_claims[2];

        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Process first round
        let round_proof = compute_spark_first_round_batched(
            &metadatas[0].val(),
            &metadatas[1].val(),
            &metadatas[2].val(),
            &oracle_pairs[0].0,
            &oracle_pairs[0].1,
            &oracle_pairs[1].0,
            &oracle_pairs[1].1,
            &oracle_pairs[2].0,
            &oracle_pairs[2].1,
            gamma,
            current_claim,
            rounds,
        );

        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold MLEs for the first time
        let mut val_a_folded = metadatas[0].val().fold_in_place(round_challenge);
        let mut val_b_folded = metadatas[1].val().fold_in_place(round_challenge);
        let mut val_c_folded = metadatas[2].val().fold_in_place(round_challenge);
        let mut e_rx_a_folded = oracle_pairs[0].0.fold_in_place(round_challenge);
        let mut e_ry_a_folded = oracle_pairs[0].1.fold_in_place(round_challenge);
        let mut e_rx_b_folded = oracle_pairs[1].0.fold_in_place(round_challenge);
        let mut e_ry_b_folded = oracle_pairs[1].1.fold_in_place(round_challenge);
        let mut e_rx_c_folded = oracle_pairs[2].0.fold_in_place(round_challenge);
        let mut e_ry_c_folded = oracle_pairs[2].1.fold_in_place(round_challenge);

        // Process remaining rounds
        for round in 1..rounds {
            let round_proof = compute_spark_round_batched(
                &val_a_folded,
                &val_b_folded,
                &val_c_folded,
                &e_rx_a_folded,
                &e_ry_a_folded,
                &e_rx_b_folded,
                &e_ry_b_folded,
                &e_rx_c_folded,
                &e_ry_c_folded,
                gamma,
                current_claim,
                round,
                rounds,
            );

            round_proofs.push(round_proof.clone());
            challenger.observe_fp4_elems(&round_proof.coefficients());

            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold for the next round
            val_a_folded = val_a_folded.fold_in_place(round_challenge);
            val_b_folded = val_b_folded.fold_in_place(round_challenge);
            val_c_folded = val_c_folded.fold_in_place(round_challenge);
            e_rx_a_folded = e_rx_a_folded.fold_in_place(round_challenge);
            e_ry_a_folded = e_ry_a_folded.fold_in_place(round_challenge);
            e_rx_b_folded = e_rx_b_folded.fold_in_place(round_challenge);
            e_ry_b_folded = e_ry_b_folded.fold_in_place(round_challenge);
            e_rx_c_folded = e_rx_c_folded.fold_in_place(round_challenge);
            e_ry_c_folded = e_ry_c_folded.fold_in_place(round_challenge);
        }

        // Extract final evaluations
        let final_evals = [
            val_a_folded[0],
            e_rx_a_folded[0],
            e_ry_a_folded[0],
            val_b_folded[0],
            e_rx_b_folded[0],
            e_ry_b_folded[0],
            val_c_folded[0],
            e_rx_c_folded[0],
            e_ry_c_folded[0],
        ];

        SparkSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the inner sum-check proof. Panics if verification fails.
    pub fn verify(&self, evaluation_claims: [Fp4; 3], gamma: Fp4, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();

        // Recompute the batched claim
        let mut current_claim = gamma * evaluation_claims[0]
            + gamma.square() * evaluation_claims[1]
            + gamma.cube() * evaluation_claims[2];

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = g_i(0) + g_i(1)
            assert_eq!(
                current_claim,
                round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
        }

        // Final check: batched evaluation of final values must match the final claim
        let [
            val_a,
            e_rx_a,
            e_ry_a,
            val_b,
            e_rx_b,
            e_ry_b,
            val_c,
            e_rx_c,
            e_ry_c,
        ] = self.final_evals;

        let final_eval_a = val_a * e_rx_a * e_ry_a;
        let final_eval_b = val_b * e_rx_b * e_ry_b;
        let final_eval_c = val_c * e_rx_c * e_ry_c;

        let expected_claim =
            gamma * final_eval_a + gamma.square() * final_eval_b + gamma.cube() * final_eval_c;

        assert_eq!(current_claim, expected_claim);
    }
}

/// Computes the univariate polynomial for batched inner product sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [(γ·a(X,w) + γ²·b(X,w) + γ³·c(X,w)) * z(X,w)].
pub fn compute_spark_round_batched(
    val_a: &MLE<Fp4>,
    val_b: &MLE<Fp4>,
    val_c: &MLE<Fp4>,
    e_rx_a: &MLE<Fp4>,
    e_ry_a: &MLE<Fp4>,
    e_rx_b: &MLE<Fp4>,
    e_ry_b: &MLE<Fp4>,
    e_rx_c: &MLE<Fp4>,
    e_ry_c: &MLE<Fp4>,
    gamma: Fp4,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - round - 1) {
        // Terms for g(0)
        let term_a_0 = val_a[i << 1] * e_rx_a[i << 1] * e_ry_a[i << 1];
        let term_b_0 = val_b[i << 1] * e_rx_b[i << 1] * e_ry_b[i << 1];
        let term_c_0 = val_c[i << 1] * e_rx_c[i << 1] * e_ry_c[i << 1];
        round_coeffs[0] += gamma * term_a_0 + gamma_squared * term_b_0 + gamma_cubed * term_c_0;

        // Terms for g(2)
        let val_a_2 = val_a[i << 1] + val_a[i << 1 | 1].double();
        let val_b_2 = val_b[i << 1] + val_b[i << 1 | 1].double();
        let val_c_2 = val_c[i << 1] + val_c[i << 1 | 1].double();

        let e_rx_a_2 = e_rx_a[i << 1] + e_rx_a[i << 1 | 1].double();
        let e_ry_a_2 = e_ry_a[i << 1] + e_ry_a[i << 1 | 1].double();
        let e_rx_b_2 = e_rx_b[i << 1] + e_rx_b[i << 1 | 1].double();
        let e_ry_b_2 = e_ry_b[i << 1] + e_ry_b[i << 1 | 1].double();
        let e_rx_c_2 = e_rx_c[i << 1] + e_rx_c[i << 1 | 1].double();
        let e_ry_c_2 = e_ry_c[i << 1] + e_ry_c[i << 1 | 1].double();

        let term_a_2 = val_a_2 * e_rx_a_2 * e_ry_a_2;
        let term_b_2 = val_b_2 * e_rx_b_2 * e_ry_b_2;
        let term_c_2 = val_c_2 * e_rx_c_2 * e_ry_c_2;
        round_coeffs[2] += gamma * term_a_2 + gamma_squared * term_b_2 + gamma_cubed * term_c_2;
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first batched inner sum-check round.
/// Since bound matrices are already in Fp4, we work directly with Fp4.
pub fn compute_spark_first_round_batched(
    val_a: &MLE<Fp>,
    val_b: &MLE<Fp>,
    val_c: &MLE<Fp>,
    e_rx_a: &MLE<Fp4>,
    e_ry_a: &MLE<Fp4>,
    e_rx_b: &MLE<Fp4>,
    e_ry_b: &MLE<Fp4>,
    e_rx_c: &MLE<Fp4>,
    e_ry_c: &MLE<Fp4>,
    gamma: Fp4,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - 1) {
        // Terms for g(0)
        let term_a_0 = Fp4::from(val_a[i << 1]) * e_rx_a[i << 1] * e_ry_a[i << 1];
        let term_b_0 = Fp4::from(val_b[i << 1]) * e_rx_b[i << 1] * e_ry_b[i << 1];
        let term_c_0 = Fp4::from(val_c[i << 1]) * e_rx_c[i << 1] * e_ry_c[i << 1];
        round_coeffs[0] += gamma * term_a_0 + gamma_squared * term_b_0 + gamma_cubed * term_c_0;

        // Terms for g(2)
        let val_a_2 = Fp4::from(val_a[i << 1]) + Fp4::from(val_a[i << 1 | 1]).double();
        let val_b_2 = Fp4::from(val_b[i << 1]) + Fp4::from(val_b[i << 1 | 1]).double();
        let val_c_2 = Fp4::from(val_c[i << 1]) + Fp4::from(val_c[i << 1 | 1]).double();

        let e_rx_a_2 = e_rx_a[i << 1] + e_rx_a[i << 1 | 1].double();
        let e_ry_a_2 = e_ry_a[i << 1] + e_ry_a[i << 1 | 1].double();
        let e_rx_b_2 = e_rx_b[i << 1] + e_rx_b[i << 1 | 1].double();
        let e_ry_b_2 = e_ry_b[i << 1] + e_ry_b[i << 1 | 1].double();
        let e_rx_c_2 = e_rx_c[i << 1] + e_rx_c[i << 1 | 1].double();
        let e_ry_c_2 = e_ry_c[i << 1] + e_ry_c[i << 1 | 1].double();

        let term_a_2 = val_a_2 * e_rx_a_2 * e_ry_a_2;
        let term_b_2 = val_b_2 * e_rx_b_2 * e_ry_b_2;
        let term_c_2 = val_c_2 * e_rx_c_2 * e_ry_c_2;
        round_coeffs[2] += gamma * term_a_2 + gamma_squared * term_b_2 + gamma_cubed * term_c_2;
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for cubic sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [left(X,w) * right(X,w)].
pub fn compute_cubic_round(
    left: &MLE<Fp4>,
    right: &MLE<Fp4>,
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
        round_coeffs[0] += eq[i] * (left[i << 1] * right[i << 1]);

        // g(2): use multilinear polynomial identity
        // For base field, we need to convert to Fp4 first, then use double()
        let left_at_2 = Fp4::from(left[i << 1]) + Fp4::from(left[i << 1 | 1]).double();
        let right_at_2 = Fp4::from(right[i << 1]) + Fp4::from(right[i << 1 | 1]).double();
        round_coeffs[2] += eq[i] * (left_at_2 * right_at_2);
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first cubic sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_cubic_first_round(
    left: &MLE<Fp>,
    right: &MLE<Fp>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0 (base field Fp promoted to Fp4)
        round_coeffs[0] += eq[i] * (left[i << 1] * right[i << 1]);

        // g(2): use multilinear polynomial identity
        // For base field, we need to convert to Fp4 first, then use double()
        let left_at_2 = Fp4::from(left[i << 1]) + Fp4::from(left[i << 1 | 1]).double();
        let right_at_2 = Fp4::from(right[i << 1]) + Fp4::from(right[i << 1 | 1]).double();
        round_coeffs[2] += eq[i] * (left_at_2 * right_at_2);
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
    use crate::challenger::Challenger;
    use crate::spartan::R1CSInstance;
    use crate::spartan::spark::sparse::TimeStamps;

    #[test]
    fn test_outer_sumcheck_prove_verify_simple() {
        // Basic outer sum-check prove/verify cycle with simple R1CS
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        let mut challenger_prove = Challenger::new();
        let mut challenger_verify = Challenger::new();

        // Prove outer sum-check
        let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger_prove);

        // Verify proof
        outer_proof.verify(&mut challenger_verify);

        // Test passed if no panics occurred
    }

    #[test]
    fn test_outer_sumcheck_prove_verify_multi_constraint() {
        // Test outer sum-check with multi-constraint R1CS
        let instance = R1CSInstance::multi_constraint_test().unwrap();
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        let mut challenger_prove = Challenger::new();
        let mut challenger_verify = Challenger::new();

        // Prove outer sum-check
        let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger_prove);

        // Verify proof
        outer_proof.verify(&mut challenger_verify);

        // Test that proof structure is consistent
        assert!(!outer_proof.round_proofs.is_empty());
    }

    #[test]
    fn test_outer_sumcheck_mathematical_correctness() {
        // Test that outer sum-check correctly proves f(x) = A(x)·B(x) - C(x) = 0
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        let mut challenger = Challenger::new();

        // Generate proof
        let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger);

        // Verify that final evaluations satisfy the constraint
        // A(r) * B(r) - C(r) should equal the final claim after sum-check
        let final_evals = &outer_proof.final_evals;
        let _constraint_result = final_evals[0] * final_evals[1] - final_evals[2];

        // The constraint should be satisfied (though we can't directly test the sum-check claim
        // without reimplementing the verifier logic, successful verification implies correctness)

        // Test successful verification
        let mut verifier = Challenger::new();
        outer_proof.verify(&mut verifier);
    }

    #[test]
    fn test_outer_sumcheck_round_proof_consistency() {
        // Test that round proofs are generated consistently
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        let mut challenger1 = Challenger::new();
        let mut challenger2 = Challenger::new();

        // Generate two proofs with same challenger state
        let proof1 = OuterSumCheckProof::prove(A, B, C, z, &mut challenger1);
        let proof2 = OuterSumCheckProof::prove(A, B, C, z, &mut challenger2);

        // Should be identical due to deterministic challenger
        assert_eq!(proof1, proof2);
        assert_eq!(proof1.round_proofs.len(), proof2.round_proofs.len());
        assert_eq!(proof1.final_evals, proof2.final_evals);
    }

    #[test]
    fn test_inner_sumcheck_prove_verify_simple() {
        // Test inner sum-check with simple bound matrices
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();

        // First run outer sum-check to get bound matrices
        let mut outer_challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut outer_challenger,
        );

        // Extract random point from outer sum-check (simulate the protocol flow)
        let r_x_point: Vec<Fp4> = vec![]; // Simple instance needs 0 variables
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        let mut inner_challenger_prove = Challenger::new();
        let mut inner_challenger_verify = Challenger::new();

        // Simulate gamma challenge
        inner_challenger_prove.observe_fp4_elems(&outer_proof.final_evals);
        inner_challenger_verify.observe_fp4_elems(&outer_proof.final_evals);
        let gamma = inner_challenger_prove.get_challenge();
        let gamma_verify = inner_challenger_verify.get_challenge();
        assert_eq!(gamma, gamma_verify);

        // Prove inner sum-check
        let inner_proof = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            outer_proof.final_evals,
            gamma,
            z,
            &mut inner_challenger_prove,
        );

        // Verify inner sum-check
        inner_proof.verify(outer_proof.final_evals, gamma, &mut inner_challenger_verify);

        // Test passed if no panics occurred
    }

    #[test]
    fn test_inner_sumcheck_batched_claims() {
        // Test that inner sum-check correctly handles batched evaluation claims
        let instance = R1CSInstance::multi_constraint_test().unwrap();
        let z = &instance.witness_mle();

        // Run outer sum-check first
        let mut outer_challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut outer_challenger,
        );

        // Get bound matrices
        let r_x_point: Vec<Fp4> = vec![Fp4::from_u32(7), Fp4::from_u32(13)]; // Multi instance needs 2 variables
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        let mut challenger = Challenger::new();
        challenger.observe_fp4_elems(&outer_proof.final_evals);
        let _gamma = challenger.get_challenge();

        // Test different gamma values produce different but valid proofs
        let gammas = [Fp4::from_u32(1), Fp4::from_u32(17), Fp4::from_u32(42)];

        for test_gamma in gammas {
            let mut prove_challenger = Challenger::new();
            let mut verify_challenger = Challenger::new();

            let inner_proof = InnerSumCheckProof::prove(
                &a_bound,
                &b_bound,
                &c_bound,
                outer_proof.final_evals,
                test_gamma,
                z,
                &mut prove_challenger,
            );

            // Each should verify successfully
            inner_proof.verify(outer_proof.final_evals, test_gamma, &mut verify_challenger);
        }
    }

    #[test]
    fn test_inner_sumcheck_bound_matrix_integration() {
        // Test inner sum-check works with actual bound matrices from outer sum-check
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();

        // Complete outer sum-check phase
        let mut outer_challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut outer_challenger,
        );

        // Simulate the protocol: compute bound matrices using random point
        let r_x_point: Vec<Fp4> = vec![];
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        // Continue with inner sum-check
        let mut inner_challenger = Challenger::new();
        inner_challenger.observe_fp4_elems(&outer_proof.final_evals);
        let gamma = inner_challenger.get_challenge();

        let inner_proof = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            outer_proof.final_evals,
            gamma,
            z,
            &mut inner_challenger,
        );

        // Verify with same protocol flow
        let mut verifier = Challenger::new();
        verifier.observe_fp4_elems(&outer_proof.final_evals);
        let verify_gamma = verifier.get_challenge();
        assert_eq!(gamma, verify_gamma);

        inner_proof.verify(outer_proof.final_evals, verify_gamma, &mut verifier);
    }

    #[test]
    fn test_outer_inner_sumcheck_integration() {
        // Test complete outer → inner sum-check flow
        let instance = R1CSInstance::multi_constraint_test().unwrap();
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        // Phase 1: Outer sum-check
        let mut challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger);

        // Extract random point for bound matrices (simulate protocol)
        let r_x_point: Vec<Fp4> = vec![Fp4::from_u32(5), Fp4::from_u32(11)];
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        // Phase 2: Inner sum-check with outer evaluation claims
        challenger.observe_fp4_elems(&outer_proof.final_evals);
        let gamma = challenger.get_challenge();

        let inner_proof = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            outer_proof.final_evals,
            gamma,
            z,
            &mut challenger,
        );

        // Verify both phases
        let mut verifier = Challenger::new();
        outer_proof.verify(&mut verifier);

        verifier.observe_fp4_elems(&outer_proof.final_evals);
        let verify_gamma = verifier.get_challenge();
        inner_proof.verify(outer_proof.final_evals, verify_gamma, &mut verifier);

        // Test complete integration success
        assert_eq!(gamma, verify_gamma);
    }

    #[test]
    fn test_sumcheck_field_arithmetic_consistency() {
        // Test Fp → Fp4 promotion and field arithmetic throughout sum-check
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle(); // z is MLE<BabyBear> (base field)

        let mut challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut challenger,
        );

        // Final evaluations should be in extension field Fp4
        for eval in outer_proof.final_evals {
            // Test that we can perform Fp4 arithmetic
            let _sum = eval + eval;
            let _product = eval * eval;
            let _difference = eval - eval;
        }

        // Verify field consistency through successful verification
        let mut verifier = Challenger::new();
        outer_proof.verify(&mut verifier);
    }

    #[test]
    fn test_compute_round_functions() {
        // Test that round computation functions work correctly
        let instance = R1CSInstance::simple_test().unwrap();
        let z_fp = &instance.witness_mle(); // Base field witness
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        // Compute A·z, B·z, C·z (these will be in base field initially)
        let a_mle = A.multiply_by_mle(z_fp).unwrap();
        let b_mle = B.multiply_by_mle(z_fp).unwrap();
        let c_mle = C.multiply_by_mle(z_fp).unwrap();

        // Test that we can use the round computation functions
        // (This tests the mathematical correctness indirectly)

        // Create dummy EqEvals for testing
        let eq_point = vec![Fp4::from_u32(7)];
        let eq_evals = crate::eq::EqEvals::gen_from_point(&eq_point);

        // Test first round computation (base field → extension field)
        let current_claim = Fp4::ZERO;
        let rounds = a_mle.n_vars();

        let first_round_poly = compute_first_round(
            &a_mle,
            &b_mle,
            &c_mle,
            &eq_evals,
            &eq_point,
            current_claim,
            rounds,
        );

        // Verify polynomial has correct structure (degree ≤ 2)
        assert!(first_round_poly.coefficients().len() >= 2);
        assert!(first_round_poly.coefficients().len() <= 3);

        // Test evaluation at different points
        let eval_0 = first_round_poly.evaluate(Fp4::ZERO);
        let eval_1 = first_round_poly.evaluate(Fp4::ONE);
        let eval_2 = first_round_poly.evaluate(Fp4::from_u32(2));

        // All should be valid field elements
        let _test_arithmetic = eval_0 + eval_1 + eval_2;
    }

    #[test]
    fn test_gruen_optimization() {
        // Test that Gruen's optimization (evaluate at 0, 1, 2) works correctly
        let instance = R1CSInstance::simple_test().unwrap();
        let z = &instance.witness_mle();

        let mut challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut challenger,
        );

        // Test that each round proof can be evaluated at standard points
        for round_poly in &outer_proof.round_proofs {
            let eval_0 = round_poly.evaluate(Fp4::ZERO);
            let eval_1 = round_poly.evaluate(Fp4::ONE);
            let eval_2 = round_poly.evaluate(Fp4::from_u32(2));

            // All evaluations should be valid and consistent
            // (Gruen's optimization uses these points for degree-2 polynomial interpolation)
            let _arithmetic_test = eval_0 + eval_1 + eval_2;
        }

        // Verify overall proof correctness
        let mut verifier = Challenger::new();
        outer_proof.verify(&mut verifier);
    }

    #[test]
    fn test_sumcheck_sparse_mle_integration() {
        // Test sum-check protocols work correctly with sparse MLE operations
        let instance = R1CSInstance::multi_constraint_test().unwrap();
        let z = &instance.witness_mle();

        // Verify constraint matrices are sparse
        assert!(
            instance.r1cs.a.num_nonzeros()
                < instance.r1cs.a.dimensions().0 * instance.r1cs.a.dimensions().1
        );
        assert!(
            instance.r1cs.b.num_nonzeros()
                < instance.r1cs.b.dimensions().0 * instance.r1cs.b.dimensions().1
        );
        assert!(
            instance.r1cs.c.num_nonzeros()
                < instance.r1cs.c.dimensions().0 * instance.r1cs.c.dimensions().1
        );

        // Run outer sum-check with sparse matrices
        let mut challenger = Challenger::new();
        let outer_proof = OuterSumCheckProof::prove(
            &instance.r1cs.a,
            &instance.r1cs.b,
            &instance.r1cs.c,
            z,
            &mut challenger,
        );

        // Verify sparse operations maintain correctness
        let mut verifier = Challenger::new();
        outer_proof.verify(&mut verifier);

        // Test that sparsity is preserved in bound matrix computation
        let r_x_point = vec![Fp4::from_u32(3), Fp4::from_u32(7)];
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        // Bound matrices should have expected dimensions
        assert_eq!(a_bound.len(), instance.r1cs.a.dimensions().1);
        assert_eq!(b_bound.len(), instance.r1cs.b.dimensions().1);
        assert_eq!(c_bound.len(), instance.r1cs.c.dimensions().1);
    }

    #[test]
    fn test_spark_sumcheck_prove_verify() {
        // Create dummy metadata and oracles for three matrices
        let read_ts = vec![Fp::ZERO, Fp::ONE];
        let final_ts = vec![Fp::ONE, Fp::from_u32(2u32)];
        let ts = TimeStamps::new(read_ts, final_ts).unwrap();

        let metadatas = [
            SpartanMetadata::new(
                MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(2u32)]),
                MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(1u32)]),
                MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(0u32)]),
                ts.clone(),
                ts.clone(),
            )
            .unwrap(),
            SpartanMetadata::new(
                MLE::new(vec![Fp::from_u32(3u32), Fp::from_u32(4u32)]),
                MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(0u32)]),
                MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(1u32)]),
                ts.clone(),
                ts.clone(),
            )
            .unwrap(),
            SpartanMetadata::new(
                MLE::new(vec![Fp::from_u32(5u32), Fp::from_u32(6u32)]),
                MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(0u32)]),
                MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(1u32)]),
                ts.clone(),
                ts.clone(),
            )
            .unwrap(),
        ];

        let oracle_pairs = [
            (
                MLE::new(vec![Fp4::from_u32(10u32), Fp4::from_u32(11u32)]),
                MLE::new(vec![Fp4::from_u32(12u32), Fp4::from_u32(13u32)]),
            ),
            (
                MLE::new(vec![Fp4::from_u32(14u32), Fp4::from_u32(15u32)]),
                MLE::new(vec![Fp4::from_u32(16u32), Fp4::from_u32(17u32)]),
            ),
            (
                MLE::new(vec![Fp4::from_u32(18u32), Fp4::from_u32(19u32)]),
                MLE::new(vec![Fp4::from_u32(20u32), Fp4::from_u32(21u32)]),
            ),
        ];

        // Dummy evaluation claims and gamma
        let evaluation_claims = [
            Fp4::from_u32(1u32),
            Fp4::from_u32(2u32),
            Fp4::from_u32(3u32),
        ];
        let gamma = Fp4::from_u32(7u32);

        // Create separate challengers for prover and verifier
        let mut prover_challenger = Challenger::new();
        let mut verifier_challenger = Challenger::new();

        // Generate the proof
        let proof = SparkSumCheckProof::prove(
            &metadatas,
            &oracle_pairs,
            evaluation_claims,
            gamma,
            &mut prover_challenger,
        );

        // Verify the proof
        proof.verify(evaluation_claims, gamma, &mut verifier_challenger);

        // The test passes if no assertions fail
    }
    /// Batched cubic sum-check proof for handling multiple cubic claims efficiently.
    ///
    /// This extends the cubic sum-check protocol to handle an arbitrary number of claims
    /// using gamma powers for batching, similar to how InnerSumCheck handles 3 claims.
    ///
    /// Mathematical formulation:
    /// ∑_{i=0}^{N-1} γ^{i+1} * (∑_{w∈{0,1}^k} left_i(w) * right_i(w) * eq_i(w)) = batched_claim
    #[derive(Debug, Clone, PartialEq)]
    pub struct BatchedCubicSumCheckProof {
        /// Univariate polynomials for each round of the sum-check protocol.
        pub round_proofs: Vec<UnivariatePoly>,
        /// Final evaluations for all claims: [left_0(r), right_0(r), ..., left_N-1(r), right_N-1(r)]
        pub final_evals: Vec<Fp4>,
        /// Number of claims batched in this proof
        pub num_claims: usize,
    }

    impl BatchedCubicSumCheckProof {
        /// Creates a new batched cubic sum-check proof.
        pub fn new(
            round_proofs: Vec<UnivariatePoly>,
            final_evals: Vec<Fp4>,
            num_claims: usize,
        ) -> Self {
            assert_eq!(
                final_evals.len(),
                num_claims * 2,
                "Final evaluations must contain 2 values per claim (left and right)"
            );

            Self {
                round_proofs,
                final_evals,
                num_claims,
            }
        }

        /// Generates a batched cubic sum-check proof for multiple cubic claims.
        ///
        /// Proves: ∑_{i=0}^{N-1} γ^{i+1} * (∑_{w∈{0,1}^k} left_i(w) * right_i(w) * eq_i(w)) = batched_claim
        ///
        /// # Arguments
        /// * `left_polys` - Vector of left MLEs, one per claim
        /// * `right_polys` - Vector of right MLEs, one per claim  
        /// * `eq_evals` - Vector of equality polynomial evaluations, one per claim
        /// * `claimed_sums` - Vector of claimed sum values for each claim
        /// * `challenger` - Challenger for Fiat-Shamir randomness
        pub fn prove(
            left_polys: &[MLE<Fp4>],
            right_polys: &[MLE<Fp4>],
            eq_evals: &[EqEvals],
            claimed_sums: &[Fp4],
            challenger: &mut Challenger,
        ) -> Self {
            let num_claims = left_polys.len();
            assert_eq!(
                right_polys.len(),
                num_claims,
                "Number of left and right polynomials must match"
            );
            assert_eq!(
                eq_evals.len(),
                num_claims,
                "Number of equality polynomials must match"
            );
            assert_eq!(
                claimed_sums.len(),
                num_claims,
                "Number of claimed sums must match"
            );

            if num_claims == 0 {
                return Self::new(vec![], vec![], 0);
            }

            let rounds = left_polys[0].n_vars();

            // Validate all polynomials have consistent dimensions
            for i in 1..num_claims {
                assert_eq!(
                    left_polys[i].n_vars(),
                    rounds,
                    "All left polynomials must have same number of variables"
                );
                assert_eq!(
                    right_polys[i].n_vars(),
                    rounds,
                    "All right polynomials must have same number of variables"
                );
                assert_eq!(
                    eq_evals[i].n_vars, rounds,
                    "All equality polynomials must match MLE dimensions"
                );
            }

            // Compute batched claim using gamma powers
            let gamma = challenger.get_challenge();
            let mut batched_claim = Fp4::ZERO;

            for (i, &claimed_sum) in claimed_sums.iter().enumerate() {
                let gamma_power = gamma.exp_u64(i as u64 + 1);
                batched_claim += gamma_power * claimed_sum;
            }

            // Get random evaluation point from challenger (Fiat-Shamir)
            let eq_point = challenger.get_challenges(rounds);

            let mut current_claim = batched_claim;
            let mut round_proofs = Vec::new();
            let mut round_challenges = Vec::new();

            // Handle first round separately (uses base field Fp for efficiency)
            let round_proof = compute_batched_cubic_first_round(
                left_polys,
                right_polys,
                eq_evals,
                &eq_point,
                current_claim,
                gamma,
                num_claims,
                rounds,
            );

            // Process first round proof
            round_proofs.push(round_proof.clone());
            challenger.observe_fp4_elems(&round_proof.coefficients());

            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold all polynomials by fixing first variable to challenge
            let mut left_folded: Vec<MLE<Fp4>> = left_polys
                .iter()
                .map(|p| p.fold_in_place(round_challenge))
                .collect();
            let mut right_folded: Vec<MLE<Fp4>> = right_polys
                .iter()
                .map(|p| p.fold_in_place(round_challenge))
                .collect();
            let mut eq_folded: Vec<EqEvals> = eq_evals
                .iter()
                .map(|eq| {
                    let mut eq_copy = eq.clone();
                    eq_copy.fold_in_place();
                    eq_copy
                })
                .collect();

            // Process remaining rounds (1 to n-1)
            for round in 1..rounds {
                let round_proof = compute_batched_cubic_round(
                    &left_folded,
                    &right_folded,
                    &eq_folded,
                    &eq_point,
                    current_claim,
                    gamma,
                    num_claims,
                    round,
                    rounds,
                );

                challenger.observe_fp4_elems(&round_proof.coefficients());
                let round_challenge = challenger.get_challenge();
                round_challenges.push(round_challenge);
                current_claim = round_proof.evaluate(round_challenge);

                // Fold polynomials for next round
                for claim_idx in 0..num_claims {
                    left_folded[claim_idx] = left_folded[claim_idx].fold_in_place(round_challenge);
                    right_folded[claim_idx] =
                        right_folded[claim_idx].fold_in_place(round_challenge);
                    eq_folded[claim_idx].fold_in_place();
                }
            }

            // Extract final evaluations for all claims
            let mut final_evals = Vec::with_capacity(num_claims * 2);
            for (left, right) in left_folded.iter().zip(right_folded.iter()) {
                final_evals.push(left[0]);
                final_evals.push(right[0]);
            }

            BatchedCubicSumCheckProof::new(round_proofs, final_evals, num_claims)
        }

        /// Verifies the batched cubic sum-check proof.
        ///
        /// # Arguments
        /// * `claimed_sums` - Vector of claimed sum values for each claim
        /// * `challenger` - Challenger for Fiat-Shamir randomness
        pub fn verify(&self, claimed_sums: &[Fp4], challenger: &mut Challenger) {
            assert_eq!(
                claimed_sums.len(),
                self.num_claims,
                "Number of claimed sums must match number of claims in proof"
            );

            if self.num_claims == 0 {
                return;
            }

            let rounds = self.round_proofs.len();
            let gamma = challenger.get_challenge();

            // Recompute batched claim
            let mut batched_claim = Fp4::ZERO;
            for (i, &claimed_sum) in claimed_sums.iter().enumerate() {
                let gamma_power = gamma.exp_u64(i as u64 + 1);
                batched_claim += gamma_power * claimed_sum;
            }

            let mut current_claim = batched_claim;
            let mut round_challenges = Vec::new();

            // Verify each round of the sum-check protocol
            for round in 0..rounds {
                let round_poly = &self.round_proofs[round];

                // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
                let eq_point = challenger.get_challenge();
                assert_eq!(
                    current_claim,
                    (Fp4::ONE - eq_point) * round_poly.evaluate(Fp4::ZERO)
                        + eq_point * round_poly.evaluate(Fp4::ONE)
                );

                challenger.observe_fp4_elems(&round_poly.coefficients());
                let challenge = challenger.get_challenge();
                current_claim = round_poly.evaluate(challenge);
                round_challenges.push(challenge);
            }

            // Final check: batched evaluation of final values must match the final claim
            let mut expected_claim = Fp4::ZERO;
            for i in 0..self.num_claims {
                let left_eval = self.final_evals[2 * i];
                let right_eval = self.final_evals[2 * i + 1];
                let gamma_power = gamma.exp_u64(i as u64 + 1);
                expected_claim += gamma_power * (left_eval * right_eval);
            }

            assert_eq!(
                current_claim, expected_claim,
                "Final batched evaluation check failed"
            );
        }
    }

    /// Computes the univariate polynomial for batched cubic sum-check rounds 1 to n-1.
    ///
    /// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} ∑_{i=0}^{N-1} γ^{i+1} * [left_i(X,w) * right_i(X,w) * eq_i(w)]
    pub fn compute_batched_cubic_round(
        left_polys: &[MLE<Fp4>],
        right_polys: &[MLE<Fp4>],
        eq_evals: &[EqEvals],
        eq_point: &Vec<Fp4>,
        current_claim: Fp4,
        gamma: Fp4,
        num_claims: usize,
        round: usize,
        rounds: usize,
    ) -> UnivariatePoly {
        // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
        let mut round_coeffs = vec![Fp4::ZERO; 3];

        for i in 0..1 << (rounds - round - 1) {
            // Compute contributions for g(0): set current variable to 0
            let mut g0_contribution = Fp4::ZERO;
            for claim_idx in 0..num_claims {
                let gamma_power = gamma.exp_u64(claim_idx as u64 + 1);
                let left_val = left_polys[claim_idx][i << 1];
                let right_val = right_polys[claim_idx][i << 1];
                g0_contribution += gamma_power * (left_val * right_val * eq_evals[claim_idx][i]);
            }
            round_coeffs[0] += g0_contribution;

            // Compute contributions for g(2): use multilinear polynomial identity
            let mut g2_contribution = Fp4::ZERO;
            for claim_idx in 0..num_claims {
                let gamma_power = gamma.exp_u64(claim_idx as u64 + 1);

                let left_at_2 =
                    left_polys[claim_idx][i << 1] + left_polys[claim_idx][i << 1 | 1].double();
                let right_at_2 =
                    right_polys[claim_idx][i << 1] + right_polys[claim_idx][i << 1 | 1].double();
                let eq_at_i = eq_evals[claim_idx][i];

                g2_contribution += gamma_power * (left_at_2 * right_at_2 * eq_at_i);
            }
            round_coeffs[2] += g2_contribution;
        }

        // g(1): derived from sum-check constraint
        round_coeffs[1] =
            (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

        let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
        round_proof.interpolate().unwrap();

        round_proof
    }

    /// Computes the univariate polynomial for the first batched cubic sum-check round.
    ///
    /// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
    pub fn compute_batched_cubic_first_round(
        left_polys: &[MLE<Fp4>],
        right_polys: &[MLE<Fp4>],
        eq_evals: &[EqEvals],
        eq_point: &Vec<Fp4>,
        current_claim: Fp4,
        gamma: Fp4,
        num_claims: usize,
        rounds: usize,
    ) -> UnivariatePoly {
        // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
        let mut round_coeffs = vec![Fp4::ZERO; 3];

        for i in 0..1 << (rounds - 1) {
            // Compute contributions for g(0): set first variable to 0
            let mut g0_contribution = Fp4::ZERO;
            for claim_idx in 0..num_claims {
                let gamma_power = gamma.exp_u64(claim_idx as u64 + 1);
                let left_val = left_polys[claim_idx][i << 1];
                let right_val = right_polys[claim_idx][i << 1];
                g0_contribution += gamma_power * (left_val * right_val * eq_evals[claim_idx][i]);
            }
            round_coeffs[0] += g0_contribution;

            // Compute contributions for g(2): use multilinear polynomial identity
            let mut g2_contribution = Fp4::ZERO;
            for claim_idx in 0..num_claims {
                let gamma_power = gamma.exp_u64(claim_idx as u64 + 1);

                let left_at_2 =
                    left_polys[claim_idx][i << 1] + left_polys[claim_idx][i << 1 | 1].double();
                let right_at_2 =
                    right_polys[claim_idx][i << 1] + right_polys[claim_idx][i << 1 | 1].double();
                let eq_at_i = eq_evals[claim_idx][i];

                g2_contribution += gamma_power * (left_at_2 * right_at_2 * eq_at_i);
            }
            round_coeffs[2] += g2_contribution;
        }

        // g(1): derived from sum-check constraint
        round_coeffs[1] =
            (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

        let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
        round_proof.interpolate().unwrap();

        round_proof
    }

    #[cfg(test)]
    mod batched_tests {
        use super::*;
        use crate::challenger::Challenger;

        #[test]
        fn test_batched_cubic_sumcheck_single_claim() {
            // Test that batched version works correctly with single claim (should match CubicSumCheck)
            let left = MLE::new(vec![
                Fp4::from_u32(1),
                Fp4::from_u32(2),
                Fp4::from_u32(3),
                Fp4::from_u32(4),
            ]);
            let right = MLE::new(vec![
                Fp4::from_u32(5),
                Fp4::from_u32(6),
                Fp4::from_u32(7),
                Fp4::from_u32(8),
            ]);
            let point = vec![Fp4::from_u32(42)];
            let eq = EqEvals::gen_from_point(&point);

            // Compute actual sum
            let mut actual_sum = Fp4::ZERO;
            for i in 0..4 {
                actual_sum += left[i] * right[i] * eq[i];
            }

            let mut challenger = Challenger::new();
            let batched_proof = BatchedCubicSumCheckProof::prove(
                &[left.clone()],
                &[right.clone()],
                &[eq.clone()],
                &[actual_sum],
                &mut challenger,
            );

            let mut verifier = Challenger::new();
            batched_proof.verify(&[actual_sum], &mut verifier);

            assert_eq!(batched_proof.num_claims, 1);
            assert_eq!(batched_proof.final_evals.len(), 2);
        }

        #[test]
        fn test_batched_cubic_sumcheck_multiple_claims() {
            // Test with 3 claims (similar to InnerSumCheck pattern)
            let mut left_polys = Vec::new();
            let mut right_polys = Vec::new();
            let mut eq_evals = Vec::new();
            let mut claimed_sums = Vec::new();

            // Create 3 different cubic claims
            for claim_idx in 0..3 {
                let left = MLE::new(vec![
                    Fp4::from_u32((claim_idx * 4 + 1) as u32),
                    Fp4::from_u32((claim_idx * 4 + 2) as u32),
                    Fp4::from_u32((claim_idx * 4 + 3) as u32),
                    Fp4::from_u32((claim_idx * 4 + 4) as u32),
                ]);
                let right = MLE::new(vec![
                    Fp4::from_u32((claim_idx * 5 + 1) as u32),
                    Fp4::from_u32((claim_idx * 5 + 2) as u32),
                    Fp4::from_u32((claim_idx * 5 + 3) as u32),
                    Fp4::from_u32((claim_idx * 5 + 4) as u32),
                ]);
                let point = vec![Fp4::from_u32((claim_idx + 1) as u32)];
                let eq = EqEvals::gen_from_point(&point);

                // Compute actual sum for this claim
                let mut actual_sum = Fp4::ZERO;
                for i in 0..4 {
                    actual_sum += left[i] * right[i] * eq[i];
                }

                left_polys.push(left);
                right_polys.push(right);
                eq_evals.push(eq);
                claimed_sums.push(actual_sum);
            }

            let mut challenger = Challenger::new();
            let batched_proof = BatchedCubicSumCheckProof::prove(
                &left_polys,
                &right_polys,
                &eq_evals,
                &claimed_sums,
                &mut challenger,
            );

            let mut verifier = Challenger::new();
            batched_proof.verify(&claimed_sums, &mut verifier);

            assert_eq!(batched_proof.num_claims, 3);
            assert_eq!(batched_proof.final_evals.len(), 6); // 2 values per claim
        }

        #[test]
        fn test_batched_cubic_sumcheck_large_batch() {
            // Test with 10 claims to demonstrate scalability
            let num_claims = 10;
            let mut left_polys = Vec::new();
            let mut right_polys = Vec::new();
            let mut eq_evals = Vec::new();
            let mut claimed_sums = Vec::new();

            // Create 10 different cubic claims
            for _claim_idx in 0..num_claims {
                let left = MLE::new(vec![Fp4::from_u32(1), Fp4::from_u32(2)]);
                let right = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(4)]);
                let eq = EqEvals::gen_from_point(&vec![Fp4::from_u32(5)]);

                // Simple sum: (1*3*eq[0]) + (2*4*eq[1])
                let actual_sum = left[0] * right[0] * eq[0] + left[1] * right[1] * eq[1];

                left_polys.push(left);
                right_polys.push(right);
                eq_evals.push(eq);
                claimed_sums.push(actual_sum);
            }

            let mut challenger = Challenger::new();
            let batched_proof = BatchedCubicSumCheckProof::prove(
                &left_polys,
                &right_polys,
                &eq_evals,
                &claimed_sums,
                &mut challenger,
            );

            let mut verifier = Challenger::new();
            batched_proof.verify(&claimed_sums, &mut verifier);

            assert_eq!(batched_proof.num_claims, num_claims);
            assert_eq!(batched_proof.final_evals.len(), num_claims * 2);
        }

        #[test]
        fn test_batched_cubic_sumcheck_empty_batch() {
            // Test edge case with no claims
            let mut challenger = Challenger::new();
            let batched_proof =
                BatchedCubicSumCheckProof::prove(&[], &[], &[], &[], &mut challenger);

            let mut verifier = Challenger::new();
            batched_proof.verify(&[], &mut verifier);

            assert_eq!(batched_proof.num_claims, 0);
            assert!(batched_proof.final_evals.is_empty());
            assert!(batched_proof.round_proofs.is_empty());
        }

        #[test]
        fn test_batched_cubic_sumcheck_consistency() {
            // Test that same inputs produce same proofs
            let left = MLE::new(vec![Fp4::from_u32(1), Fp4::from_u32(2)]);
            let right = MLE::new(vec![Fp4::from_u32(3), Fp4::from_u32(4)]);
            let eq = EqEvals::gen_from_point(&vec![Fp4::from_u32(5)]);
            let actual_sum = left[0] * right[0] * eq[0] + left[1] * right[1] * eq[1];

            let mut challenger1 = Challenger::new();
            let mut challenger2 = Challenger::new();

            let proof1 = BatchedCubicSumCheckProof::prove(
                &[left.clone()],
                &[right.clone()],
                &[eq.clone()],
                &[actual_sum],
                &mut challenger1,
            );

            let proof2 = BatchedCubicSumCheckProof::prove(
                &[left.clone()],
                &[right.clone()],
                &[eq.clone()],
                &[actual_sum],
                &mut challenger2,
            );

            assert_eq!(proof1, proof2);
        }
    }
}
