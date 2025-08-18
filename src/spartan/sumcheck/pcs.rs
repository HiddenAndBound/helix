use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4, challenger::Challenger, eq::EqEvals, polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};

/// Sum-check proof for polynomial commitment scheme evaluation claims.
///
/// Proves that a multilinear polynomial p evaluates to a claimed value at a point z
/// by converting the evaluation claim into a sum-check protocol:
/// p(z) = ∑_{x∈{0,1}^d} p(x) * eq_z(x) = claimed_evaluation
///
/// This follows the same architectural pattern as OuterSumCheck and InnerSumCheck
/// but is specifically designed for polynomial commitment scheme integration.
#[derive(Debug, Clone, PartialEq)]
pub struct PCSSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    pub round_proofs: Vec<UnivariatePoly>,
    /// Final evaluation p(r) at the random point r.
    pub final_eval: Fp4,
}

impl PCSSumCheckProof {
    /// Creates a new PCS sum-check proof from round polynomials and final evaluation.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_eval: Fp4) -> Self {
        Self {
            round_proofs,
            final_eval,
        }
    }

    /// Generates a sum-check proof for polynomial evaluation: p(z) = claimed_evaluation.
    ///
    /// Converts the evaluation claim to sum-check via the identity:
    /// p(z) = ∑_{x∈{0,1}^d} p(x) * eq_z(x)
    ///
    /// # Arguments
    /// * `poly` - The multilinear polynomial to prove evaluation for
    /// * `point` - The evaluation point z
    /// * `evaluation` - The claimed evaluation p(z)
    /// * `challenger` - Challenger for Fiat-Shamir randomness
    pub fn prove(
        poly: &MLE<Fp>,
        point: &[Fp4],
        evaluation: Fp4,
        challenger: &mut Challenger,
    ) -> Self {
        let rounds = poly.n_vars();
        assert_eq!(
            point.len(),
            rounds,
            "Evaluation point must have same dimension as polynomial"
        );

        // Initialize equality polynomial eq(x, z) for the evaluation point
        let mut eq = EqEvals::gen_from_point(point);

        let mut current_claim = evaluation;
        let mut round_proofs = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_pcs_first_round(poly, &eq, point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomial and equality evaluations by fixing first variable to challenge
        let mut poly_fold = poly.fold_in_place(round_challenge);
        eq.fold_in_place();

        // Process remaining rounds (1 to d-1)
        for round in 1..rounds {
            let round_proof =
                compute_pcs_round(&poly_fold, &eq, point, current_claim, round, rounds);

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            poly_fold = poly_fold.fold_in_place(round_challenge);
            eq.fold_in_place();
        }

        // Final evaluation is p(r) where r is the vector of all round challenges
        let final_eval = poly_fold[0];

        PCSSumCheckProof::new(round_proofs, final_eval)
    }

    /// Verifies the PCS sum-check proof. Panics if verification fails.
    ///
    /// # Arguments
    /// * `point` - The evaluation point z
    /// * `evaluation` - The claimed evaluation p(z)
    /// * `challenger` - Challenger for Fiat-Shamir randomness
    pub fn verify(&self, point: &[Fp4], evaluation: Fp4, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();
        assert_eq!(
            point.len(),
            rounds,
            "Evaluation point must match number of rounds"
        );

        let mut current_claim = evaluation;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = g_i(0) + g_i(1)
            // This is the standard sum-check verification for polynomials over {0,1}
            assert_eq!(
                current_claim,
                round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: verify that the final evaluation matches the claim
        // This requires that p(r) * eq_z(r) = final_claim where r is the challenge vector
        assert_eq!(current_claim, self.final_eval);
    }
}

/// Computes the univariate polynomial for the first PCS sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
/// Evaluates at 0 and 1 only (degree-1 polynomial).
pub fn compute_pcs_first_round(
    poly: &MLE<Fp>,
    eq: &EqEvals,
    point: &[Fp4],
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Evaluate at X = 0 and X = 1 only
    let mut round_coeffs = vec![Fp4::ZERO; 2];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0
        round_coeffs[0] += Fp4::from(eq[i] * poly[i << 1]);
    }

    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + point[0])) / point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for subsequent PCS sum-check rounds.
/// Works in extension field (Fp4) for all computations.
/// Evaluates at 0 and 1 only (degree-1 polynomial).
pub fn compute_pcs_round(
    poly: &MLE<Fp4>,
    eq: &EqEvals,
    point: &[Fp4],
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Evaluate at X = 0 and X = 1 only
    let mut round_coeffs = vec![Fp4::ZERO; 2];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += Fp4::from(eq[i]) * poly[i << 1];
    }

    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + point[0])) / point[0];
    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}
