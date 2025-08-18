use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};

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
    /// Final evaluations for all claims: [(left_0(r), right_0(r)), ..., (left_N-1(r), right_N-1(r))]
    pub final_evals: Vec<(Fp4, Fp4)>,
    /// Number of claims batched in this proof
    pub num_claims: usize,
}

impl BatchedCubicSumCheckProof {
    /// Creates a new batched cubic sum-check proof.
    pub fn new(
        round_proofs: Vec<UnivariatePoly>,
        final_evals: Vec<(Fp4, Fp4)>,
        num_claims: usize,
    ) -> Self {
        assert_eq!(
            final_evals.len(),
            num_claims,
            "Final evaluations must contain one tuple per claim (left and right)"
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
            claimed_sums.len(),
            num_claims,
            "Number of claimed sums must match"
        );

        if num_claims == 0 {
            return Self::new(vec![], vec![], 0);
        }

        let rounds = left_polys[0].n_vars();

        // Error case: polynomials must have at least 1 variable for sum-check
        assert!(
            rounds > 0,
            "BatchedCubicSumCheckProof requires polynomials with at least 1 variable"
        );

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
        }

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);
        // Compute batched claim using gamma powers
        let gamma = challenger.get_challenge();
        let mut batched_claim = Fp4::ZERO;

        for (i, &claimed_sum) in claimed_sums.iter().enumerate() {
            let gamma_power = gamma.exp_u64(i as u64 + 1);
            batched_claim += gamma_power * claimed_sum;
        }

        let mut eq_evals = EqEvals::gen_from_point(&eq_point);
        let mut current_claim = batched_claim;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_batched_cubic_round(
            left_polys,
            right_polys,
            &eq_evals,
            &eq_point,
            current_claim,
            gamma,
            num_claims,
            0,
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

        eq_evals.fold_in_place();
        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_batched_cubic_round(
                &left_folded,
                &right_folded,
                &eq_evals,
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

            eq_evals.fold_in_place();
            // Fold polynomials for next round
            for claim_idx in 0..num_claims {
                left_folded[claim_idx] = left_folded[claim_idx].fold_in_place(round_challenge);
                right_folded[claim_idx] = right_folded[claim_idx].fold_in_place(round_challenge);
            }
        }

        // Extract final evaluations for all claims
        let mut final_evals = Vec::with_capacity(num_claims);
        for (left, right) in left_folded.iter().zip(right_folded.iter()) {
            final_evals.push((left[0], right[0]));
        }

        BatchedCubicSumCheckProof::new(round_proofs, final_evals, num_claims)
    }

    /// Verifies the batched cubic sum-check proof.
    ///
    /// # Arguments
    /// * `claimed_sums` - Vector of claimed sum values for each claim
    /// * `challenger` - Challenger for Fiat-Shamir randomness
    pub fn verify(&self, claimed_sums: &[Fp4], challenger: &mut Challenger) -> bool {
        if claimed_sums.len() != self.num_claims {
            return false;
        }

        if self.num_claims == 0 {
            return true;
        }

        let rounds = self.round_proofs.len();
        let eq_point = challenger.get_challenges(rounds);
        let gamma = challenger.get_challenge();

        // Recompute batched claim
        let mut batched_claim = Fp4::ZERO;
        for (i, &claimed_sum) in claimed_sums.iter().enumerate() {
            batched_claim = claimed_sum + gamma * batched_claim;
        }

        let mut current_claim = batched_claim;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)

            let expected_claim = (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                + eq_point[round] * round_poly.evaluate(Fp4::ONE);
            if current_claim != expected_claim {
                return false;
            }

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: batched evaluation of final values must match the final claim
        let mut expected_claim = Fp4::ZERO;
        for (i, &(left_eval, right_eval)) in self.final_evals.iter().enumerate() {
            expected_claim = (left_eval * right_eval) + gamma * expected_claim;
        }

        current_claim == expected_claim
    }
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

/// Computes the univariate polynomial for the first batched cubic sum-check round.
///
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_batched_cubic_round(
    left_polys: &[MLE<Fp4>],
    right_polys: &[MLE<Fp4>],
    eq_evals: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    gamma: Fp4,
    num_claims: usize,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![vec![Fp4::ZERO; 3]; num_claims];

    //To be parallelised
    for i in 0..1 << (rounds - round - 1) {
        // Compute contributions for g(0): set first variable to 0
        for claim_idx in 0..num_claims {
            let left_val = left_polys[claim_idx][i << 1];
            let right_val = right_polys[claim_idx][i << 1];
            let left_at_2 =
                left_polys[claim_idx][i << 1] + left_polys[claim_idx][i << 1 | 1].double();
            let right_at_2 =
                right_polys[claim_idx][i << 1] + right_polys[claim_idx][i << 1 | 1].double();

            round_coeffs[claim_idx][0] += left_val * right_val * eq_evals[i];
            round_coeffs[claim_idx][2] += left_at_2 * right_at_2 * eq_evals[i];
        }
    }

    let mut batched_coeffs = vec![Fp4::ZERO; 3];
    for i in 0..num_claims {
        batched_coeffs[0] = round_coeffs[i][0] + gamma * batched_coeffs[0];
        batched_coeffs[2] = round_coeffs[i][0] + gamma * batched_coeffs[0];
    }
    // g(1): derived from sum-check constraint
    batched_coeffs[1] =
        (current_claim - batched_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(batched_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}