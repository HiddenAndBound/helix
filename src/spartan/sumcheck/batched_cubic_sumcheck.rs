//! Batched cubic sum-check proof implementation.

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

    // To be parallelised
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        challenger::Challenger,
        pcs::PolynomialCommitmentScheme,
        spartan::{
            spark::{
                gkr::Gkr,
                sparse::{SpartanMetadata, TimeStamps},
            },
            R1CSInstance,
        },
    };

    #[test]
    fn test_batched_cubic_sumcheck_prove_verify() {
        // Create dummy metadata and oracle pairs for three claims
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

        // Dummy claimed sums and gamma
        let claimed_sums = [
            Fp4::from_u32(1u32),
            Fp4::from_u32(2u32),
            Fp4::from_u32(3u32),
        ];
        let gamma = Fp4::from_u32(7u32);

        // Create separate challengers for prover and verifier
        let mut prover_challenger = Challenger::new();
        let mut verifier_challenger = Challenger::new();

        // Generate the proof
        let proof = BatchedCubicSumCheckProof::prove(
            &metadatas,
            &oracle_pairs,
            &claimed_sums,
            &mut prover_challenger,
        );

        // Verify the proof
        assert!(proof.verify(&claimed_sums, &mut verifier_challenger));

        // Test proof structure
        assert_eq!(proof.num_claims, 3);
        assert!(!proof.round_proofs.is_empty());
        assert_eq!(proof.final_evals.len(), 3);
    }

    #[test]
    fn test_batched_cubic_sumcheck_different_gammas() {
        // Test that different gamma values produce different but valid proofs
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

        let claimed_sums = [
            Fp4::from_u32(1u32),
            Fp4::from_u32(2u32),
            Fp4::from_u32(3u32),
        ];

        let gammas = [Fp4::from_u32(1), Fp4::from_u32(17), Fp4::from_u32(42)];

        for test_gamma in gammas {
            let mut prover_challenger = Challenger::new();
            let mut verifier_challenger = Challenger::new();

            let proof = BatchedCubicSumCheckProof::prove(
                &metadatas,
                &oracle_pairs,
                &claimed_sums,
                &mut prover_challenger,
            );

            // Verify with the correct gamma value for this specific proof generation
            assert!(proof.verify(&claimed_sums, test_gamma, &mut verifier_challenger));
        }
    }

    #[test]
    fn test_batched_cubic_sumcheck_empty() {
        // Test case for empty claims
        let metadatas: [SpartanMetadata; 0] = [];
        let oracle_pairs: [(MLE<Fp4>, MLE<Fp4>); 0] = [];
        let claimed_sums: [Fp4; 0] = [];
        let gamma = Fp4::ZERO;

        let mut prover_challenger = Challenger::new();
        let mut verifier_challenger = Challenger::new();

        let proof = BatchedCubicSumCheckProof::prove(
            &metadatas,
            &oracle_pairs,
            &claimed_sums,
            &mut prover_challenger,
        );

        assert!(proof.verify(&claimed_sums, gamma, &mut verifier_challenger));
    }
}