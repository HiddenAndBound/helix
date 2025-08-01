use crate::utils::challenger::Challenger;
use crate::utils::polynomial::MLE;
use crate::utils::{Fp, Fp4};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use std::marker::PhantomData;

/// Sumcheck proof consisting of univariate polynomials for each round
#[derive(Debug, Clone)]
pub struct SumcheckProof {
    /// Univariate polynomials sent by the prover in each round
    pub round_polynomials: Vec<Vec<Fp4>>,
    /// Final claimed sum
    pub claimed_sum: Fp4,
}

/// Prover for the sumcheck protocol
pub struct SumcheckProver {
    _phantom: PhantomData<BabyBear>,
}

impl SumcheckProver {
    /// Create a new sumcheck prover
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Execute the sumcheck protocol to prove the sum of a multilinear polynomial
    /// over the boolean hypercube {0,1}^n
    pub fn prove(
        &self,
        polynomial: &MLE<BabyBear>,
        challenger: &mut Challenger,
    ) -> (SumcheckProof, Vec<Fp4>) {
        let mut current_poly = MLE::<Fp4>::new(
            polynomial
                .coeffs()
                .iter()
                .map(|c| Fp4::from(Fp::from_u32(c.as_canonical_u32())))
                .collect(),
        );
        let mut round_polynomials = Vec::new();
        let mut challenges = Vec::new();

        // Calculate the actual sum over the boolean hypercube
        let claimed_sum = self.calculate_boolean_hypercube_sum(polynomial);

        let num_variables = current_poly.n_vars();

        for _round in 0..num_variables {
            // Compute the univariate polynomial for this round
            let round_poly = self.compute_round_polynomial_fp4(&current_poly);
            round_polynomials.push(round_poly.clone());

            // Observe the round polynomial coefficients
            challenger.observe_fp4_elems(&round_poly);

            // Get challenge from verifier
            let challenge = challenger.get_challenge();
            challenges.push(challenge);

            // Fold the polynomial using the challenge
            current_poly = current_poly.fold_in_place(challenge);
        }

        (
            SumcheckProof {
                round_polynomials,
                claimed_sum,
            },
            challenges,
        )
    }

    /// Calculate the sum of the polynomial over the boolean hypercube
    fn calculate_boolean_hypercube_sum(&self, poly: &MLE<BabyBear>) -> Fp4 {
        let mut sum = Fp4::ZERO;

        // Iterate over all points in the boolean hypercube
        for coeff in poly.coeffs() {
            let fp = Fp::from_u32(coeff.as_canonical_u32());
            sum += Fp4::from(fp);
        }

        sum
    }

    /// Compute the univariate polynomial for a given round
    fn compute_round_polynomial(&self, poly: &MLE<BabyBear>) -> Vec<Fp4> {
        // The univariate polynomial is g_i(X) = sum_{b_{i+1},...,b_n} f(r_1,...,r_{i-1},X,b_{i+1},...,b_n)
        // For simplicity, we'll compute it by evaluating at X=0 and X=1

        let mut g_0 = Fp4::ZERO; // g_i(0)
        let mut g_1 = Fp4::ZERO; // g_i(1)

        let half_len = poly.len() / 2;

        // Split coefficients into even (x_i=0) and odd (x_i=1) indices
        for i in 0..half_len {
            let low_idx = i * 2; // Even indices: x_i=0
            let high_idx = i * 2 + 1; // Odd indices: x_i=1

            let low_coeff = &poly.coeffs()[low_idx];
            let high_coeff = &poly.coeffs()[high_idx];

            g_0 += Fp4::from(Fp::from_u32(low_coeff.as_canonical_u32()));
            g_1 += Fp4::from(Fp::from_u32(high_coeff.as_canonical_u32()));
        }

        // The univariate polynomial is g_i(X) = (g_1 - g_0) * X + g_0
        vec![g_0, g_1 - g_0]
    }

    /// Compute the univariate polynomial for a given round (Fp4 version)
    fn compute_round_polynomial_fp4(&self, poly: &MLE<Fp4>) -> Vec<Fp4> {
        // The univariate polynomial is g_i(X) = sum_{b_{i+1},...,b_n} f(r_1,...,r_{i-1},X,b_{i+1},...,b_n)
        // For simplicity, we'll compute it by evaluating at X=0 and X=1

        let mut g_0 = Fp4::ZERO; // g_i(0)
        let mut g_1 = Fp4::ZERO; // g_i(1)

        let half_len = poly.len() / 2;

        // Split coefficients into even (x_i=0) and odd (x_i=1) indices
        for i in 0..half_len {
            let low_idx = i * 2; // Even indices: x_i=0
            let high_idx = i * 2 + 1; // Odd indices: x_i=1

            let low_coeff = &poly.coeffs()[low_idx];
            let high_coeff = &poly.coeffs()[high_idx];

            g_0 += *low_coeff;
            g_1 += *high_coeff;
        }

        // The univariate polynomial is g_i(X) = (g_1 - g_0) * X + g_0
        vec![g_0, g_1 - g_0]
    }
}

/// Verifier for the sumcheck protocol
pub struct SumcheckVerifier {
    /// Number of variables in the polynomial
    pub num_variables: usize,
}

impl SumcheckVerifier {
    /// Create a new sumcheck verifier
    pub fn new(num_variables: usize) -> Self {
        Self { num_variables }
    }

    /// Verify a sumcheck proof
    pub fn verify(
        &self,
        proof: &SumcheckProof,
        claimed_sum: Fp4,
        challenger: &mut Challenger,
        polynomial: &MLE<BabyBear>,
    ) -> bool {
        let mut current_sum = claimed_sum;
        let mut challenges = Vec::new();

        for round in 0..self.num_variables {
            if round >= proof.round_polynomials.len() {
                return false;
            }

            let round_poly = &proof.round_polynomials[round];
            if round_poly.len() != 2 {
                return false; // Should be degree-1 univariate polynomial
            }

            // Check that g_i(0) + g_i(1) = current_sum
            let g_0 = round_poly[0];
            let g_1 = g_0 + round_poly[1];

            if g_0 + g_1 != current_sum {
                return false;
            }

            // Get challenge from verifier
            challenger.observe_fp4_elems(round_poly);
            let challenge = challenger.get_challenge();
            challenges.push(challenge);

            // Update current_sum for next round: g_i(challenge)
            current_sum = g_0 + challenge * round_poly[1];
        }

        // Finally, check that the polynomial evaluated at the challenges equals the final claim
        let expected_eval = polynomial.evaluate(&challenges);
        expected_eval == current_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::challenger::Challenger;
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_simple_sumcheck() {
        let prover = SumcheckProver::new();
        let verifier = SumcheckVerifier::new(2);

        // Create a simple polynomial: f(x,y) = x + y
        let coeffs = vec![
            BabyBear::from_u32(0), // f(0,0) = 0
            BabyBear::from_u32(1), // f(0,1) = 1
            BabyBear::from_u32(1), // f(1,0) = 1
            BabyBear::from_u32(2), // f(1,1) = 2
        ];
        let poly = MLE::new(coeffs);

        let mut challenger = Challenger::new();
        let (proof, challenges) = prover.prove(&poly, &mut challenger);

        // The sum should be 0 + 1 + 1 + 2 = 4
        assert_eq!(proof.claimed_sum, Fp4::from(Fp::from_u32(4)));

        let mut verifier_challenger = Challenger::new();
        let is_valid = verifier.verify(&proof, proof.claimed_sum, &mut verifier_challenger, &poly);
        assert!(is_valid);
    }

    #[test]
    fn test_constant_polynomial() {
        let prover = SumcheckProver::new();
        let verifier = SumcheckVerifier::new(2);

        // Create a constant polynomial: f(x,y) = 3
        let coeffs = vec![
            BabyBear::from_u32(3), // f(0,0) = 3
            BabyBear::from_u32(3), // f(0,1) = 3
            BabyBear::from_u32(3), // f(1,0) = 3
            BabyBear::from_u32(3), // f(1,1) = 3
        ];
        let poly = MLE::new(coeffs);

        let mut challenger = Challenger::new();
        let (proof, challenges) = prover.prove(&poly, &mut challenger);

        // The sum should be 3 * 4 = 12
        assert_eq!(proof.claimed_sum, Fp4::from(Fp::from_u32(12)));

        let mut verifier_challenger = Challenger::new();
        let is_valid = verifier.verify(&proof, proof.claimed_sum, &mut verifier_challenger, &poly);
        assert!(is_valid);
    }

    #[test]
    fn test_linear_polynomial() {
        let prover = SumcheckProver::new();
        let verifier = SumcheckVerifier::new(3);

        // Create a polynomial: f(x,y,z) = x + 2y + 3z
        let coeffs = vec![
            BabyBear::from_u32(0), // f(0,0,0) = 0
            BabyBear::from_u32(3), // f(0,0,1) = 3
            BabyBear::from_u32(2), // f(0,1,0) = 2
            BabyBear::from_u32(5), // f(0,1,1) = 5
            BabyBear::from_u32(1), // f(1,0,0) = 1
            BabyBear::from_u32(4), // f(1,0,1) = 4
            BabyBear::from_u32(3), // f(1,1,0) = 3
            BabyBear::from_u32(6), // f(1,1,1) = 6
        ];
        let poly = MLE::new(coeffs);

        let mut challenger = Challenger::new();
        let (proof, challenges) = prover.prove(&poly, &mut challenger);

        // The sum should be 0+3+2+5+1+4+3+6 = 24
        assert_eq!(proof.claimed_sum, Fp4::from(Fp::from_u32(24)));

        let mut verifier_challenger = Challenger::new();
        let is_valid = verifier.verify(&proof, proof.claimed_sum, &mut verifier_challenger, &poly);
        assert!(is_valid);
    }
}
