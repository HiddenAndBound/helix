use crate::utils::{Fp, Fp4, challenger::Challenger};
use crate::sumcheck::SumcheckProof;
use anyhow::Result;
use p3_field::{PrimeCharacteristicRing, Field};

/// Sumcheck verifier implementing the verification side of the interactive sumcheck protocol
pub struct SumcheckVerifier {
    /// Number of variables in the original polynomial
    num_variables: usize,
    /// Random challenges generated during verification
    challenges: Vec<Fp4>,
}

impl SumcheckVerifier {
    /// Create a new sumcheck verifier
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            challenges: Vec::new(),
        }
    }
    
    /// Verify a sumcheck proof
    /// Returns the final evaluation that should be checked externally
    pub fn verify(&mut self, proof: &SumcheckProof, challenger: &mut Challenger, claimed_sum: Fp4) -> Result<Fp4> {
        if proof.num_rounds() != self.num_variables {
            return Err(anyhow::anyhow!(
                "Proof has {} rounds but expected {}",
                proof.num_rounds(),
                self.num_variables
            ));
        }
        
        if proof.final_evaluations.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected exactly 1 final evaluation, got {}",
                proof.final_evaluations.len()
            ));
        }
        
        let mut current_sum = claimed_sum;
        
        // Verify each round
        for (round, round_poly) in proof.round_polynomials.iter().enumerate() {
            // Each round polynomial should have degree 2 (3 coefficients)
            if round_poly.len() != 3 {
                return Err(anyhow::anyhow!(
                    "Round {} polynomial has {} coefficients, expected 3",
                    round, round_poly.len()
                ));
            }
            
            // Verify consistency: g(0) + g(1) should equal the current sum
            let consistency_check = round_poly[0] + round_poly[1];
            if consistency_check != current_sum {
                return Err(anyhow::anyhow!(
                    "Round {} consistency check failed: g(0) + g(1) = {}, expected {}",
                    round, consistency_check, current_sum
                ));
            }
            
            // Generate the same challenge as the prover using Fiat-Shamir
            challenger.observe_fp4_elems(round_poly);
            let challenge = challenger.get_challenge();
            self.challenges.push(challenge);
            
            // Update current sum to g(challenge) for next round
            current_sum = self.evaluate_univariate_at_challenge(round_poly, challenge);
        }
        
        // The final sum should match the final evaluation in the proof
        if current_sum != proof.final_evaluations[0] {
            return Err(anyhow::anyhow!(
                "Final evaluation mismatch: computed {}, proof claims {}",
                current_sum, proof.final_evaluations[0]
            ));
        }
        
        Ok(current_sum)
    }
    
    /// Evaluate a degree-2 univariate polynomial at a given challenge
    /// Polynomial is represented as [f(0), f(1), f(2)]
    /// Uses Lagrange interpolation: f(x) = f(0)L₀(x) + f(1)L₁(x) + f(2)L₂(x)
    /// where L₀(x) = (x-1)(x-2)/2, L₁(x) = -x(x-2), L₂(x) = x(x-1)/2
    fn evaluate_univariate_at_challenge(&self, coeffs: &[Fp4], challenge: Fp4) -> Fp4 {
        assert_eq!(coeffs.len(), 3, "Expected degree-2 polynomial with 3 coefficients");
        
        let f0 = coeffs[0];
        let f1 = coeffs[1];  
        let f2 = coeffs[2];
        
        // Lagrange basis polynomials evaluated at challenge
        let two = Fp4::from(Fp::from_u32(2));
        let two_inv = two.try_inverse().expect("2 should be invertible"); // 1/2
        
        // L₀(x) = (x-1)(x-2)/2
        let l0 = (challenge - Fp4::ONE) * (challenge - two) * two_inv;
        
        // L₁(x) = -x(x-2) = x(2-x)
        let l1 = challenge * (two - challenge);
        
        // L₂(x) = x(x-1)/2  
        let l2 = challenge * (challenge - Fp4::ONE) * two_inv;
        
        f0 * l0 + f1 * l1 + f2 * l2
    }
    
    /// Get the challenges used in verification (for external validation)
    pub fn get_challenges(&self) -> &[Fp4] {
        &self.challenges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::prover::SumcheckProver;
    use crate::utils::polynomial::MLE;
    use p3_field::PrimeCharacteristicRing;
    
    #[test]
    fn test_end_to_end_sumcheck() {
        // Create a simple 2-variable polynomial: f(x,y) = x + y
        let coeffs = vec![
            Fp::ZERO,                    // f(0,0) = 0
            Fp::ONE,                     // f(1,0) = 1  
            Fp::ONE,                     // f(0,1) = 1
            Fp::from_u32(2),   // f(1,1) = 2
        ];
        
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(2);
        let mut verifier = SumcheckVerifier::new(2);
        
        // Generate proof
        let mut challenger_prover = Challenger::new();
        let claimed_sum = Fp4::from(Fp::from_u32(4));
        let proof = prover.prove(polynomial.clone(), &mut challenger_prover, claimed_sum).unwrap();
        
        // Verify proof
        let mut challenger_verifier = Challenger::new();
        let final_eval = verifier.verify(&proof, &mut challenger_verifier, claimed_sum).unwrap();
        
        // Verify final evaluation by evaluating original polynomial at challenge point
        let challenges = verifier.get_challenges();
        assert_eq!(challenges.len(), 2);
        
        let expected_final = polynomial.evaluate(challenges);
        assert_eq!(final_eval, expected_final);
    }
    
    #[test]
    fn test_polynomial_evaluation() {
        let verifier = SumcheckVerifier::new(3);
        
        // Test evaluation of degree-2 polynomial f(x) = x² - 2x + 1 = (x-1)²
        // f(0) = 1, f(1) = 0, f(2) = 1
        let coeffs = vec![Fp4::ONE, Fp4::ZERO, Fp4::ONE];
        
        // Test at x = 3: f(3) = 9 - 6 + 1 = 4
        let challenge = Fp4::from(Fp::from_u32(3));
        let result = verifier.evaluate_univariate_at_challenge(&coeffs, challenge);
        let expected = Fp4::from(Fp::from_u32(4));
        assert_eq!(result, expected);
        
        // Test at x = 1: f(1) = 0
        let challenge = Fp4::ONE;
        let result = verifier.evaluate_univariate_at_challenge(&coeffs, challenge);
        assert_eq!(result, Fp4::ZERO);
    }
    
    #[test]
    fn test_constant_polynomial_verification() {
        let coeffs = vec![
            Fp::from_u32(5), // f(0) = 5
            Fp::from_u32(5), // f(1) = 5
        ];
        
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(1);
        let mut verifier = SumcheckVerifier::new(1);
        
        let claimed_sum = Fp4::from(Fp::from_u32(10));
        
        let mut challenger_prover = Challenger::new();
        let proof = prover.prove(polynomial, &mut challenger_prover, claimed_sum).unwrap();
        
        let mut challenger_verifier = Challenger::new();
        let final_eval = verifier.verify(&proof, &mut challenger_verifier, claimed_sum).unwrap();
        
        // Final evaluation should be f(challenge) = 5 for any challenge
        assert_eq!(final_eval, Fp4::from(Fp::from_u32(5)));
    }
    
    #[test]
    fn test_invalid_sum_rejection() {
        let coeffs = vec![Fp::ONE; 4]; // f(x,y) = 1 everywhere
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(2);
        let mut verifier = SumcheckVerifier::new(2);
        
        // True sum is 4, but claim 5
        let true_sum = Fp4::from(Fp::from_u32(4));
        let false_claim = Fp4::from(Fp::from_u32(5));
        
        let mut challenger_prover = Challenger::new();
        // This should fail at the prover level
        let result = prover.prove(polynomial, &mut challenger_prover, false_claim);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_zero_polynomial() {
        let coeffs = vec![Fp::ZERO; 4]; // f(x,y) = 0 everywhere
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(2);
        let mut verifier = SumcheckVerifier::new(2);
        
        let claimed_sum = Fp4::ZERO;
        
        let mut challenger_prover = Challenger::new();
        let proof = prover.prove(polynomial, &mut challenger_prover, claimed_sum).unwrap();
        
        let mut challenger_verifier = Challenger::new();
        let final_eval = verifier.verify(&proof, &mut challenger_verifier, claimed_sum).unwrap();
        
        assert_eq!(final_eval, Fp4::ZERO);
    }
}