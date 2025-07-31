use crate::utils::{Fp, Fp4, polynomial::MLE, challenger::Challenger};
use crate::sumcheck::SumcheckProof;
use anyhow::Result;
use p3_field::PrimeCharacteristicRing;

/// Sumcheck prover implementing the interactive sumcheck protocol
pub struct SumcheckProver {
    /// Number of variables in the polynomial
    num_variables: usize,
    /// Random challenges from the verifier (generated via Fiat-Shamir)
    challenges: Vec<Fp4>,
}

impl SumcheckProver {
    /// Create a new sumcheck prover
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            challenges: Vec::new(),
        }
    }
    
    /// Generate a sumcheck proof for the claimed sum
    pub fn prove(&mut self, polynomial: MLE<Fp>, challenger: &mut Challenger, claimed_sum: Fp4) -> Result<SumcheckProof> {
        let mut proof = SumcheckProof::new();
        
        // Convert initial polynomial to Fp4 for consistency
        let mut current_polynomial = self.convert_to_fp4(polynomial);
        
        // Process each variable/round
        for round in 0..self.num_variables {
            // Compute the round polynomial g_i(X) 
            let round_poly = self.compute_round_polynomial(&current_polynomial)?;
            
            // Verify first round consistency: g_0(0) + g_0(1) should equal claimed sum
            if round == 0 {
                let sum_check = round_poly[0] + round_poly[1];
                if sum_check != claimed_sum {
                    return Err(anyhow::anyhow!("First round polynomial inconsistent with claimed sum"));
                }
            }
            
            // Send round polynomial to challenger (Fiat-Shamir)
            challenger.observe_fp4_elems(&round_poly);
            
            // Get random challenge from verifier
            let challenge = challenger.get_challenge();
            self.challenges.push(challenge);
            
            // Add to proof
            proof.round_polynomials.push(round_poly.clone());
            
            // Bind the variable to the challenge value for next round
            current_polynomial = current_polynomial.fold_in_place(challenge);
        }
        
        // Final evaluation is the constant remaining after all variables are bound
        if current_polynomial.n_vars() != 0 {
            return Err(anyhow::anyhow!("Final polynomial should have 0 variables"));
        }
        let final_eval = current_polynomial.coeffs()[0];
        proof.final_evaluations.push(final_eval);
        
        Ok(proof)
    }
    
    /// Convert MLE<Fp> to MLE<Fp4> 
    fn convert_to_fp4(&self, mle: MLE<Fp>) -> MLE<Fp4> {
        let fp4_coeffs: Vec<Fp4> = mle.coeffs().iter()
            .map(|&coeff| Fp4::from(coeff))
            .collect();
        MLE::new(fp4_coeffs)
    }
    
    /// Compute the round polynomial for the current polynomial
    /// For a polynomial with n variables, computes g(X) = sum over all boolean assignments
    /// to variables 1..n-1 of f(X, x_1, ..., x_{n-1})
    fn compute_round_polynomial(&self, polynomial: &MLE<Fp4>) -> Result<Vec<Fp4>> {
        let n_vars = polynomial.n_vars();
        if n_vars == 0 {
            return Err(anyhow::anyhow!("Cannot compute round polynomial for 0-variable polynomial"));
        }
        
        // We compute g(0), g(1), g(2) where g(X) is the round polynomial
        let mut round_poly = vec![Fp4::ZERO; 3];
        
        // Sum over all (2^{n-1}) boolean assignments to the remaining variables
        let num_assignments = 1 << (n_vars - 1);
        
        for assignment in 0..num_assignments {
            // For each X ∈ {0, 1, 2}, evaluate at (X, assignment bits)
            for x_val in 0..3 {
                let mut eval_point = Vec::with_capacity(n_vars);
                eval_point.push(Fp4::from(Fp::from_u32(x_val)));
                
                // Add the boolean assignment for remaining variables  
                for bit_pos in 0..(n_vars - 1) {
                    let bit_val = if (assignment >> bit_pos) & 1 == 1 {
                        Fp4::ONE
                    } else {
                        Fp4::ZERO
                    };
                    eval_point.push(bit_val);
                }
                
                // Evaluate polynomial at this point and add to round polynomial coefficient
                let eval = polynomial.evaluate(&eval_point);
                round_poly[x_val as usize] += eval;
            }
        }
        
        Ok(round_poly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::polynomial::MLE;
    use p3_field::PrimeCharacteristicRing;
    
    #[test]
    fn test_simple_sumcheck() {
        // Create a simple 2-variable polynomial: f(x,y) = x + y
        // Coefficients for f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=2
        let coeffs = vec![
            Fp::ZERO,                    // f(0,0) = 0 + 0 = 0
            Fp::ONE,                     // f(1,0) = 1 + 0 = 1  
            Fp::ONE,                     // f(0,1) = 0 + 1 = 1
            Fp::from_u32(2),   // f(1,1) = 1 + 1 = 2
        ];
        
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(2);
        let mut challenger = Challenger::new();
        
        // Sum over all boolean assignments: f(0,0) + f(1,0) + f(0,1) + f(1,1) = 0 + 1 + 1 + 2 = 4
        let claimed_sum = Fp4::from(Fp::from_u32(4));
        
        let proof = prover.prove(polynomial, &mut challenger, claimed_sum).unwrap();
        
        // Should have 2 rounds for 2 variables
        assert_eq!(proof.num_rounds(), 2);
        
        // Each round polynomial should have 3 coefficients (degree 2)
        for round_poly in &proof.round_polynomials {
            assert_eq!(round_poly.len(), 3);
        }
        
        // Should have 1 final evaluation
        assert_eq!(proof.final_evaluations.len(), 1);
    }
    
    #[test]
    fn test_constant_polynomial() {
        // Constant polynomial f(x) = 5
        let coeffs = vec![
            Fp::from_u32(5), // f(0) = 5
            Fp::from_u32(5), // f(1) = 5
        ];
        
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(1);
        let mut challenger = Challenger::new();
        
        // Sum: f(0) + f(1) = 5 + 5 = 10
        let claimed_sum = Fp4::from(Fp::from_u32(10));
        
        let proof = prover.prove(polynomial, &mut challenger, claimed_sum).unwrap();
        
        // Should have 1 round for 1 variable  
        assert_eq!(proof.num_rounds(), 1);
        
        // First round polynomial g(X) where g(0) + g(1) = 10
        let first_round = &proof.round_polynomials[0];
        assert_eq!(first_round[0] + first_round[1], claimed_sum);
    }
    
    #[test]
    fn test_zero_polynomial() {
        // Zero polynomial f(x,y) = 0
        let coeffs = vec![Fp::ZERO; 4];
        let polynomial = MLE::new(coeffs);
        let mut prover = SumcheckProver::new(2);
        let mut challenger = Challenger::new();
        
        let claimed_sum = Fp4::ZERO;
        let proof = prover.prove(polynomial, &mut challenger, claimed_sum).unwrap();
        
        // All round polynomials should be zero
        for round_poly in &proof.round_polynomials {
            for &coeff in round_poly {
                assert_eq!(coeff, Fp4::ZERO);
            }
        }
        
        assert_eq!(proof.final_evaluations[0], Fp4::ZERO);
    }
}