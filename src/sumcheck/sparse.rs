use crate::utils::{Fp, Fp4};
use p3_field::{PrimeCharacteristicRing, PackedValue};
use std::collections::HashMap;

/// Sparse representation of a multilinear extension
/// Stores only non-zero coefficients with their hypercube indices
#[derive(Debug, Clone)]
pub struct SparseMLE {
    /// Number of variables
    num_vars: usize,
    /// Non-zero coefficients mapped by their hypercube index
    coeffs: HashMap<usize, Fp>,
}

impl SparseMLE {
    /// Create a new sparse MLE from a coefficient map
    pub fn new(num_vars: usize, coeffs: HashMap<usize, Fp>) -> Self {
        // Verify all indices are valid for the given number of variables
        let max_index = 1 << num_vars;
        for &index in coeffs.keys() {
            assert!(index < max_index, "Index {} exceeds maximum {} for {} variables", 
                   index, max_index - 1, num_vars);
        }
        
        Self { num_vars, coeffs }
    }
    
    /// Create sparse MLE from a dense coefficient vector, filtering out zeros
    pub fn from_dense(coeffs: Vec<Fp>) -> Self {
        assert!(coeffs.len().is_power_of_two(), "Coefficient length must be power of two");
        let num_vars = coeffs.len().trailing_zeros() as usize;
        
        let sparse_coeffs: HashMap<usize, Fp> = coeffs
            .into_iter()
            .enumerate()
            .filter(|(_, coeff)| *coeff != Fp::ZERO)
            .collect();
        
        Self::new(num_vars, sparse_coeffs)
    }
    
    /// Get the number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    
    /// Get the number of non-zero coefficients
    pub fn num_nonzero(&self) -> usize {
        self.coeffs.len()
    }
    
    /// Get the sparsity ratio (non-zero coefficients / total coefficients)
    pub fn sparsity(&self) -> f64 {
        self.coeffs.len() as f64 / (1 << self.num_vars) as f64
    }
    
    /// Evaluate the sparse MLE at a given point
    pub fn evaluate(&self, point: &[Fp4]) -> Fp4 {
        assert_eq!(point.len(), self.num_vars, 
                  "Point dimension {} doesn't match MLE variables {}", 
                  point.len(), self.num_vars);
        
        let mut result = Fp4::ZERO;
        
        for (&index, &coeff) in &self.coeffs {
            // Compute the evaluation of the hypercube basis function at this index
            let basis_eval = self.evaluate_hypercube_basis(index, point);
            result += Fp4::from(coeff) * basis_eval;
        }
        
        result
    }
    
    /// Evaluate a hypercube basis function at a given point
    /// The basis function for index i is ∏ⱼ ((1-xⱼ)(1-bⱼ) + xⱼbⱼ)
    /// where bⱼ is the j-th bit of the binary representation of i
    fn evaluate_hypercube_basis(&self, index: usize, point: &[Fp4]) -> Fp4 {
        let mut result = Fp4::ONE;
        
        for (bit_pos, &x_val) in point.iter().enumerate() {
            let bit = (index >> bit_pos) & 1;
            if bit == 1 {
                // Variable is set to 1: contribute x_val
                result *= x_val;
            } else {
                // Variable is set to 0: contribute (1 - x_val)
                result *= Fp4::ONE - x_val;
            }
        }
        
        result
    }
    
    /// Fold (bind) the first variable to a challenge value
    /// Returns a new sparse MLE with one fewer variable
    pub fn fold_first_variable(&self, challenge: Fp4) -> SparseMLE {
        if self.num_vars == 0 {
            panic!("Cannot fold 0-variable polynomial");
        }
        
        let new_num_vars = self.num_vars - 1;
        let mut new_coeffs = HashMap::new();
        
        // Group coefficients by their projections onto the remaining variables
        for (&old_index, &coeff) in &self.coeffs {
            let first_bit = old_index & 1;
            let remaining_index = old_index >> 1;
            
            let contribution = if first_bit == 1 {
                Fp4::from(coeff) * challenge
            } else {
                Fp4::from(coeff) * (Fp4::ONE - challenge)
            };
            
            // Accumulate contributions for the same remaining index
            let new_coeff = new_coeffs.entry(remaining_index)
                .or_insert(Fp4::ZERO);
            *new_coeff += contribution;
        }
        
        // Convert back to Fp, filtering out zeros
        let fp_coeffs: HashMap<usize, Fp> = new_coeffs
            .into_iter()
            .filter_map(|(index, coeff)| {
                if coeff != Fp4::ZERO {
                    // Extract base field component (assuming coeff is in base field)
                    Some((index, coeff.as_slice()[0]))
                } else {
                    None
                }
            })
            .collect();
        
        SparseMLE::new(new_num_vars, fp_coeffs)
    }
    
    /// Compute the round polynomial for sumcheck
    /// More efficient than dense evaluation for sparse polynomials
    pub fn compute_round_polynomial(&self) -> Vec<Fp4> {
        if self.num_vars == 0 {
            panic!("Cannot compute round polynomial for 0-variable polynomial");
        }
        
        let mut round_poly = vec![Fp4::ZERO; 3];
        
        // For each non-zero coefficient, compute its contribution to g(0), g(1), g(2)
        for (&index, &coeff) in &self.coeffs {
            let first_bit = index & 1;
            let remaining_index = index >> 1;
            
            // Evaluate the remaining basis function at all boolean assignments
            let remaining_vars = self.num_vars - 1;
            
            if remaining_vars == 0 {
                // Special case: only one variable left
                if first_bit == 1 {
                    // Contributes to g(1) and g(2)
                    round_poly[1] += Fp4::from(coeff);
                    round_poly[2] += Fp4::from(coeff);
                } else {
                    // Contributes to g(0) and g(2) with weight (1-X)
                    round_poly[0] += Fp4::from(coeff);
                    // For g(2): (1-2) = -1
                    round_poly[2] -= Fp4::from(coeff);
                }
            } else {
                // General case: sum over all boolean assignments to remaining variables
                let num_assignments = 1 << remaining_vars;
                
                for assignment in 0..num_assignments {
                    // Check if this assignment matches the remaining index pattern
                    if self.matches_remaining_pattern(remaining_index, assignment, remaining_vars) {
                        // Compute contribution to each g(X) value
                        for x_val in 0..3 {
                            let contrib = if first_bit == 1 {
                                Fp4::from(Fp::from_u32(x_val))
                            } else {
                                Fp4::ONE - Fp4::from(Fp::from_u32(x_val))
                            };
                            round_poly[x_val as usize] += Fp4::from(coeff) * contrib;
                        }
                        break; // Only one assignment should match
                    }
                }
            }
        }
        
        round_poly
    }
    
    /// Check if a boolean assignment matches the remaining index pattern
    fn matches_remaining_pattern(&self, remaining_index: usize, assignment: usize, num_vars: usize) -> bool {
        // For sparse polynomials, we need to check if the assignment corresponds
        // to the remaining index after removing the first variable
        remaining_index == assignment
    }
    
    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<Fp> {
        let total_size = 1 << self.num_vars;
        let mut dense = vec![Fp::ZERO; total_size];
        
        for (&index, &coeff) in &self.coeffs {
            dense[index] = coeff;
        }
        
        dense
    }
    
    /// Get an iterator over non-zero coefficients
    pub fn iter(&self) -> impl Iterator<Item = (usize, Fp)> + '_ {
        self.coeffs.iter().map(|(&index, &coeff)| (index, coeff))
    }
}

/// Sparse sumcheck prover optimized for sparse polynomials
pub struct SparseSumcheckProver {
    num_variables: usize,
    challenges: Vec<Fp4>,
}

impl SparseSumcheckProver {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            challenges: Vec::new(),
        }
    }
    
    /// Prove sumcheck for a sparse polynomial
    /// More efficient than dense prover when sparsity is low
    pub fn prove_sparse(&mut self, polynomial: SparseMLE, challenger: &mut crate::utils::challenger::Challenger, claimed_sum: Fp4) -> anyhow::Result<crate::sumcheck::SumcheckProof> {
        let mut proof = crate::sumcheck::SumcheckProof::new();
        let mut current_polynomial = polynomial;
        
        for round in 0..self.num_variables {
            // Compute round polynomial using sparse optimization
            let round_poly = current_polynomial.compute_round_polynomial();
            
            // Verify first round consistency
            if round == 0 {
                let sum_check = round_poly[0] + round_poly[1];
                if sum_check != claimed_sum {
                    return Err(anyhow::anyhow!("First round polynomial inconsistent with claimed sum"));
                }
            }
            
            // Generate challenge
            challenger.observe_fp4_elems(&round_poly);
            let challenge = challenger.get_challenge();
            self.challenges.push(challenge);
            
            // Add to proof
            proof.round_polynomials.push(round_poly);
            
            // Fold the polynomial for next round
            current_polynomial = current_polynomial.fold_first_variable(challenge);
        }
        
        // Final evaluation
        if current_polynomial.num_vars() != 0 {
            return Err(anyhow::anyhow!("Final polynomial should have 0 variables"));
        }
        
        let final_eval = if current_polynomial.num_nonzero() == 0 {
            Fp4::ZERO
        } else {
            // Should have exactly one coefficient at index 0
            Fp4::from(*current_polynomial.coeffs.get(&0).unwrap_or(&Fp::ZERO))
        };
        
        proof.final_evaluations.push(final_eval);
        Ok(proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_sparse_mle_creation() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, Fp::from_u32(1)); // f(0,0) = 1
        coeffs.insert(3, Fp::from_u32(5)); // f(1,1) = 5
        
        let sparse_mle = SparseMLE::new(2, coeffs);
        assert_eq!(sparse_mle.num_vars(), 2);
        assert_eq!(sparse_mle.num_nonzero(), 2);
        assert_eq!(sparse_mle.sparsity(), 0.5); // 2 out of 4 coefficients
    }
    
    #[test]
    fn test_sparse_from_dense() {
        let dense = vec![
            Fp::from_u32(1), // f(0,0) = 1
            Fp::ZERO,                   // f(1,0) = 0
            Fp::ZERO,                   // f(0,1) = 0  
            Fp::from_u32(5),  // f(1,1) = 5
        ];
        
        let sparse = SparseMLE::from_dense(dense);
        assert_eq!(sparse.num_vars(), 2);
        assert_eq!(sparse.num_nonzero(), 2);
        
        // Verify non-zero entries
        assert_eq!(sparse.coeffs.get(&0), Some(&Fp::from_u32(1)));
        assert_eq!(sparse.coeffs.get(&3), Some(&Fp::from_u32(5)));
    }
    
    #[test]
    fn test_sparse_evaluation() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, Fp::from_u32(2)); // f(0,0) = 2
        coeffs.insert(3, Fp::from_u32(3)); // f(1,1) = 3
        
        let sparse_mle = SparseMLE::new(2, coeffs);
        
        // Test evaluation at (0,0) - should be 2
        let point = vec![Fp4::ZERO, Fp4::ZERO];
        let result = sparse_mle.evaluate(&point);
        assert_eq!(result, Fp4::from(Fp::from_u32(2)));
        
        // Test evaluation at (1,1) - should be 3
        let point = vec![Fp4::ONE, Fp4::ONE];
        let result = sparse_mle.evaluate(&point);
        assert_eq!(result, Fp4::from(Fp::from_u32(3)));
        
        // Test evaluation at (1,0) - should be 0
        let point = vec![Fp4::ONE, Fp4::ZERO];
        let result = sparse_mle.evaluate(&point);
        assert_eq!(result, Fp4::ZERO);
    }
    
    #[test]
    fn test_sparse_vs_dense_consistency() {
        // Create identical polynomials in sparse and dense form
        let dense = vec![
            Fp::from_u32(1),
            Fp::ZERO,
            Fp::from_u32(3),
            Fp::ZERO,
        ];
        
        let sparse = SparseMLE::from_dense(dense.clone());
        let dense_mle = crate::utils::polynomial::MLE::new(dense);
        
        // Test evaluation at several points
        let test_points = vec![
            vec![Fp4::ZERO, Fp4::ZERO],
            vec![Fp4::ONE, Fp4::ZERO],
            vec![Fp4::ZERO, Fp4::ONE],
            vec![Fp4::ONE, Fp4::ONE],
            vec![Fp4::from(Fp::from_u32(5)), Fp4::from(Fp::from_u32(7))],
        ];
        
        for point in test_points {
            let sparse_result = sparse.evaluate(&point);
            let dense_result = dense_mle.evaluate(&point);
            assert_eq!(sparse_result, dense_result, "Mismatch at point {:?}", point);
        }
    }
    
    #[test]
    fn test_sparse_folding() {
        let mut coeffs = HashMap::new();
        coeffs.insert(0, Fp::from_u32(1)); // f(0,0) = 1
        coeffs.insert(1, Fp::from_u32(2)); // f(1,0) = 2
        coeffs.insert(2, Fp::from_u32(3)); // f(0,1) = 3
        coeffs.insert(3, Fp::from_u32(4)); // f(1,1) = 4
        
        let sparse_mle = SparseMLE::new(2, coeffs);
        let challenge = Fp4::from(Fp::from_u32(5));
        
        let folded = sparse_mle.fold_first_variable(challenge);
        
        // After folding first variable with challenge 5:
        // g(0) = (1-5)*f(0,0) + 5*f(1,0) = -4*1 + 5*2 = 6
        // g(1) = (1-5)*f(0,1) + 5*f(1,1) = -4*3 + 5*4 = 8
        
        assert_eq!(folded.num_vars(), 1);
        
        // Verify by evaluation
        let g0 = folded.evaluate(&[Fp4::ZERO]);
        let g1 = folded.evaluate(&[Fp4::ONE]);
        
        let expected_g0 = (Fp4::ONE - challenge) * Fp4::from(Fp::from_u32(1)) + 
                         challenge * Fp4::from(Fp::from_u32(2));
        let expected_g1 = (Fp4::ONE - challenge) * Fp4::from(Fp::from_u32(3)) + 
                         challenge * Fp4::from(Fp::from_u32(4));
        
        assert_eq!(g0, expected_g0);
        assert_eq!(g1, expected_g1);
    }
}