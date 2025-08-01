//! Karatsuba-Style Reduction optimization for the enhanced sum-check module.
//! 
//! This module implements the Karatsuba-style reduction optimization described in
//! ePrint 2024/1046, which provides up to 30× improvement for degree-3 compositions
//! by exploiting the structure of polynomial multiplication.
//! 
//! The optimization works by:
//! 1. Using Karatsuba multiplication for polynomial operations
//! 2. Reducing the number of field multiplications from O(d²) to O(d^log₂(3))
//! 3. Optimizing degree-3 compositions specifically
//! 4. Caching intermediate results for repeated operations
//! 
//! # Examples
//! 
//! ```rust
//! use deep_fri::sumcheck::optimizations::karatsuba::{KaratsubaComposer, KaratsubaProver};
//! use deep_fri::sumcheck::config::SumcheckConfig;
//! use deep_fri::utils::polynomial::MLE;
//! use deep_fri::utils::challenger::Challenger;
//! use p3_baby_bear::BabyBear;
//! 
//! // Create a Karatsuba-optimized composer
//! let config = SumcheckConfig::optimized();
//! let composer = KaratsubaComposer::new(config);
//! 
//! // Use with multiple MLEs for degree-3 composition
//! let mle1 = MLE::new(vec![BabyBear::from_canonical_u64(1), BabyBear::from_canonical_u64(2)]);
//! let mle2 = MLE::new(vec![BabyBear::from_canonical_u64(3), BabyBear::from_canonical_u64(4)]);
//! let mles = vec![&mle1, &mle2];
//! let result = composer.compose_batches(&mles, 3);
//! ```

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::sumcheck::config::SumcheckConfig;
use crate::sumcheck::traits::composer::LowDegreeComposer;
use crate::sumcheck::traits::prover::SumcheckProver;
use crate::sumcheck::traits::types::{SumcheckProof, ProverHints, ProverCost};
use crate::utils::polynomial::MLE;
// Note: Challenger trait will be defined elsewhere
use p3_field::{Field, ExtensionField};

/// Threshold for using Karatsuba multiplication over schoolbook
const KARATSUBA_THRESHOLD: usize = 4;

/// Maximum degree for which Karatsuba optimization is most effective
const OPTIMAL_DEGREE_THRESHOLD: usize = 3;

/// Cache size for precomputed Karatsuba matrices
const KARATSUBA_CACHE_SIZE: usize = 256;

/// Karatsuba-optimized low-degree composer.
/// 
/// This composer implements the Karatsuba-style reduction from ePrint 2024/1046,
/// providing significant speedups for degree-3 compositions by reducing the
/// computational complexity of polynomial multiplication.
#[derive(Debug, Clone)]
pub struct KaratsubaComposer<F> 
where
    F: Field,
{
    config: SumcheckConfig,
    /// Cache for precomputed evaluation points
    evaluation_cache: HashMap<usize, Vec<F>>,
    /// Cache for interpolation matrices
    interpolation_cache: HashMap<usize, Vec<Vec<F>>>,
    _phantom: PhantomData<F>,
}

impl<F> KaratsubaComposer<F>
where
    F: Field,
{
    /// Creates a new Karatsuba-optimized composer with the given configuration.
    pub fn new(config: SumcheckConfig) -> Self {
        let mut composer = Self {
            config,
            evaluation_cache: HashMap::new(),
            interpolation_cache: HashMap::new(),
            _phantom: PhantomData,
        };
        
        composer.initialize_caches();
        composer
    }
    
    /// Initialize precomputed caches for common operations.
    fn initialize_caches(&mut self) {
        // Precompute evaluation points for common degrees
        for degree in 1..=8 {
            let points = self.generate_evaluation_points(degree);
            self.evaluation_cache.insert(degree, points);
            
            // Precompute interpolation matrix for this degree
            if degree <= 4 {
                let matrix = self.compute_interpolation_matrix(degree);
                self.interpolation_cache.insert(degree, matrix);
            }
        }
    }
    
    /// Generate evaluation points for Karatsuba multiplication.
    fn generate_evaluation_points(&self, degree: usize) -> Vec<F> {
        let num_points = 2 * degree + 1;
        let mut points = Vec::with_capacity(num_points);
        
        // Use consecutive integers as evaluation points
        for i in 0..num_points {
            points.push(F::from_canonical_usize(i));
        }
        
        points
    }
    
    /// Compute the interpolation matrix for a given degree.
    fn compute_interpolation_matrix(&self, degree: usize) -> Vec<Vec<F>> {
        let num_points = 2 * degree + 1;
        let mut matrix = vec![vec![F::zero(); num_points]; num_points];
        
        // Compute Vandermonde matrix for interpolation
        let points = self.evaluation_cache.get(&degree).unwrap();
        
        for i in 0..num_points {
            for j in 0..num_points {
                matrix[i][j] = points[i].pow(&[j as u64]);
            }
        }
        
        // Invert the matrix (simplified implementation)
        self.invert_matrix(&mut matrix);
        matrix
    }
    
    /// Simple matrix inversion (for small matrices only).
    fn invert_matrix(&self, matrix: &mut Vec<Vec<F>>) {
        let n = matrix.len();
        
        // Create augmented matrix [A|I]
        for i in 0..n {
            matrix[i].resize(2 * n, F::zero());
            matrix[i][n + i] = F::one();
        }
        
        // Gaussian elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            for k in i + 1..n {
                if matrix[k][i] != F::zero() {
                    pivot_row = k;
                    break;
                }
            }
            
            // Swap rows if needed
            if pivot_row != i {
                matrix.swap(i, pivot_row);
            }
            
            // Skip if pivot is zero (singular matrix)
            if matrix[i][i] == F::zero() {
                continue;
            }
            
            // Scale pivot row
            let pivot = matrix[i][i];
            let pivot_inv = pivot.inverse();
            for j in 0..2 * n {
                matrix[i][j] *= pivot_inv;
            }
            
            // Eliminate column
            for k in 0..n {
                if k != i && matrix[k][i] != F::zero() {
                    let factor = matrix[k][i];
                    for j in 0..2 * n {
                        matrix[k][j] -= factor * matrix[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        for i in 0..n {
            matrix[i].drain(0..n);
        }
    }
    
    /// Check if Karatsuba optimization should be used for the given parameters.
    fn should_use_karatsuba(&self, mles: &[&MLE<F>], degree: usize) -> bool {
        if !self.config.use_karatsuba_optimization() {
            return false;
        }
        
        // Karatsuba is most effective for degree-3 compositions
        if degree != OPTIMAL_DEGREE_THRESHOLD {
            return false;
        }
        
        // Check if input size justifies the overhead
        let total_coeffs: usize = mles.iter().map(|mle| mle.coeffs().len()).sum();
        total_coeffs >= KARATSUBA_THRESHOLD
    }
    
    /// Karatsuba multiplication for polynomials.
    fn karatsuba_multiply(&self, a: &[F], b: &[F]) -> Vec<F> {
        let n = a.len().max(b.len());
        
        // Base case: use schoolbook multiplication for small polynomials
        if n <= KARATSUBA_THRESHOLD {
            return self.schoolbook_multiply(a, b);
        }
        
        // Ensure both polynomials have the same length
        let mut a_padded = a.to_vec();
        let mut b_padded = b.to_vec();
        let padded_size = n.next_power_of_two();
        a_padded.resize(padded_size, F::zero());
        b_padded.resize(padded_size, F::zero());
        
        let half = padded_size / 2;
        
        // Split polynomials: a = a_low + x^half * a_high
        let a_low = &a_padded[..half];
        let a_high = &a_padded[half..];
        let b_low = &b_padded[..half];
        let b_high = &b_padded[half..];
        
        // Recursive calls
        let z0 = self.karatsuba_multiply(a_low, b_low);
        let z2 = self.karatsuba_multiply(a_high, b_high);
        
        // Compute (a_low + a_high) * (b_low + b_high)
        let a_sum: Vec<F> = a_low.iter().zip(a_high.iter())
            .map(|(low, high)| *low + *high)
            .collect();
        let b_sum: Vec<F> = b_low.iter().zip(b_high.iter())
            .map(|(low, high)| *low + *high)
            .collect();
        let z1_temp = self.karatsuba_multiply(&a_sum, &b_sum);
        
        // z1 = z1_temp - z0 - z2
        let mut z1 = z1_temp;
        for i in 0..z0.len().min(z1.len()) {
            z1[i] -= z0[i];
        }
        for i in 0..z2.len().min(z1.len()) {
            z1[i] -= z2[i];
        }
        
        // Combine results: result = z0 + x^half * z1 + x^n * z2
        let result_size = a.len() + b.len() - 1;
        let mut result = vec![F::zero(); result_size];
        
        // Add z0
        for (i, &coeff) in z0.iter().enumerate() {
            if i < result.len() {
                result[i] += coeff;
            }
        }
        
        // Add x^half * z1
        for (i, &coeff) in z1.iter().enumerate() {
            let pos = i + half;
            if pos < result.len() {
                result[pos] += coeff;
            }
        }
        
        // Add x^n * z2
        for (i, &coeff) in z2.iter().enumerate() {
            let pos = i + padded_size;
            if pos < result.len() {
                result[pos] += coeff;
            }
        }
        
        result
    }
    
    /// Schoolbook polynomial multiplication for small polynomials.
    fn schoolbook_multiply(&self, a: &[F], b: &[F]) -> Vec<F> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        
        let mut result = vec![F::zero(); a.len() + b.len() - 1];
        
        for (i, &a_coeff) in a.iter().enumerate() {
            for (j, &b_coeff) in b.iter().enumerate() {
                result[i + j] += a_coeff * b_coeff;
            }
        }
        
        result
    }
    
    /// Optimized degree-3 composition using Karatsuba reduction.
    fn compose_degree_3_optimized(&self, mles: &[&MLE<F>]) -> Vec<F> {
        if mles.len() < 3 {
            return self.compose_standard(mles, 3);
        }
        
        // For degree-3 composition: f(x) = a(x) * b(x) * c(x)
        let a_coeffs = mles[0].coeffs();
        let b_coeffs = mles[1].coeffs();
        let c_coeffs = mles[2].coeffs();
        
        // First multiply a(x) * b(x) using Karatsuba
        let ab_product = self.karatsuba_multiply(a_coeffs, b_coeffs);
        
        // Then multiply (a*b)(x) * c(x) using Karatsuba
        let result = self.karatsuba_multiply(&ab_product, c_coeffs);
        
        // Truncate to degree 3
        let mut truncated = result;
        truncated.truncate(4); // degree 3 means 4 coefficients
        truncated.resize(4, F::zero());
        
        truncated
    }
    
    /// Standard composition (fallback).
    fn compose_standard(&self, mles: &[&MLE<F>], degree: usize) -> Vec<F> {
        if mles.is_empty() {
            return vec![];
        }
        
        let mut result = vec![F::zero(); degree + 1];
        
        // Simple composition implementation
        for &mle in mles {
            for (i, &coeff) in mle.coeffs().iter().enumerate() {
                if i <= degree {
                    result[i] += coeff;
                }
            }
        }
        
        result
    }
}

impl<F> LowDegreeComposer<F> for KaratsubaComposer<F>
where
    F: Field,
{
    fn compose_batches(&self, mles: &[&MLE<F>], degree: usize) -> Vec<F> {
        if degree > self.max_supported_degree() {
            panic!("Degree {} exceeds maximum supported degree {}", degree, self.max_supported_degree());
        }
        
        if self.should_use_karatsuba(mles, degree) && degree == 3 {
            self.compose_degree_3_optimized(mles)
        } else {
            self.compose_standard(mles, degree)
        }
    }
    
    fn max_supported_degree(&self) -> usize {
        if self.config.use_karatsuba_optimization() {
            8 // Higher limit with Karatsuba optimization
        } else {
            3 // Standard limit
        }
    }
    
    fn estimated_memory_usage(&self, mles: &[&MLE<F>], degree: usize) -> usize {
        let total_coeffs: usize = mles.iter().map(|mle| mle.coeffs().len()).sum();
        let base_usage = total_coeffs * std::mem::size_of::<F>() * (degree + 1);
        
        if self.should_use_karatsuba(mles, degree) {
            // Karatsuba uses more temporary memory but is more efficient
            base_usage * 3 / 2
        } else {
            base_usage
        }
    }
    
    fn is_suitable(&self, mles: &[&MLE<F>], degree: usize) -> bool {
        if degree > self.max_supported_degree() || mles.is_empty() {
            return false;
        }
        
        // Karatsuba is most suitable for degree-3 compositions
        if degree == 3 && self.config.use_karatsuba_optimization() {
            let total_coeffs: usize = mles.iter().map(|mle| mle.coeffs().len()).sum();
            total_coeffs >= KARATSUBA_THRESHOLD
        } else {
            true
        }
    }
}

/// Karatsuba-optimized sumcheck prover.
/// 
/// This prover integrates the Karatsuba-style reduction optimization
/// into the sumcheck protocol, providing significant speedups for
/// degree-3 polynomial compositions.
#[derive(Debug, Clone)]
pub struct KaratsubaProver<F, EF, C>
where
    F: Field,
    EF: ExtensionField<F>,
    C: Clone,
{
    config: SumcheckConfig,
    composer: KaratsubaComposer<F>,
    _phantom_ef: PhantomData<EF>,
    _phantom_c: PhantomData<C>,
}

impl<F, EF, C> KaratsubaProver<F, EF, C>
where
    F: Field,
    EF: ExtensionField<F>,
    C: Clone,
{
    /// Creates a new Karatsuba-optimized prover.
    pub fn new(config: SumcheckConfig) -> Self {
        let composer = KaratsubaComposer::new(config.clone());
        
        Self {
            config,
            composer,
            _phantom_ef: PhantomData,
            _phantom_c: PhantomData,
        }
    }
    
    /// Check if this prover should be used for the given parameters.
    fn should_use_karatsuba_prover(&self, poly: &MLE<F>, num_rounds: usize) -> bool {
        if !self.config.use_karatsuba_optimization() {
            return false;
        }
        
        // Karatsuba prover is most effective for larger polynomials
        poly.coeffs().len() >= KARATSUBA_THRESHOLD && num_rounds >= 2
    }
    
    /// Compute univariate polynomial using Karatsuba optimization.
    fn compute_karatsuba_univariate(&self, poly: &MLE<F>, round: usize) -> Vec<F> {
        let coeffs = poly.coeffs();
        let half_len = coeffs.len() / 2;
        
        if half_len == 0 {
            return vec![coeffs.get(0).copied().unwrap_or(F::zero())];
        }
        
        // Split coefficients for this round
        let low_coeffs: Vec<F> = (0..half_len).map(|i| coeffs[i * 2]).collect();
        let high_coeffs: Vec<F> = (0..half_len).map(|i| coeffs[i * 2 + 1]).collect();
        
        // Use Karatsuba multiplication for combining
        if low_coeffs.len() >= KARATSUBA_THRESHOLD {
            let combined = self.composer.karatsuba_multiply(&low_coeffs, &high_coeffs);
            combined.into_iter().take(3).collect() // Degree 2 polynomial
        } else {
            // Fallback to standard computation
            let g_0: F = low_coeffs.iter().sum();
            let g_1: F = high_coeffs.iter().sum();
            vec![g_0, g_1 - g_0]
        }
    }
    
    /// Standard univariate computation (fallback).
    fn compute_standard_univariate(&self, poly: &MLE<F>, _round: usize) -> Vec<F> {
        let coeffs = poly.coeffs();
        let half_len = coeffs.len() / 2;
        
        if half_len == 0 {
            return vec![coeffs.get(0).copied().unwrap_or(F::zero())];
        }
        
        let g_0: F = (0..half_len).map(|i| coeffs[i * 2]).sum();
        let g_1: F = (0..half_len).map(|i| coeffs[i * 2 + 1]).sum();
        
        vec![g_0, g_1 - g_0]
    }
    
    /// Fix a variable in the polynomial using the challenge.
    fn fix_variable(&self, poly: &MLE<F>, challenge: F) -> MLE<F> {
        let coeffs = poly.coeffs();
        let half_len = coeffs.len() / 2;
        
        let mut new_coeffs = Vec::with_capacity(half_len);
        
        for i in 0..half_len {
            let low = coeffs[i * 2];
            let high = coeffs[i * 2 + 1];
            // Linear interpolation: low + challenge * (high - low)
            new_coeffs.push(low + challenge * (high - low));
        }
        
        MLE::new(new_coeffs)
    }
}

impl<F, EF, C> SumcheckProver<F, EF, C> for KaratsubaProver<F, EF, C>
where
    F: Field,
    EF: ExtensionField<F>,
    C: Clone,
{
    fn prove(
        &self,
        poly: &MLE<F>,
        challenger: &mut C,
        num_rounds: usize,
    ) -> (SumcheckProof<F>, EF) {
        self.prove_with_hints(poly, challenger, num_rounds, &ProverHints::default())
    }
    
    fn prove_with_hints(
        &self,
        poly: &MLE<F>,
        challenger: &mut C,
        num_rounds: usize,
        _hints: &ProverHints,
    ) -> (SumcheckProof<F>, EF) {
        // Validate inputs
        if num_rounds == 0 {
            panic!("Number of rounds must be positive");
        }
        
        if poly.n_vars() < num_rounds {
            panic!("Polynomial has fewer variables than required rounds");
        }
        
        let use_karatsuba = self.should_use_karatsuba_prover(poly, num_rounds);
        
        // Initialize proof structure
        let mut polynomials = Vec::new();
        let mut challenges = Vec::new();
        
        // Start with the original polynomial
        let mut current_poly = poly.clone();
        
        // Perform sumcheck rounds
        for round in 0..num_rounds {
            // Compute the univariate polynomial for this round
            let univariate_poly = if use_karatsuba {
                self.compute_karatsuba_univariate(&current_poly, round)
            } else {
                self.compute_standard_univariate(&current_poly, round)
            };
            
            // Get challenge from challenger
            // Note: This would need to be implemented based on the actual Challenger trait
            let challenge = F::zero(); // Placeholder
            challenges.push(challenge);
            
            // Store the polynomial
            polynomials.push(univariate_poly);
            
            // Update current polynomial by fixing the variable
            current_poly = self.fix_variable(&current_poly, challenge);
        }
        
        // Compute final evaluation
        let final_evaluation = EF::from_base(current_poly.coeffs()[0]);
        
        // Create proof
        let proof = SumcheckProof {
            polynomials,
            challenges,
            final_evaluation: final_evaluation.as_base().unwrap_or(F::zero()),
        };
        
        (proof, final_evaluation)
    }
    
    fn estimate_cost(&self, poly: &MLE<F>, num_rounds: usize) -> ProverCost {
        let evaluations = poly.coeffs().len();
        let use_karatsuba = self.should_use_karatsuba_prover(poly, num_rounds);
        
        if use_karatsuba {
            // Karatsuba reduces multiplication complexity from O(n²) to O(n^log₂(3))
            let karatsuba_factor = (evaluations as f64).powf(1.585); // log₂(3) ≈ 1.585
            
            ProverCost {
                field_multiplications: (karatsuba_factor * num_rounds as f64) as usize,
                field_additions: evaluations * num_rounds / 2,
                memory_usage: evaluations * std::mem::size_of::<F>() * 3 / 2,
                time_complexity: (karatsuba_factor * num_rounds as f64) as usize,
            }
        } else {
            // Standard complexity
            ProverCost {
                field_multiplications: evaluations * num_rounds * 2,
                field_additions: evaluations * num_rounds,
                memory_usage: evaluations * std::mem::size_of::<F>() * 2,
                time_complexity: evaluations * num_rounds,
            }
        }
    }
    
    fn is_suitable(&self, poly: &MLE<F>, num_rounds: usize) -> bool {
        if !self.config.use_karatsuba_optimization() {
            return false;
        }
        
        // Karatsuba is most suitable for larger polynomials and degree-3 compositions
        poly.coeffs().len() >= KARATSUBA_THRESHOLD && num_rounds >= 2
    }
}

/// Performance metrics for Karatsuba optimization.
#[derive(Debug, Clone, Default)]
pub struct KaratsubaMetrics {
    /// Number of Karatsuba multiplications performed.
    pub karatsuba_multiplications: usize,
    
    /// Number of schoolbook multiplications performed.
    pub schoolbook_multiplications: usize,
    
    /// Total time saved compared to standard approach.
    pub time_saved_ratio: f64,
    
    /// Memory overhead ratio.
    pub memory_overhead_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    
    type F = BabyBear;
    type EF = F; // Simplified for now
    
    #[test]
    fn test_karatsuba_composer_creation() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        assert!(composer.config.use_karatsuba_optimization());
        assert_eq!(composer.max_supported_degree(), 8);
    }
    
    #[test]
    fn test_karatsuba_multiplication() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let a = vec![F::from_canonical_u64(1), F::from_canonical_u64(2), F::from_canonical_u64(3)];
        let b = vec![F::from_canonical_u64(4), F::from_canonical_u64(5)];
        
        let result = composer.karatsuba_multiply(&a, &b);
        
        // Verify result length
        assert_eq!(result.len(), a.len() + b.len() - 1);
        
        // Verify first coefficient (1 * 4 = 4)
        assert_eq!(result[0], F::from_canonical_u64(4));
    }
    
    #[test]
    fn test_schoolbook_multiplication() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let a = vec![F::from_canonical_u64(1), F::from_canonical_u64(2)];
        let b = vec![F::from_canonical_u64(3), F::from_canonical_u64(4)];
        
        let result = composer.schoolbook_multiply(&a, &b);
        
        // Expected: (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], F::from_canonical_u64(3));
        assert_eq!(result[1], F::from_canonical_u64(10));
        assert_eq!(result[2], F::from_canonical_u64(8));
    }
    
    #[test]
    fn test_degree_3_composition() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let mle1 = MLE::new(vec![F::from_canonical_u64(1), F::from_canonical_u64(2)]);
        let mle2 = MLE::new(vec![F::from_canonical_u64(3), F::from_canonical_u64(4)]);
        let mle3 = MLE::new(vec![F::from_canonical_u64(5), F::from_canonical_u64(6)]);
        let mles = vec![&mle1, &mle2, &mle3];
        
        let result = composer.compose_batches(&mles, 3);
        assert_eq!(result.len(), 4); // Degree 3 polynomial
    }
    
    #[test]
    fn test_karatsuba_prover_creation() {
        let config = SumcheckConfig::optimized();
        let prover = KaratsubaProver::<F, EF, ()>::new(config);
        
        assert!(prover.config.use_karatsuba_optimization());
    }
    
    #[test]
    fn test_karatsuba_prover_suitability() {
        let config = SumcheckConfig::optimized();
        let prover = KaratsubaProver::<F, EF, ()>::new(config);
        
        // Small polynomial - not suitable
        let small_poly = MLE::new(vec![F::from_canonical_u64(1), F::from_canonical_u64(2)]);
        assert!(!prover.is_suitable(&small_poly, 1));
        
        // Large polynomial - suitable
        let large_poly = MLE::new(vec![F::from_canonical_u64(1); 16]);
        assert!(prover.is_suitable(&large_poly, 3));
    }
    
    #[test]
    fn test_cost_estimation() {
        let config = SumcheckConfig::optimized();
        let prover = KaratsubaProver::<F, EF, ()>::new(config);
        
        let poly = MLE::new(vec![F::from_canonical_u64(1); 16]);
        let cost = prover.estimate_cost(&poly, 3);
        
        assert!(cost.field_multiplications > 0);
        assert!(cost.memory_usage > 0);
        assert!(cost.time_complexity > 0);
    }
    
    #[test]
    fn test_composer_suitability() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let mle1 = MLE::new(vec![F::from_canonical_u64(1); 8]);
        let mle2 = MLE::new(vec![F::from_canonical_u64(2); 8]);
        let mles = vec![&mle1, &mle2];
        
        // Test degree-3 composition suitability
        assert!(composer.is_suitable(&mles, 3));
        
        // Test unsupported degree
        assert!(!composer.is_suitable(&mles, 10));
    }
    
    #[test]
    fn test_evaluation_points_generation() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let points = composer.generate_evaluation_points(3);
        assert_eq!(points.len(), 7); // 2 * 3 + 1
        
        // Verify points are consecutive integers
        for (i, &point) in points.iter().enumerate() {
            assert_eq!(point, F::from_canonical_usize(i));
        }
    }
    
    #[test]
    fn test_memory_usage_estimation() {
        let config = SumcheckConfig::optimized();
        let composer = KaratsubaComposer::<F>::new(config);
        
        let mle1 = MLE::new(vec![F::from_canonical_u64(1); 8]);
        let mle2 = MLE::new(vec![F::from_canonical_u64(2); 8]);
        let mles = vec![&mle1, &mle2];
        
        let memory_usage = composer.estimated_memory_usage(&mles, 3);
        assert!(memory_usage > 0);
    }
}