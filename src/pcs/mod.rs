//! # BaseFold Polynomial Commitment Scheme
//!
//! BaseFold is a field-agnostic polynomial commitment scheme designed for the Spartan zkSNARK protocol.
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use helix::pcs::{Basefold, BasefoldCommitment};
//! use helix::polynomial::MLE;
//! use helix::challenger::Challenger;
//!
//! // Commit to a polynomial
//! let poly = MLE::new(vec![Fp::ONE, Fp::TWO, Fp::ZERO, Fp::ONE]);
//! let roots = generate_fft_roots(); // FFT roots for encoding
//! let (commitment, prover_data) = Basefold::commit(&poly, roots);
//!
//! // Generate evaluation proof
//! let eval_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
//! let evaluation = poly.evaluate(&eval_point);
//! let mut challenger = Challenger::new();
//! let proof = Basefold::evaluate(&poly, &eval_point, &mut challenger,
//!                                evaluation, prover_data, roots)?;
//!
//! // Verify the proof
//! let mut verifier_challenger = Challenger::new();
//! Basefold::verify(proof, evaluation, &eval_point, commitment,
//!                  &roots, &mut verifier_challenger)?;
//! ```

use crate::pcs::utils::{Commitment, Encoding};
use crate::{
    Fp4,
    merkle_tree::{MerklePath, MerkleTree},
    spartan::univariate::UnivariatePoly,
};

pub mod prover;
pub mod utils;
pub mod verifier;

/// Configuration parameters for BaseFold PCS.
#[derive(Debug, Clone)]
pub struct BaseFoldConfig {
    /// Number of random queries for soundness verification.
    pub queries: usize,
    /// Reed-Solomon encoding rate (expansion factor).
    pub rate: usize,
    /// Enable parallel processing for folding operations.
    pub enable_parallel: bool,
    /// Enable optimizations like hash pruning and early stopping.
    pub enable_optimizations: bool,
}

impl Default for BaseFoldConfig {
    fn default() -> Self {
        Self {
            queries: 144,
            rate: 2,
            enable_parallel: false,
            enable_optimizations: false,
        }
    }
}

impl BaseFoldConfig {
    /// Creates a new BaseFold configuration with default security parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of queries for soundness verification.
    pub fn with_queries(mut self, queries: usize) -> Self {
        self.queries = queries;
        self
    }

    /// Sets the Reed-Solomon encoding rate.
    pub fn with_rate(mut self, rate: usize) -> Self {
        self.rate = rate;
        self
    }

    /// Enables or disables parallel processing.
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Enables or disables performance optimizations.
    pub fn with_optimizations(mut self, enable: bool) -> Self {
        self.enable_optimizations = enable;
        self
    }

    /// Creates a high-security configuration with more queries.
    pub fn high_security() -> Self {
        Self {
            queries: 256,
            rate: 2,
            enable_parallel: true,
            enable_optimizations: true,
        }
    }

    /// Creates a fast configuration with fewer queries (lower security).
    pub fn fast() -> Self {
        Self {
            queries: 80,
            rate: 2,
            enable_parallel: true,
            enable_optimizations: true,
        }
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.queries == 0 {
            anyhow::bail!("Query count must be greater than 0");
        }
        if self.rate == 0 || !self.rate.is_power_of_two() {
            anyhow::bail!("Rate must be a positive power of 2");
        }
        Ok(())
    }
}

/// The BaseFold polynomial commitment scheme implementation.
///
/// BaseFold combines Reed-Solomon encoding with FRI-like folding techniques to create
/// a field-agnostic polynomial commitment scheme optimized for integration with sum-check protocols.
///
/// The actual implementation methods are defined in the `prover` and `verifier` modules.
pub struct Basefold;

// TODO: Optimizations - hash pruning, oracle skipping, early stopping, rate customisation
// Implementation methods are in the prover and verifier modules

/// A cryptographic commitment to a polynomial using BaseFold.
#[derive(Debug)]
pub struct BasefoldCommitment {
    /// The Merkle root serving as the cryptographic commitment.
    pub commitment: Commitment,
}

/// Prover-specific data required for generating evaluation proofs.
#[derive(Debug)]
pub struct ProverData {
    /// Merkle tree built over the Reed-Solomon encoded codewords.
    pub merkle_tree: MerkleTree,

    /// Reed-Solomon encoding of the polynomial coefficients.
    pub encoding: Encoding,
}

/// Evaluation proof demonstrating polynomial evaluation correctness.
///
/// This proof demonstrates that a committed polynomial P evaluates to a specific value v
/// at a given point r, i.e., P(r) = v. The proof combines sum-check transcripts with
/// query-response data to achieve both correctness and soundness.
///
/// # Protocol Structure
/// The proof is generated through n rounds (where n = number of variables):
/// 1. Each round produces a sum-check univariate polynomial
/// 2. Encoding and polynomial are folded using verifier's challenge
/// 3. New commitment is generated for the folded encoding
/// 4. Random queries verify consistency of folding operations
///
/// # Verification Process
/// - Sum-check rounds ensure polynomial evaluation correctness
/// - Merkle paths authenticate queried codewords
/// - Folding consistency checks detect encoding manipulation
#[derive(Debug)]
pub struct EvalProof {
    /// Univariate polynomials from each sum-check round.
    pub sum_check_rounds: Vec<UnivariatePoly>,

    /// Merkle authentication paths for queried positions.
    pub paths: Vec<Vec<MerklePath>>,

    /// Merkle root commitments for each folding round.
    pub commitments: Vec<Commitment>,

    /// Reed-Solomon codeword pairs for each query.
    pub codewords: Vec<Vec<(Fp4, Fp4)>>,
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;
    use crate::pcs::prover::update_query;
    use crate::pcs::utils::{encode_mle, fold, fold_pair, get_codewords};
    use crate::{Fp, challenger::Challenger, polynomial::MLE};

    #[test]
    fn test_basefold() -> Result<(), anyhow::Error> {
        // Test the BaseFold commitment scheme
        let mut rng = StdRng::seed_from_u64(0);
        let mut challenger = Challenger::new();

        const N_VARS: usize = 4;
        let roots = Fp::roots_of_unity_table(1 << (N_VARS + 1));
        let mle = MLE::new(
            (0..1 << N_VARS)
                .map(|_| Fp::from_u32(rng.r#gen()))
                .collect(),
        );

        let eval_point: Vec<Fp4> = (0..N_VARS).map(|_| Fp4::from_u128(rng.r#gen())).collect();
        let evaluation = mle.evaluate(&eval_point);
        let mut config = BaseFoldConfig::new();

        config = config.with_queries(5);
        let (commitment, prover_data) = Basefold::commit(&mle, &roots, &config).unwrap();
        let eval_proof = Basefold::evaluate(
            &mle,
            &eval_point,
            &mut challenger,
            evaluation,
            prover_data,
            &roots,
            &config,
        )
        .unwrap();
        let mut challenger = Challenger::new();
        Basefold::verify(
            eval_proof,
            evaluation,
            &eval_point,
            commitment,
            &roots,
            &mut challenger,
            &config,
        )?;

        Ok(())
    }

    #[test]
    fn test_fold() {
        let mut rng = StdRng::seed_from_u64(0);

        let poly = MLE::new((0..1 << 4).map(|_| Fp::from_u32(rng.r#gen())).collect());
        let roots = Fp::roots_of_unity_table(1 << 5);
        let eval_point: Vec<Fp4> = (0..4).map(|_| Fp4::from_u128(rng.r#gen())).collect();
        let eval = poly.evaluate(&eval_point);
        let encoding = encode_mle(&poly, &roots, 2);
        let mut encoding: Vec<Fp4> = encoding.iter().map(|&x| Fp4::from(x)).collect();
        for i in 0..4 {
            let r = eval_point[i];
            encoding = fold(&encoding, r, &roots[i]);
        }
        println!("{:?}", eval);
        println!("{:?}", encoding);
    }

    #[test]
    fn test_fft() {
        let poly: Vec<Fp> = (0..16).map(|i| Fp::new(i)).collect();

        let roots = Fp::roots_of_unity_table(1 << 4);

        let mut test = poly.clone();

        BabyBear::forward_fft(&mut test, &roots);

        println!("{:?}", test);

        let powers: Vec<Fp> = roots[0][1].powers().take(16).collect();
        println!("{:?}", powers);
        println!("{:?}", roots[0]);
    }

    #[test]
    fn test_query() {
        let val = 1 << (6 + 1 - 1 - 1);
        let test = 33 & ((1 << (6 + 1 - 1 - 1)) - 1);
        println!("{:b}", val);
        println!("{:b}", test);
    }

    #[test]
    fn test_folding_queries() {
        // The test should test that the queried positions checked each round is correct

        let mut rng = StdRng::seed_from_u64(0);

        let poly = MLE::new((0..1 << 4).map(|_| Fp::from_u32(rng.r#gen())).collect());
        let roots = Fp::roots_of_unity_table(1 << 5);
        let eval_point: Vec<Fp4> = (0..4).map(|_| Fp4::from_u128(rng.r#gen())).collect();
        let encoding = encode_mle(&poly, &roots, 2);

        // We start with query at 5. Thus the provided codeword should be encoding[5] and encoding[(5 + domain_size) = 21]
        let queries = vec![5];

        let correct_codeword = (encoding[5].into(), encoding[21].into());

        let received_codeword = get_codewords(&queries, &encoding);

        assert_eq!(correct_codeword, received_codeword[0]);

        let folded_codeword = fold_pair(received_codeword[0], eval_point[0], roots[0][5]);

        let folded_oracle = fold(&encoding, eval_point[0], &roots[0]);

        assert_eq!(folded_codeword, folded_oracle[5]);
    }

    #[test]
    fn test_bitwise_query_updates() {
        // Test that bitwise operations produce identical results to arithmetic operations
        let test_cases = [
            (5, 8),   // query < halfsize
            (13, 8),  // query >= halfsize
            (7, 16),  // query < halfsize
            (23, 16), // query >= halfsize
        ];

        for (query, halfsize) in test_cases {
            let mut bitwise_query = query;
            let mut arithmetic_query = query;

            // Bitwise operation (new implementation)
            update_query(&mut bitwise_query, halfsize);

            // Arithmetic operation (old implementation)
            if arithmetic_query >= halfsize {
                arithmetic_query -= halfsize;
            }

            assert_eq!(
                bitwise_query, arithmetic_query,
                "Bitwise and arithmetic operations should produce identical results for query={}, halfsize={}",
                query, halfsize
            );
        }
    }
}
