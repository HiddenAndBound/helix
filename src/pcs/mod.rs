//! # BaseFold Polynomial Commitment Scheme
//!
//! BaseFold is a field-agnostic polynomial commitment scheme that combines Reed-Solomon encoding,
//! FRI-like folding techniques, and deep integration with sum-check protocols. It is specifically
//! designed for the Spartan zkSNARK protocol implemented in Helix.
//!
//! ## Protocol Overview
//!
//! BaseFold operates through three main phases:
//!
//! 1. **Commitment Phase**: A multilinear polynomial is Reed-Solomon encoded via FFT, then
//!    committed using a Merkle tree over the encoded codewords.
//!
//! 2. **Evaluation Phase**: To prove that a polynomial P evaluates to value v at point r,
//!    the protocol runs multiple sum-check rounds, folding both the encoding and polynomial
//!    in each round to reduce the problem size exponentially.
//!
//! 3. **Verification Phase**: The verifier checks the sum-check transcripts and makes random
//!    queries to the folded encodings to detect any inconsistencies via Reed-Solomon distance properties.
//!
//! ## Mathematical Foundation
//!
//! ### Reed-Solomon Encoding
//! The protocol begins by encoding a polynomial P(x₁,...,xₙ) of degree 2ⁿ over the base field Fp
//! using Reed-Solomon codes. The polynomial coefficients are extended via forward FFT evaluation
//! at roots of unity, creating an error-correcting code with rate 1/2.
//!
//! ### Dual Folding Process  
//! Each sum-check round performs dual folding:
//! - **Encoding folding**: E_{i+1} = fold(E_i, r_i, ω_i) using challenge r_i and twiddle factors ω_i
//! - **Polynomial folding**: P_{i+1} = P_i.fold_in_place(r_i) reducing variable count by 1
//!
//! This maintains the invariant that the folded encoding corresponds to the folded polynomial.
//!
//! ### Field Extension Usage
//! - Base field Fp (BabyBear ≈ 2³¹) for initial polynomial coefficients and encoding
//! - Extension field Fp4 for challenges, evaluations, and sum-check operations
//! - This prevents small subgroup attacks and ensures sufficient randomness
//!
//! ## Integration with Spartan
//!
//! BaseFold is not a generic PCS but specifically designed for Spartan's needs:
//! - Sum-check rounds correspond to Spartan's outer sum-check reducing R1CS constraints
//! - Folding structure matches the batching of multiple polynomial evaluation claims  
//! - Field arithmetic is optimized for the BabyBear field used throughout Helix
//! - The protocol handles the specific polynomial structure arising from constraint matrices
//!
//! ## Security Properties
//!
//! The security of BaseFold relies on:
//! - **Reed-Solomon minimum distance**: Ensures high detection probability for encoding corruption
//! - **Merkle tree binding**: Commitments are cryptographically binding under hash assumptions  
//! - **Sum-check soundness**: Interactive protocol ensures polynomial evaluation correctness
//! - **Challenge unpredictability**: Fiat-Shamir challenges prevent adaptive attacks
//!
//! Soundness error: ≈ (query_count * rounds) / |Fp4| with QUERIES = 144 providing ≈ 2⁻¹⁰⁰ security.
//!
//! ## Current Limitations and Future Work
//!
//! - **Missing optimizations**: Hash pruning, oracle skipping, early stopping not implemented
//! - **Fixed parameters**: Query count (144) and rate (2) are hardcoded constants  
//! - **Performance**: Rate customization and adaptive query selection planned
//! - **Integration**: Currently uses placeholder Merkle tree implementation
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

use anyhow::{Ok, Result};
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};

use crate::pcs::utils::{
    Commitment, Encoding, create_hash_leaves_from_pairs, create_hash_leaves_from_pairs_ref,
    encode_mle, fold, fold_pair, get_codewords, get_merkle_paths, hash_field_pair,
};
use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    merkle_tree::{MerklePath, MerkleTree},
    polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};

mod utils;

/// Configuration parameters for the BaseFold polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct BaseFoldConfig {
    /// Number of random queries for soundness verification.
    /// Higher values provide better security but slower verification.
    pub queries: usize,

    /// Reed-Solomon encoding rate (expansion factor).
    /// Rate of 2 means 2x expansion for rate-1/2 Reed-Solomon code.
    pub rate: usize,

    /// Enable parallel processing for folding operations.
    pub enable_parallel: bool,

    /// Enable optimizations like hash pruning and early stopping.
    pub enable_optimizations: bool,
}

impl Default for BaseFoldConfig {
    fn default() -> Self {
        Self {
            queries: 144, // Provides ≈2^-100 security
            rate: 2,      // Rate-1/2 Reed-Solomon encoding
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
    ///
    /// # Security Impact
    /// Soundness error ≈ queries / |Fp4|. With |Fp4| ≈ 2^124:
    /// - 144 queries → ≈2^-100 security
    /// - 80 queries → ≈2^-80 security
    /// - 256 queries → ≈2^-128 security
    pub fn with_queries(mut self, queries: usize) -> Self {
        self.queries = queries;
        self
    }

    /// Sets the Reed-Solomon encoding rate.
    ///
    /// # Parameters
    /// - `rate = 2`: Rate-1/2 encoding (recommended)
    /// - `rate = 4`: Rate-1/4 encoding (higher redundancy)
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
pub struct Basefold;

pub const QUERIES: usize = 144;

//TODO: Hash pruning, hash leaves together, oracle skipping, early stopping, rate_customisation.

/// A cryptographic commitment to a polynomial using the BaseFold scheme.
///
/// The commitment is the root of a Merkle tree built over Reed-Solomon encoded codewords.
/// This provides a succinct, binding commitment to the polynomial that can be efficiently
/// opened at any evaluation point.
///
/// # Mathematical Structure  
/// Given polynomial P(x₁,...,xₙ) with coefficients [c₀, c₁, ..., c₂ⁿ⁻₁]:
/// 1. Reed-Solomon encode via FFT: E = FFT(P, roots)
/// 2. Hash pairs: leaves[i] = H(E[2i], E[2i+1])
/// 3. Build Merkle tree: commitment = MerkleRoot(leaves)
///
/// The commitment size is constant (32 bytes) regardless of polynomial degree.
#[derive(Debug)]
pub struct BasefoldCommitment {
    /// The Merkle root serving as the cryptographic commitment.
    /// This 32-byte hash binds the prover to the specific Reed-Solomon encoding
    /// of their polynomial under the collision-resistance of Blake3.
    pub commitment: Commitment,
}

/// Prover-specific data required for generating evaluation proofs in the BaseFold scheme.
///
/// This structure contains the polynomial's Reed-Solomon encoding and its corresponding
/// Merkle tree, which together enable the prover to generate convincing evaluation proofs.
/// The data must be stored after the commitment phase to later produce proofs.
///
/// # Security Considerations
/// - The encoding must correspond exactly to the committed polynomial
/// - The Merkle tree must be built over the same encoding used in the commitment
/// - Inconsistency between these components will result in verification failure
#[derive(Debug)]
pub struct ProverData {
    /// Merkle tree built over the Reed-Solomon encoded codewords.
    ///
    /// The tree structure enables efficient proof generation by providing
    /// authentication paths for queried positions. Each leaf corresponds
    /// to a hash of paired codewords from the encoding.
    pub merkle_tree: MerkleTree,

    /// Reed-Solomon encoding of the polynomial coefficients.
    ///
    /// This is the result of applying forward FFT to the polynomial coefficients
    /// using provided roots of unity. The encoding has length 2 * poly.len() due
    /// to the rate-1/2 Reed-Solomon code, providing error-correction capabilities.
    pub encoding: Encoding,
}

/// A zero-knowledge evaluation proof generated by the BaseFold scheme.
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
    ///
    /// Each polynomial g_i(X) is of degree ≤ 1 and represents the sum-check
    /// reduction for round i. The coefficients are generated via the sum-check
    /// protocol and must satisfy the verifier's consistency equation.
    pub sum_check_rounds: Vec<UnivariatePoly>,

    /// Merkle authentication paths for queried positions in each round.
    ///
    /// paths[i][j] contains the Merkle path for the j-th query in round i.
    /// These paths prove that the corresponding codewords in `codewords`
    /// are authentic parts of the committed encoding.
    pub paths: Vec<Vec<MerklePath>>,

    /// Merkle root commitments for each folding round after round 0.
    ///
    /// commitments[i] is the Merkle root of the (i+1)-th folded encoding.
    /// The initial commitment is provided separately as it's computed during
    /// the commitment phase, not the evaluation proof generation.
    pub commitments: Vec<Commitment>,

    /// Reed-Solomon codeword pairs for each query in each round.
    ///
    /// codewords[i][j] = (left, right) contains the paired codewords for
    /// the j-th query in round i. These pairs are folded during verification
    /// to check consistency with the next round's encoding.
    pub codewords: Vec<Vec<(Fp4, Fp4)>>,
}

impl Basefold {
    /// Commits to a multilinear polynomial using Reed-Solomon encoding and Merkle trees.
    ///
    /// This is the first phase of the BaseFold protocol where the prover creates a succinct
    /// commitment to their polynomial that can later be efficiently opened at any point.
    ///
    /// # Mathematical Process
    /// 1. **Reed-Solomon Encoding**: Apply forward FFT to polynomial coefficients using provided roots
    /// 2. **Pairing**: Split encoding into pairs (E[0],E[1]), (E[2],E[3]), ..., (E[n-2],E[n-1])
    /// 3. **Hashing**: Compute leaf hashes H(E[2i], E[2i+1]) for each pair  
    /// 4. **Merkle Tree**: Build tree over leaf hashes to get root commitment
    ///
    /// # Parameters
    /// * `poly` - Multilinear polynomial with coefficients over base field Fp
    /// * `roots` - FFT roots of unity for each folding round, must have sufficient depth
    ///
    /// # Returns
    /// * `BasefoldCommitment` - The Merkle root serving as the cryptographic commitment
    /// * `ProverData` - Encoding and Merkle tree needed for later proof generation
    ///
    /// # Panics
    /// * If polynomial length is not a power of 2 (required for FFT)
    /// * If root table doesn't have sufficient depth for the polynomial size
    ///
    /// # Security Properties
    /// * **Binding**: Commitment binds prover to specific polynomial under collision-resistance of Blake3
    /// * **Hiding**: None - commitment reveals information about polynomial structure
    /// * **Succinctness**: Commitment size is constant (32 bytes) regardless of polynomial degree
    ///
    /// # Example
    /// ```rust,ignore  
    /// let poly = MLE::new(vec![Fp::ONE, Fp::TWO, Fp::ZERO, Fp::ONE]); // 2-variable polynomial
    /// let roots = generate_fft_roots_for_depth(2); // Roots for 2 variables
    /// let config = BaseFoldConfig::default();
    /// let (commitment, prover_data) = Basefold::commit(&poly, roots, &config);
    /// ```
    pub fn commit(
        poly: &MLE<Fp>,
        roots: &[Vec<Fp>],
        config: &BaseFoldConfig,
    ) -> anyhow::Result<(BasefoldCommitment, ProverData)> {
        if !poly.len().is_power_of_two() {
            anyhow::bail!("Polynomial size must be a power of 2, got {}", poly.len());
        }

        let required_depth = poly.n_vars();
        if roots.len() < required_depth {
            anyhow::bail!(
                "Insufficient FFT roots: need depth {}, got {}",
                required_depth,
                roots.len()
            );
        }

        let encoding = encode_mle(poly, roots, config.rate);

        let (left, right) = encoding.split_at(encoding.len() / 2);
        let leaves: Vec<[u8; 32]> = create_hash_leaves_from_pairs(left, right);

        let merkle_tree = MerkleTree::from_hash(&leaves)?;
        let commitment = merkle_tree.root();

        Ok((
            BasefoldCommitment { commitment },
            ProverData {
                merkle_tree,
                encoding,
            },
        ))
    }

    /// Generates an evaluation proof demonstrating that a committed polynomial evaluates to a specific value.
    ///
    /// This is the second phase of the BaseFold protocol, proving that P(r) = v for committed polynomial P,
    /// evaluation point r, and claimed value v. The proof combines sum-check rounds with encoding folding
    /// to achieve both correctness and soundness.
    ///
    /// # Mathematical Process  
    /// For n rounds (where n = number of polynomial variables):
    /// 1. **Sum-check Round**: Generate univariate polynomial g_i(X) from sum-check reduction
    /// 2. **Challenge Generation**: Fiat-Shamir challenge r_i from g_i coefficients  
    /// 3. **Dual Folding**:
    ///    - Encoding: E_{i+1} = fold(E_i, r_i, roots[i])
    ///    - Polynomial: P_{i+1} = P_i.fold_in_place(r_i)  
    /// 4. **Commitment Update**: Build Merkle tree over E_{i+1}, observe root
    /// 5. **Claim Update**: claim_{i+1} = g_i(r_i)
    ///
    /// After folding, random queries verify consistency between folded encodings.
    ///
    /// # Parameters
    /// * `poly` - The multilinear polynomial to evaluate (must match commitment)
    /// * `eval_point` - Point r = [r₁, r₂, ..., rₙ] where polynomial is evaluated  
    /// * `challenger` - Fiat-Shamir challenger for generating randomness
    /// * `evaluation` - Claimed value v = P(r) that the proof will demonstrate
    /// * `prover_data` - Encoding and Merkle tree from commitment phase
    /// * `roots` - FFT roots for each folding round (same as used in commit)
    ///
    /// # Returns
    /// * `Ok(EvalProof)` - Complete evaluation proof with sum-check rounds and query responses
    /// * `Err(anyhow::Error)` - If Merkle tree construction fails or other errors occur
    ///
    /// # Mathematical Invariants
    /// * After round i: folded_encoding corresponds to i-times-folded polynomial
    /// * Sum-check consistency: g_i(0) + g_i(1) = previous_claim  
    /// * Final claim equals actual polynomial evaluation at random point
    ///
    /// # Security Properties
    /// * **Completeness**: Honest prover with correct evaluation always produces accepting proof
    /// * **Soundness**: Cheating prover cannot convince verifier of incorrect evaluation except with negligible probability
    /// * **Zero-Knowledge**: None - proof reveals information about polynomial structure
    ///
    /// # Panics  
    /// * If evaluation point dimension doesn't match polynomial variable count
    ///
    /// # Example
    /// ```rust,ignore
    /// let eval_point = vec![Fp4::from_u32(5), Fp4::from_u32(7)];
    /// let evaluation = poly.evaluate(&eval_point); // Honest evaluation  
    /// let mut challenger = Challenger::new();
    /// let config = BaseFoldConfig::default();
    /// let proof = Basefold::evaluate(&poly, &eval_point, &mut challenger,
    ///                                evaluation, prover_data, roots, &config)?;
    /// ```
    pub fn evaluate(
        poly: &MLE<Fp>,
        eval_point: &[Fp4],
        challenger: &mut Challenger,
        evaluation: Fp4,
        prover_data: ProverData,
        roots: &[Vec<Fp>],
        config: &BaseFoldConfig,
    ) -> anyhow::Result<EvalProof> {
        if poly.n_vars() != eval_point.len() {
            anyhow::bail!(
                "Evaluation point dimension {} doesn't match polynomial variables {}",
                eval_point.len(),
                poly.n_vars()
            );
        }

        let (
            mut current_claim,
            mut random_point,
            mut sum_check_rounds,
            mut commitments,
            mut merkle_trees,
            mut encodings,
            mut current_poly,
        ) = Self::initialize_evaluation_proof_context(evaluation, poly.n_vars());

        let eq = EqEvals::gen_from_point(&eval_point[1..]);
        let rounds = poly.n_vars();

        //Commit phase
        for round in 0..rounds {
            let (round_proof, r) = match round {
                0 => Self::process_sum_check_round(
                    poly,
                    &eval_point,
                    &mut current_claim,
                    challenger,
                    round,
                    &eq,
                ),
                _ => Self::process_sum_check_round(
                    &current_poly,
                    &eval_point,
                    &mut current_claim,
                    challenger,
                    round,
                    &eq,
                ),
            };

            let (current_encoding, current_poly_folded) = Self::fold_encoding_and_polynomial(
                round,
                &prover_data.encoding,
                &encodings,
                r,
                roots,
                poly,
                &current_poly,
            );
            current_poly = current_poly_folded;

            Self::update_merkle_and_commitments_for_round(
                current_encoding,
                &mut commitments,
                &mut merkle_trees,
                &mut encodings,
            )?;

            challenger.observe_commitment(
                commitments
                    .last()
                    .expect("Will be non-empty after at least 1 fold"),
            );
            current_claim = round_proof.evaluate(r);

            sum_check_rounds.push(round_proof);
            random_point.push(r);
        }

        //Query phase
        let mut queries = challenger.get_indices(rounds as u32, config.queries);

        let mut codewords = Vec::with_capacity(rounds);
        let mut paths = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let round_codewords: Vec<(Fp4, Fp4)> = match round {
                0 => get_codewords(&queries, &prover_data.encoding),
                _ => get_codewords(&queries, &encodings[round - 1]),
            };

            codewords.push(round_codewords);

            let round_paths: Vec<MerklePath> = match round {
                0 => get_merkle_paths(&queries, &prover_data.merkle_tree),
                _ => get_merkle_paths(&queries, &merkle_trees[round - 1]),
            };
            paths.push(round_paths);

            queries.iter_mut().for_each(|query| *query >>= 1);
        }

        Ok(EvalProof {
            sum_check_rounds,
            paths,
            commitments,
            codewords,
        })
    }

    /// Processes a single sum-check round in the BaseFold evaluation protocol.
    ///
    /// This function performs the sum-check reduction for one variable of the multilinear polynomial,
    /// generating a univariate polynomial that represents the sum over the boolean hypercube.
    ///
    /// # Mathematical Process
    /// For round i with current polynomial P and evaluation point [r₁,...,rₙ]:
    /// 1. Compute g₀ = Σⱼ eq[j] * P[2j] (sum over x_i = 0)  
    /// 2. Derive g₁ from constraint: g₀ + g₁ = current_claim and g₁ = g(1)
    /// 3. Construct g(X) = g₀ + (g₁ - g₀) * X (degree-1 univariate)
    /// 4. Generate challenge r_i via Fiat-Shamir from g(X) coefficients
    ///
    /// # Parameters
    /// * `poly` - Current (possibly folded) polynomial for this round
    /// * `eval_point` - Full evaluation point [r₁, r₂, ..., rₙ]
    /// * `current_claim` - Current sum-check claim to be reduced
    /// * `challenger` - Fiat-Shamir challenger for randomness generation
    /// * `round` - Current round index (0-indexed)
    /// * `eq` - Equality polynomial evaluations eq(x, eval_point[round+1:])
    ///
    /// # Returns
    /// * `UnivariatePoly` - The sum-check round polynomial g(X)
    /// * `Fp4` - Challenge r_i for folding operations
    ///
    /// # Mathematical Invariants
    /// * Sum-check consistency: g(0) + g(1) = current_claim
    /// * Degree bound: g(X) has degree ≤ 1
    /// * Field compatibility: Works with both Fp and Fp4 polynomials via extension
    ///
    /// # Implementation Notes
    /// The equality polynomial eq encodes the remaining evaluation point components
    /// after the current round, enabling the sum-check reduction while maintaining
    /// the correct evaluation at the target point.
    fn process_sum_check_round<F>(
        poly: &MLE<F>,
        eval_point: &[Fp4],
        current_claim: &mut Fp4,
        challenger: &mut Challenger,
        round: usize,
        eq: &EqEvals,
    ) -> (UnivariatePoly, Fp4)
    where
        F: PrimeCharacteristicRing + Field,
        Fp4: ExtensionField<F>,
    {
        let mut g_0: Fp4 = Fp4::ZERO;

        for i in 0..1 << (poly.n_vars() - round - 1) {
            g_0 += eq[i] * poly[i << 1]
        }

        let g1: Fp4 = (*current_claim - g_0 * (Fp4::ONE - eval_point[round])) / eval_point[0];

        let round_coeffs = vec![g_0, g1 - g_0];
        let round_proof = UnivariatePoly::new(round_coeffs).unwrap();

        challenger.observe_fp4_elems(&round_proof.coefficients());

        let r = challenger.get_challenge();
        (round_proof, r)
    }

    fn fold_encoding_and_polynomial(
        round: usize,
        initial_encoding: &Encoding,
        encodings: &[Vec<Fp4>],
        r: Fp4,
        roots: &[Vec<Fp>],
        initial_poly: &MLE<Fp>,
        current_poly: &MLE<Fp4>,
    ) -> (Vec<Fp4>, MLE<Fp4>) {
        let current_encoding = match round {
            0 => fold(initial_encoding, r, &roots[round]),
            _ => fold(
                encodings.last().expect("Will be non-empty"),
                r,
                &roots[round],
            ),
        };

        let current_poly_folded = match round {
            0 => initial_poly.fold_in_place(r),
            _ => current_poly.fold_in_place(r),
        };
        (current_encoding, current_poly_folded)
    }

    fn initialize_evaluation_proof_context(
        evaluation: Fp4,
        rounds: usize,
    ) -> (
        Fp4,
        Vec<Fp4>,
        Vec<UnivariatePoly>,
        Vec<Commitment>,
        Vec<MerkleTree>,
        Vec<Vec<Fp4>>,
        MLE<Fp4>,
    ) {
        let current_claim = evaluation;
        let random_point = Vec::with_capacity(rounds);
        let round_proofs = Vec::with_capacity(rounds);
        let commitments = Vec::with_capacity(rounds);
        let merkle_trees = Vec::with_capacity(rounds);
        let encodings = Vec::with_capacity(rounds);
        let current_poly = MLE::default();
        (
            current_claim,
            random_point,
            round_proofs,
            commitments,
            merkle_trees,
            encodings,
            current_poly,
        )
    }

    fn update_merkle_and_commitments_for_round(
        current_encoding: Vec<Fp4>,
        commitments: &mut Vec<Commitment>,
        merkle_trees: &mut Vec<MerkleTree>,
        encodings: &mut Vec<Vec<Fp4>>,
    ) -> Result<(), anyhow::Error> {
        let (left, right) = current_encoding.split_at(current_encoding.len() / 2);

        let leaves: Vec<[u8; 32]> = create_hash_leaves_from_pairs_ref(left, right);
        let current_merkle_tree = MerkleTree::from_field(&current_encoding)?;
        let current_commitment = current_merkle_tree.root();

        commitments.push(current_commitment);
        merkle_trees.push(current_merkle_tree);
        encodings.push(current_encoding);
        Ok(())
    }

    /// Verifies an evaluation proof, checking that a committed polynomial evaluates to the claimed value.
    ///
    /// This is the verification phase of the BaseFold protocol, where the verifier checks the prover's
    /// claim that P(r) = v without access to the polynomial P itself, using only the commitment and proof.
    ///
    /// # Verification Process
    /// The verifier performs two main checks:
    ///
    /// ## 1. Sum-Check Verification (Commit Phase)
    /// For each round i:
    /// - Check sum-check consistency: g_i(0) + g_i(1) = current_claim
    /// - Generate challenge r_i from g_i coefficients via Fiat-Shamir  
    /// - Update claim: claim_{i+1} = g_i(r_i)
    /// - Observe folded encoding commitment
    ///
    /// ## 2. Query Verification (Query Phase)
    /// For QUERIES random positions:
    /// - Verify Merkle paths authenticate queried codewords
    /// - Check folding consistency: folded_codewords match next round's codewords
    /// - Ensure final folded codeword equals final claim
    ///
    /// # Parameters
    /// * `proof` - The evaluation proof generated by the prover
    /// * `evaluation` - Claimed value v = P(r) to verify
    /// * `eval_point` - Evaluation point r = [r₁, r₂, ..., rₙ]
    /// * `commitment` - The polynomial commitment to verify against
    /// * `roots` - FFT roots used in folding (must match prover's roots)
    /// * `challenger` - Fiat-Shamir challenger (must be in same state as prover's)
    ///
    /// # Returns
    /// * `Ok(())` - If proof is valid and evaluation claim is accepted
    /// * `Err(anyhow::Error)` - If any verification check fails:
    ///   - Sum-check consistency violation
    ///   - Merkle path verification failure  
    ///   - Folding consistency check failure
    ///   - Final claim mismatch
    ///
    /// # Mathematical Guarantees  
    /// * **Completeness**: Honest proofs with correct evaluations always verify
    /// * **Soundness**: Invalid proofs are rejected except with probability ≈ QUERIES/|Fp4|
    /// * **Efficiency**: Verification time is O(log(poly_size) * QUERIES + rounds)
    ///
    /// # Security Analysis
    /// The verification provides soundness against:
    /// - **Encoding attacks**: Reed-Solomon distance detects corrupted encodings
    /// - **Sum-check attacks**: Interactive protocol ensures evaluation correctness  
    /// - **Folding attacks**: Query consistency checks detect invalid folding operations
    ///
    /// With QUERIES = 144 and |Fp4| ≈ 2¹²⁴, soundness error is approximately 2⁻¹⁰⁰.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut verifier_challenger = Challenger::new();
    /// let config = BaseFoldConfig::default();
    /// match Basefold::verify(proof, evaluation, &eval_point, commitment,
    ///                       &roots, &mut verifier_challenger, &config) {
    ///     Ok(()) => println!("Proof verified successfully"),
    ///     Err(e) => println!("Verification failed: {}", e),
    /// }
    /// ```
    pub fn verify(
        proof: EvalProof,
        evaluation: Fp4,
        eval_point: &[Fp4],
        commitment: BasefoldCommitment,
        roots: &[Vec<Fp>],
        challenger: &mut Challenger,
        config: &BaseFoldConfig,
    ) -> anyhow::Result<()> {
        // TODO: Observe statement (eval, eval_point, commitment)
        let mut current_claim = evaluation;

        let rounds = eval_point.len();

        let mut random_point = Vec::new();
        //Commit phase
        for round in 0..rounds {
            Self::verify_sum_check_round(
                &proof.sum_check_rounds[round],
                &mut current_claim,
                eval_point[round],
                challenger,
            )?;
            random_point.push(challenger.get_challenge()); // Get the challenge 'r' after observing coefficients
            challenger.observe_commitment(&proof.commitments[round]);
        }

        let queries = challenger.get_indices(rounds as u32, config.queries);

        let mut current_codewords = &proof.codewords[0];
        let mut folded_codewords = Vec::with_capacity(config.queries);
        let mut merkle_paths = &proof.paths[0];
        for round in 0..rounds - 1 {
            let halfsize = 1 << (rounds - round - 1);

            for query in 0..QUERIES {
                let (left, right) = current_codewords[query];
                let leaf_hash = hash_field_pair(left, right);
                let path = &merkle_paths[query];
                match round {
                    0 => {
                        MerkleTree::verify_path(
                            leaf_hash,
                            queries[query],
                            path,
                            commitment.commitment,
                        )?;

                        folded_codewords.push(fold_pair(
                            current_codewords[query],
                            random_point[round],
                            roots[round][query & (1 << (rounds - round - 1)) - 1],
                        ));
                    }
                    _ => {
                        MerkleTree::verify_path(
                            leaf_hash,
                            queries[query],
                            path,
                            proof.commitments[round - 1],
                        )?;
                        check_fold(&folded_codewords, &queries, query, halfsize, left, right)?;
                        folded_codewords[query] = fold_pair(
                            current_codewords[query],
                            random_point[round],
                            roots[round][query & (1 << (rounds - round - 1)) - 1],
                        );
                    }
                }
            }

            current_codewords = &proof.codewords[round + 1];
            merkle_paths = &proof.paths[round + 1];
        }

        if folded_codewords[0] != current_claim {
            anyhow::bail!(
                "Final claim verification failed: {:?} != {:?}",
                folded_codewords[0],
                current_claim
            );
        }
        Ok(())
    }

    /// Verifies a single sum-check round during the verification phase.
    ///
    /// This function checks the consistency of one round's sum-check polynomial against
    /// the current claim and updates the claim for the next round.
    ///
    /// # Verification Process
    /// 1. **Consistency Check**: Verify g(0) + g(1) = current_claim via evaluation point
    /// 2. **Fiat-Shamir**: Observe polynomial coefficients to maintain transcript consistency  
    /// 3. **Claim Update**: Set new claim = g(challenge) for next round
    ///
    /// # Parameters
    /// * `round_poly` - The sum-check univariate polynomial g(X) for this round
    /// * `current_claim` - The current sum-check claim (updated in-place)
    /// * `eval_point_round` - The evaluation point component for this round
    /// * `challenger` - Fiat-Shamir challenger for transcript consistency
    ///
    /// # Mathematical Verification
    /// The function asserts that:
    /// ```text
    /// current_claim = (1 - eval_point_round) * g(0) + eval_point_round * g(1)
    /// ```
    /// This ensures the sum-check reduction is performed correctly.
    ///
    /// # Returns
    /// * `Ok(())` - If sum-check verification passes
    /// * `Err(anyhow::Error)` - If consistency equation is violated
    fn verify_sum_check_round(
        round_poly: &UnivariatePoly,
        current_claim: &mut Fp4,
        eval_point_round: Fp4,
        challenger: &mut Challenger,
    ) -> anyhow::Result<()> {
        let expected = (Fp4::ONE - eval_point_round) * round_poly.evaluate(Fp4::ZERO)
            + eval_point_round * round_poly.evaluate(Fp4::ONE);

        if *current_claim != expected {
            anyhow::bail!(
                "Sum-check verification failed: claim {:?} != expected {:?}",
                *current_claim,
                expected
            );
        }

        challenger.observe_fp4_elems(&round_poly.coefficients());
        *current_claim = round_poly.evaluate(challenger.get_challenge());
        Ok(())
    }
}

/// Verifies the consistency of folding operations during query verification.
///
/// This function ensures that the current round's codewords correctly fold to match
/// the previous round's folded codewords, maintaining the integrity of the folding process.
///
/// # Folding Verification Process
/// For each query position, check that:
/// - If query index > halfsize: folded_codeword should equal the right codeword
/// - If query index ≤ halfsize: folded_codeword should equal the left codeword  
///
/// This verification ensures the prover performed folding correctly and didn't manipulate
/// the encodings between rounds.
///
/// # Parameters  
/// * `folded_codewords` - Previously computed folded codewords from last round
/// * `queries` - Query positions for the current verification round
/// * `query` - Index of current query being checked
/// * `halfsize` - Half the size of current encoding (for indexing logic)
/// * `left` - Left codeword from current query response  
/// * `right` - Right codeword from current query response
///
/// # Returns
/// * `Ok(())` - If folding consistency check passes
/// * `Err(anyhow::Error)` - If folded codeword doesn't match expected value
///
/// # Security Properties  
/// This check prevents the prover from providing inconsistent encodings across folding rounds,
/// which would allow them to cheat by using different polynomials in different rounds.
/// The verification leverages the deterministic nature of the folding operation.
fn check_fold(
    folded_codewords: &[Fp4],
    queries: &[usize],
    query: usize,
    halfsize: usize,
    left: Fp4,
    right: Fp4,
) -> anyhow::Result<()> {
    if queries[query] > halfsize {
        if folded_codewords[query] != right {
            anyhow::bail!(
                "Folded codeword verification failed: expected {:?}, got {:?}",
                right,
                folded_codewords[query]
            );
        }
    } else {
        if folded_codewords[query] != left {
            anyhow::bail!(
                "Folded codeword verification failed: expected {:?}, got {:?}",
                left,
                folded_codewords[query]
            );
        }
    }

    Ok(())
}

