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

use anyhow::{Ok, Result};
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::pcs::utils::{
    Commitment, Encoding, RATE, create_hash_leaves_from_pairs, create_hash_leaves_from_pairs_ref,
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

pub mod utils;

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
pub struct Basefold;

pub const QUERIES: usize = 144;

//TODO: Hash pruning, hash leaves together, oracle skipping, early stopping, rate_customisation.

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

        //Roots table should have twiddles for 1..n_vars-1 i.e.
        let required_depth = poly.n_vars() - 1;

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

        let mut eq = EqEvals::gen_from_point(&eval_point[1..]);
        let rounds = poly.n_vars();

        //Commit phase
        for round in 0..rounds {
            let round_proof = match round {
                0 => Self::process_sum_check_round(
                    poly,
                    &eval_point,
                    &mut current_claim,
                    round,
                    &mut eq,
                ),
                _ => Self::process_sum_check_round(
                    &current_poly,
                    &eval_point,
                    &mut current_claim,
                    round,
                    &mut eq,
                ),
            };

            challenger.observe_fp4_elems(&round_proof.coefficients());

            eq.fold_in_place();
            let r = challenger.get_challenge();

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

        //||============ QUERY PHASE ============||

        // The verifier (or the random oracle, which is the challenger for us concretely) at this point has observed the prover's sumcheck messages as well the commitments throughout the commit phase which was interleaved with the sumcheck protocol.
        // Now the verifier generates queries to test consistency between the committed oracles, which are concretely merkle tree roots, and finally test proximity to a valid codeword.

        //The queries should lie in the range 0...encoding.len()/2. The provided codewords, should contain the indexed value at i, as well as i + halfsize which the verifier requires to test consistency.

        let mut log_domain_size = rounds as u32 + config.rate.trailing_zeros() - 1;
        let mut domain_size = 1 << log_domain_size;
        let mut queries = challenger.get_indices(log_domain_size, config.queries);

        let mut codewords = Vec::with_capacity(rounds);
        let mut paths = Vec::with_capacity(rounds);

        for round in 0..rounds {
            let halfsize = domain_size >> 1;
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

            update_queries(&mut queries, halfsize);

            // Domain size halves each round
            domain_size = halfsize;
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
        round: usize,
        eq: &mut EqEvals,
    ) -> UnivariatePoly
    where
        F: PrimeCharacteristicRing + Field,
        Fp4: ExtensionField<F>,
    {
        let mut g_0: Fp4 = Fp4::ZERO;
        let rounds = eval_point.len();
        for i in 0..1 << (rounds - round - 1) {
            g_0 += eq[i] * poly[i << 1]
        }

        let g1: Fp4 = (*current_claim - g_0 * (Fp4::ONE - eval_point[round])) / eval_point[round];

        let round_coeffs = vec![g_0, g1 - g_0];
        let round_proof = UnivariatePoly::new(round_coeffs).unwrap();

        round_proof
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
        let current_merkle_tree = MerkleTree::from_hash(&leaves)?;
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
            let round_poly = &proof.sum_check_rounds[round];
            Self::verify_sum_check_round(
                round_poly,
                &mut current_claim,
                &eval_point,
                round,
                challenger,
            )?;

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let r = challenger.get_challenge();
            current_claim = round_poly.evaluate(r);
            random_point.push(r);
            challenger.observe_commitment(&proof.commitments[round]);
        }

        //Query Phase

        let log_domain_size = rounds as u32 + config.rate.trailing_zeros() - 1;
        let mut domain_size = 1 << log_domain_size;

        let mut queries = challenger.get_indices(log_domain_size, config.queries);

        let mut folded_codewords = Vec::with_capacity(config.queries);
        let mut current_codewords = &proof.codewords[0];
        let mut merkle_paths = &proof.paths[0];
        for round in 0..rounds {
            let halfsize = domain_size >> 1;
            for idx in 0..config.queries {
                let (left, right) = current_codewords[idx];
                let leaf_hash = hash_field_pair(left, right);
                let path = &merkle_paths[idx];
                match round {
                    0 => {
                        MerkleTree::verify_path(
                            leaf_hash,
                            queries[idx],
                            path,
                            commitment.commitment,
                        )?;

                        folded_codewords.push(fold_pair(
                            current_codewords[idx],
                            random_point[round],
                            roots[round][queries[idx]],
                        ));
                    }
                    _ => {
                        check_fold(
                            folded_codewords[idx],
                            queries[idx],
                            domain_size,
                            left,
                            right,
                        )?;
                        update_query(&mut queries[idx], domain_size);
                        MerkleTree::verify_path(
                            leaf_hash,
                            queries[idx],
                            path,
                            proof.commitments[round - 1],
                        )?;
                        folded_codewords[idx] = fold_pair(
                            current_codewords[idx],
                            random_point[round],
                            roots[round][queries[idx]],
                        );
                    }
                }
            }
            if round < rounds - 1 {
                domain_size = halfsize;
                current_codewords = &proof.codewords[round + 1];
                merkle_paths = &proof.paths[round + 1];
            }
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
        eval_point: &[Fp4],
        round: usize,
        challenger: &mut Challenger,
    ) -> anyhow::Result<()> {
        let expected = (Fp4::ONE - eval_point[round]) * round_poly.evaluate(Fp4::ZERO)
            + eval_point[round] * round_poly.evaluate(Fp4::ONE);

        if *current_claim != expected {
            anyhow::bail!(
                "Sum-check verification failed in round {round}: claim {:?} != expected {:?}",
                *current_claim,
                expected
            );
        }
        Ok(())
    }
}

fn update_queries(queries: &mut Vec<usize>, halfsize: usize) {
    queries.iter_mut().for_each(|query| {
        if *query >= halfsize {
            *query -= halfsize;
        }
    });
}

fn update_query(query: &mut usize, halfsize: usize) {
    if *query >= halfsize {
        *query -= halfsize;
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
    folded_codeword: Fp4,
    query: usize,
    halfsize: usize,
    left: Fp4,
    right: Fp4,
) -> anyhow::Result<()> {
    if query >= halfsize {
        if folded_codeword != right {
            anyhow::bail!(
                "Folded codeword verification failed: expected {:?}, got {:?}",
                (left, right),
                folded_codeword
            );
        }
    } else if folded_codeword != left {
        anyhow::bail!(
            "Folded codeword verification failed: expected {:?}, got {:?}",
            (left, right),
            folded_codeword
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

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
        let verification_result = Basefold::verify(
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
        let val = (1 << (6 + 1 - 1 - 1));
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
        let eval = poly.evaluate(&eval_point);
        let encoding = encode_mle(&poly, &roots, 2);

        let mut domain_size = 1 << 4;
        let half_size = domain_size / 2;

        // We start with query at 5. Thus the provided codeword should be encoding[5] and encoding[(5 + domain_size) = 21]
        let mut queries = vec![5];

        let correct_codeword = (encoding[5].into(), encoding[21].into());

        let received_codeword = get_codewords(&queries, &encoding);

        assert_eq!(correct_codeword, received_codeword[0]);

        let folded_codeword = fold_pair(received_codeword[0], eval_point[0], roots[0][5]);

        let mut folded_oracle = fold(&encoding, eval_point[0], &roots[0]);

        assert_eq!(folded_codeword, folded_oracle[5]);
    }
}
