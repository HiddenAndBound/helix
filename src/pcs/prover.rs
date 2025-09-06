//! BaseFold Prover Implementation
//!
//! This module contains the prover-side logic for the BaseFold polynomial commitment scheme,
//! including commitment generation, evaluation proof construction, and all prover helper functions.

use anyhow::Result;
use p3_field::{ ExtensionField, Field, PrimeCharacteristicRing };

use crate::pcs::utils::{
    Encoding,
    create_hash_leaves_from_pairs,
    create_hash_leaves_from_pairs_ref,
    encode_mle,
    fold,
    get_codewords,
    get_merkle_paths,
};
use crate::{
    Fp,
    Fp4,
    challenger::Challenger,
    eq::EqEvals,
    merkle_tree::{ MerkleTree },
    polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};

use super::{ Basefold, BaseFoldConfig, BasefoldCommitment, ProverData, EvalProof };

/// Prover-specific state maintained during evaluation proof generation.
pub struct BasefoldProverState {
    pub current_claim: Fp4,
    pub random_point: Vec<Fp4>,
    pub sum_check_rounds: Vec<UnivariatePoly>,
    pub oracle_commitments: Vec<[u8; 32]>,
    pub merkle_trees: Vec<MerkleTree>,
    pub encodings: Vec<Vec<Fp4>>,
    pub current_poly: MLE<Fp4>,
}

impl BasefoldProverState {
    pub fn update(
        &mut self,
        current_claim: Fp4,
        random_point: Fp4,
        sum_check_round: UnivariatePoly,
        current_encoding: Vec<Fp4>,
        current_poly: MLE<Fp4>
    ) -> anyhow::Result<()> {
        let (oracle_commitment, merkle_tree) = commit_oracle(&current_encoding)?;

        self.current_claim = current_claim;
        self.random_point.push(random_point);
        self.sum_check_rounds.push(sum_check_round);
        self.oracle_commitments.push(oracle_commitment);
        self.merkle_trees.push(merkle_tree);
        self.encodings.push(current_encoding);
        self.current_poly = current_poly;

        Ok(())
    }
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
        config: &BaseFoldConfig
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

        // Split Reed-Solomon encoding into pairs for Merkle tree construction
        let (left, right) = encoding.split_at(encoding.len() / 2);

        // Create leaf hashes: H(E[2i], E[2i+1]) for each codeword pair
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
        config: &BaseFoldConfig
    ) -> anyhow::Result<EvalProof> {
        if poly.n_vars() != eval_point.len() {
            anyhow::bail!(
                "Evaluation point dimension {} doesn't match polynomial variables {}",
                eval_point.len(),
                poly.n_vars()
            );
        }

        let mut state = initialize_evaluation_proof_context(evaluation, poly.n_vars());

        let mut eq = EqEvals::gen_from_point(&eval_point[1..]);
        let rounds = poly.n_vars();

        //Commit phase
        for round in 0..rounds {
            let round_proof = match round {
                0 =>
                    process_sum_check_round(
                        poly,
                        &eval_point,
                        &state.current_claim,
                        round,
                        &mut eq
                    ),
                _ =>
                    process_sum_check_round(
                        &state.current_poly,
                        &eval_point,
                        &state.current_claim,
                        round,
                        &mut eq
                    ),
            };

            challenger.observe_fp4_elems(&round_proof.coefficients());

            eq.fold_in_place();
            let r = challenger.get_challenge();

            let (current_encoding, current_poly_folded) = fold_encoding_and_polynomial(
                round,
                &prover_data.encoding,
                &state.encodings,
                r,
                roots,
                poly,
                &state.current_poly
            );
            state.update(
                round_proof.evaluate(r),
                r,
                round_proof,
                current_encoding,
                current_poly_folded
            )?;
            challenger.observe_commitment(
                state.oracle_commitments.last().expect("Will be non-empty after at least 1 fold")
            );
        }

        //||============ QUERY PHASE ============||

        // The verifier (or the random oracle, which is the challenger for us concretely) at this point has observed the prover's sumcheck messages as well the commitments throughout the commit phase which was interleaved with the sumcheck protocol.
        // Now the verifier generates queries to test consistency between the committed oracles, which are concretely merkle tree roots, and finally test proximity to a valid codeword.

        // Query generation: sample random positions for consistency verification
        // Domain starts at size 2^(vars + rate_bits - 1), halves each folding round
        let log_domain_size = (rounds as u32) + config.rate.trailing_zeros() - 1;
        let mut domain_size = 1 << log_domain_size;
        let mut queries = challenger.get_indices(log_domain_size, config.queries);

        let mut codewords = Vec::with_capacity(rounds);
        let mut paths = Vec::with_capacity(rounds);
        let BasefoldProverState { encodings, merkle_trees, .. } = state;
        for round in 0..rounds {
            let halfsize = domain_size >> 1;
            let (round_codewords, round_paths): (Vec<(Fp4, Fp4)>, Vec<_>) = match round {
                0 =>
                    (
                        get_codewords(&queries, &prover_data.encoding),
                        get_merkle_paths(&queries, &prover_data.merkle_tree),
                    ),

                _ =>
                    (
                        get_codewords(&queries, &encodings[round - 1]),
                        get_merkle_paths(&queries, &merkle_trees[round - 1]),
                    ),
            };

            codewords.push(round_codewords);
            paths.push(round_paths);

            // Update query indices for next round using bitwise masking
            update_queries(&mut queries, halfsize);

            // Domain size halves each folding round
            domain_size = halfsize;
        }

        Ok(EvalProof {
            codewords,
            paths,
            sum_check_rounds: state.sum_check_rounds,
            commitments: state.oracle_commitments,
        })
    }
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
pub fn process_sum_check_round<F>(
    poly: &MLE<F>,
    eval_point: &[Fp4],
    current_claim: &Fp4,
    round: usize,
    eq: &mut EqEvals
)
    -> UnivariatePoly
    where F: PrimeCharacteristicRing + Field, Fp4: ExtensionField<F>
{
    let mut g_0: Fp4 = Fp4::ZERO;
    let rounds = eval_point.len();

    // Size of the boolean hypercube at this round of the sumcheck
    let size = 1 << (rounds - round - 1);

    // Bounds are safe: iterations = 2^(rounds-round-1), poly.len() = 2^rounds
    // Max poly index = (iterations-1)*2 < 2^rounds, eq.len() ≥ iterations
    let poly = &poly.coeffs()[0..size * 2];
    let eq = &eq.coeffs()[0..size];
    debug_assert!(
        poly.len() >= size * 2,
        "Mathematical bounds violated: poly.len()={}, required={}",
        poly.len(),
        size * 2
    );
    debug_assert!(
        eq.len() >= size,
        "Mathematical bounds violated: eq.len()={}, required={}",
        eq.len(),
        size
    );

    // Use unsafe indexing to eliminate bounds checking overhead
    unsafe {
        for i in 0..size {
            // SAFETY: Bounds verified above, i < iterations ≤ eq.len() and i*2 < poly.len()
            // Sum-check accumulation: g₀ = Σⱼ eq[j] * P[2j] (sum over x_i = 0)
            g_0 += *eq.get_unchecked(i) * *poly.get_unchecked(i << 1);
        }
    }

    // Derive g₁ from sum-check constraint: g₀ + g₁ = current_claim
    // Rearranging: g₁ = (current_claim - g₀*(1-r)) / r where r = eval_point[round]
    let g1: Fp4 = (*current_claim - g_0 * (Fp4::ONE - eval_point[round])) / eval_point[round];

    let round_coeffs = vec![g_0, g1 - g_0];
    let round_proof = UnivariatePoly::new(round_coeffs).unwrap();

    round_proof
}

/// Performs dual folding of both encoding and polynomial using Fiat-Shamir challenge.
///
/// This is a core operation in BaseFold where both the Reed-Solomon encoding
/// and the polynomial are folded simultaneously to maintain consistency.
pub fn fold_encoding_and_polynomial(
    round: usize,
    initial_encoding: &Encoding,
    encodings: &[Vec<Fp4>],
    r: Fp4,
    roots: &[Vec<Fp>],
    initial_poly: &MLE<Fp>,
    current_poly: &MLE<Fp4>
) -> (Vec<Fp4>, MLE<Fp4>) {
    let current_encoding = match round {
        0 => fold(initial_encoding, r, &roots[round]),
        _ => fold(encodings.last().expect("Will be non-empty"), r, &roots[round]),
    };

    let current_poly_folded = match round {
        0 => initial_poly.fold_in_place(r),
        _ => current_poly.fold_in_place(r),
    };
    (current_encoding, current_poly_folded)
}

/// Initializes data structures for the evaluation proof generation.
///
/// Pre-allocates vectors with known capacity to avoid reallocations during
/// the intensive folding rounds of the sum-check protocol.
pub fn initialize_evaluation_proof_context(evaluation: Fp4, rounds: usize) -> BasefoldProverState {
    let current_claim = evaluation;
    let random_point = Vec::with_capacity(rounds);
    let sum_check_rounds = Vec::with_capacity(rounds);
    let oracle_commitments = Vec::with_capacity(rounds);
    let merkle_trees = Vec::with_capacity(rounds);
    let encodings = Vec::with_capacity(rounds);
    let current_poly = MLE::default();

    BasefoldProverState {
        current_claim,
        random_point,
        sum_check_rounds,
        oracle_commitments,
        merkle_trees,
        encodings,
        current_poly,
    }
}

/// Updates Merkle trees and commitments after each folding round.
///
/// Builds a new Merkle tree over the folded encoding pairs and stores
/// the commitment for verifier observation in the Fiat-Shamir transcript.
pub fn commit_oracle(current_encoding: &[Fp4]) -> Result<([u8; 32], MerkleTree), anyhow::Error> {
    let (left, right) = current_encoding.split_at(current_encoding.len() / 2);

    let leaves: Vec<[u8; 32]> = create_hash_leaves_from_pairs_ref(left, right);
    let current_merkle_tree = MerkleTree::from_hash(&leaves)?;
    let current_commitment = current_merkle_tree.root();
    Ok((current_commitment, current_merkle_tree))
}

/// Updates all query indices for the next folding round using bitwise optimization.
///
/// Since domain size halves each round and is always a power of 2, we use bitwise
/// masking instead of modular arithmetic for better performance.
///
/// Mathematical equivalence: `query %= halfsize` == `query &= (halfsize - 1)`
pub fn update_queries(queries: &mut Vec<usize>, halfsize: usize) {
    queries.iter_mut().for_each(|query| {
        update_query(query, halfsize);
    });
}

/// Updates a single query using bitwise masking optimization.
/// For power-of-2 halfsize: query &= (halfsize - 1) is equivalent to
/// the conditional subtraction but faster as it's a single bitwise operation.
pub fn update_query(query: &mut usize, halfsize: usize) {
    debug_assert!(halfsize.is_power_of_two(), "halfsize must be a power of 2");
    // Bitwise optimization: query &= (halfsize-1) equivalent to query %= halfsize
    // but faster for power-of-2 values
    *query &= halfsize - 1;
}
