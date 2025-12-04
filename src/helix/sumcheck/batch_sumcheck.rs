use std::time::Instant;

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    merkle_tree::{HashOutput, MerklePath, MerkleTree},
    pcs::{
        BaseFoldConfig, BasefoldCommitment, ProverData,
        prover::{commit_oracle, fold_encoding_and_polynomial, update_queries},
        utils::{
            create_hash_leaves_skip, decode_mle_ext, encode_mle, encode_mle_ext, encode_mle_rec,
            fold, get_codewords, get_merkle_paths,
        },
        verifier::{
            check_query_consistency, check_query_consistency_vec, fold_codewords,
            fold_codewords_vec, verify_paths, verify_paths_vec,
        },
    },
    polynomial::MLE,
    sparse::SparseMLE,
    helix::{
        sumcheck::{eval_at_infinity, eval_at_two, transpose_column_major},
        univariate::UnivariatePoly,
    },
};
use anyhow::{anyhow, bail};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_monty_31::dft::RecursiveDft;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use serde::Serialize;
use tracing::instrument;
#[derive(Serialize)]
pub struct BatchSumCheckProof {
    round_proofs: Vec<UnivariatePoly>,
    final_evals: [Fp4; 4],
    oracle_commitments: Vec<HashOutput>,
    z_eval: Fp4,
    early_codeword: Option<Vec<Fp4>>,
    first_round_codewords: Vec<Vec<Fp>>,
    codewords: Vec<Vec<Vec<Fp4>>>,
    paths: Vec<Vec<MerklePath>>,
}

impl BatchSumCheckProof {
    pub fn new(
        round_proofs: Vec<UnivariatePoly>,
        final_evals: [Fp4; 4],
        oracle_commitments: Vec<HashOutput>,
        z_eval: Fp4,
        early_codeword: Option<Vec<Fp4>>,
        first_round_codewords: Vec<Vec<Fp>>,
        codewords: Vec<Vec<Vec<Fp4>>>,
        paths: Vec<Vec<MerklePath>>,
    ) -> Self {
        Self {
            round_proofs,
            final_evals,
            oracle_commitments,
            z_eval,
            early_codeword,
            first_round_codewords,
            codewords,
            paths,
        }
    }

    #[instrument(name = "commit_skip", level = "debug", skip_all)]
    pub fn commit_skip(
        poly: &MLE<Fp>,
        dft: &RecursiveDft<Fp>,
        config: &BaseFoldConfig,
    ) -> anyhow::Result<(BasefoldCommitment, ProverData)> {
        if !poly.len().is_power_of_two() {
            anyhow::bail!("Polynomial size must be a power of 2, got {}", poly.len());
        }
        let encoding = encode_mle_rec(poly, dft, config.rate);
        // Create leaf hashes: H(E[2i], E[2i+1]) for each codeword pair
        let leaves = create_hash_leaves_skip(&encoding, &config);
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
    /// Generates a sum-check proof for f(x) = A(x)·B(x) - C(x).
    ///
    /// Computes A·z, B·z, C·z then runs the sum-check protocol: for each round,
    /// computes a univariate polynomial, gets a random challenge, and folds.
    #[instrument(target = "my_target", level = "debug", skip_all)]
    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z_transpose: &MLE<Fp>,
        z_transpose_commitment: &BasefoldCommitment,
        prover_data: &ProverData,
        roots: &[Vec<Fp>],
        config: &BaseFoldConfig,
        challenger: &mut Challenger,
    ) -> anyhow::Result<(Self, Vec<Fp4>)> {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let time = Instant::now();
        let (a, b, c) = (
            A.transpose_multiply_by_matrix(z_transpose.coeffs())
                .unwrap(),
            B.transpose_multiply_by_matrix(z_transpose.coeffs())
                .unwrap(),
            C.transpose_multiply_by_matrix(z_transpose.coeffs())
                .unwrap(),
        );
        println!("\n Matrix multiplication time {:?} \n", time.elapsed());
        let rounds = a.n_vars();

        assert!(rounds > 0, "MLEs need to be non empty");

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let z_eval = z_transpose.evaluate(&eq_point);

        let gamma = challenger.get_challenge();

        // gamma.c_0 + c_1
        let mut current_claim = gamma * z_eval;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // TODO: Handle the case where early stopping threshold is greater than the number of rounds
        let early_stopping_threshold = config.early_stopping_threshold;
        let fri_rounds = rounds.saturating_sub(early_stopping_threshold);
        let skip_rounds = config.round_skip;

        let mut a_fold = MLE::default();
        let mut b_fold = MLE::default();
        let mut c_fold = MLE::default();
        let mut z_fold = MLE::default();

        let mut state = BasefoldState::default();

        for round in 0..rounds {
            let round_proof = match round {
                0 => compute_round(
                    &a,
                    &b,
                    &c,
                    &z_transpose,
                    &eq,
                    &eq_point,
                    current_claim,
                    gamma,
                    round,
                ),
                _ => compute_round(
                    &a_fold,
                    &b_fold,
                    &c_fold,
                    &z_fold,
                    &eq,
                    &eq_point,
                    current_claim,
                    gamma,
                    round,
                ),
            };

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // If remaining rounds are more than the early stopping threshold continue FRI folding else stop.
            if round < fri_rounds {
                let current_encoding = match round {
                    0 => fold(&prover_data.encoding, round_challenge, &roots[round]),
                    _ => fold(
                        state.encodings.last().expect("Will be non-empty"),
                        round_challenge,
                        &roots[round],
                    ),
                };

                state.update(current_encoding, round >= skip_rounds)?;
                if round > skip_rounds {
                    challenger.observe_commitment(
                        state
                            .oracle_commitments
                            .last()
                            .expect("Will be non-empty after at least 1 fold"),
                    );
                }
            }

            (
                // Fold polynomials for next round
                a_fold, b_fold, c_fold, z_fold,
            ) = match round {
                0 => (
                    a.fold(round_challenge),
                    b.fold(round_challenge),
                    c.fold(round_challenge),
                    z_transpose.fold(round_challenge),
                ),
                _ => (
                    a_fold.fold(round_challenge),
                    b_fold.fold(round_challenge),
                    c_fold.fold(round_challenge),
                    z_fold.fold(round_challenge),
                ),
            };
            eq.fold_in_place();
            round_proofs.push(round_proof);
        }

        // Extract final evaluations A(r), B(r), C(r)
        let final_evals = [a_fold[0], b_fold[0], c_fold[0], z_fold[0]];

        //||============ QUERY PHASE ============||

        // The verifier (or the random oracle, which is the challenger for us concretely) at this point has observed the prover's sumcheck messages as well the commitments throughout the commit phase which was interleaved with the sumcheck protocol.
        // Now the verifier generates queries to test consistency between the committed oracles, which are concretely merkle tree roots, and finally test proximity to a valid codeword.

        // Query generation: sample random positions for consistency verification
        // Domain starts at size 2^(vars + rate_bits - 1), halves each folding round
        let log_domain_size =
            (rounds as u32) + config.rate.trailing_zeros() - (config.round_skip as u32) - 1;
        let mut domain_size = 1 << log_domain_size;
        let mut queries = challenger.get_indices(log_domain_size, config.queries);

        let early_codeword = match early_stopping_threshold {
            0 => None,
            _ => Some(
                state
                    .encodings
                    .last()
                    .expect("Will be more than 1 round")
                    .clone(),
            ),
        };

        let mut codewords = Vec::with_capacity(rounds);
        let mut first_round_codewords = Vec::with_capacity(rounds);
        let mut paths = Vec::with_capacity(rounds);

        let BasefoldState {
            encodings,
            merkle_trees,
            ..
        } = state;
        eprintln!(" difference in rounds {:?}", fri_rounds - skip_rounds);
        eprintln!(" merkle trees {:?}", merkle_trees.len());
        eprintln!(" encodings {:?}", encodings.len());
        for round in 0..fri_rounds {
            // Only add merkle paths to the proof if it is the first round, or the round after skipped rounds.

            if round == 0 {
                first_round_codewords =
                    get_first_round_codewords(&queries, &prover_data.encoding, skip_rounds);
                paths.push(get_merkle_paths(&queries, &prover_data.merkle_tree)?);
            } else if round >= skip_rounds {
                // Update query indices for next round using bitwise masking
                update_queries(&mut queries, domain_size >> 1);

                // Domain size halves each folding round
                domain_size >>= 1;

                eprintln!(" merkle trees index {:?}", round - skip_rounds);
                codewords.push(get_codewords_vec(&queries, &encodings[round - 1]));
                paths.push(get_merkle_paths(
                    &queries,
                    &merkle_trees[round - skip_rounds],
                )?);
            }
        }
        Ok((
            BatchSumCheckProof::new(
                round_proofs,
                final_evals,
                state.oracle_commitments,
                z_eval,
                early_codeword,
                first_round_codewords,
                codewords,
                paths,
            ),
            round_challenges,
        ))
    }

    /// Verifies the sum-check proof. Panics if verification fails.
    pub fn verify(
        &self,
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        commitment: BasefoldCommitment,
        roots: &[Vec<Fp>],
        challenger: &mut Challenger,
        config: &BaseFoldConfig,
    ) -> anyhow::Result<(Vec<Fp4>, [Fp4; 4])> {
        let rounds = self.round_proofs.len();

        let eq_point = challenger.get_challenges(rounds);

        let gamma = challenger.get_challenge();
        let mut current_claim = gamma * self.z_eval;
        let mut round_challenges = Vec::new();
        let skip_rounds = config.round_skip;
        let fri_rounds = rounds.saturating_sub(config.early_stopping_threshold);
        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            let round_eval = (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                + eq_point[round] * round_poly.evaluate(Fp4::ONE);
            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            if current_claim != round_eval {
                return Err(anyhow!(
                    "OuterSumcheck round verification failed in round {round}, expected {current_claim} got {round_eval}"
                ));
            }

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);

            if round > skip_rounds && round < fri_rounds {
                challenger.observe_commitment(&self.oracle_commitments[round - skip_rounds]);
            }
        }

        //Query Phase

        // The queries are always in the range 0..encoding.len()/2
        let log_query_range =
            (rounds as u32) + config.rate.trailing_zeros() - (config.round_skip as u32) - 1;
        let mut query_range = 1 << log_query_range;
        let mut queries = challenger.get_indices(log_query_range, config.queries);
        let mut folded_codewords = vec![Fp4::ZERO; config.queries];

        eprintln!("query range {query_range}");
        for round in 0..fri_rounds {
            eprintln!("Round {round}");
            let halfsize = query_range >> 1;
            let oracle_commitment = match round {
                0 => commitment.commitment,
                _ => self.oracle_commitments[round - 1],
            };

            let paths = &self.paths[round];
            match round {
                0 => {
                    let codewords = &self.first_round_codewords;
                    verify_paths_vec(codewords, paths, &queries, oracle_commitment)?;

                    fold_codewords_vec(
                        &mut folded_codewords,
                        codewords,
                        &queries,
                        round_challenges[round],
                        &roots[round],
                    );
                }
                _ => {
                    let codewords = &self.codewords[round - 1];
                    check_query_consistency_vec(
                        &mut queries,
                        &folded_codewords,
                        codewords,
                        query_range,
                        round,
                    )?;

                    verify_paths_vec(codewords, paths, &queries, oracle_commitment)?;

                    fold_codewords_vec(
                        &mut folded_codewords,
                        codewords,
                        &queries,
                        round_challenges[round],
                        &roots[round],
                    );
                }
            }

            query_range = halfsize;
        }

        let final_fold;

        let mut a_claim = Fp4::ZERO;
        let mut b_claim = Fp4::ZERO;
        let mut c_claim = Fp4::ZERO;
        if let Some(mut early_code) = self.early_codeword.clone() {
            // TODO: Better error messaging.
            for (&query, &folded_codeword) in queries.iter().zip(&folded_codewords) {
                if early_code[query] != folded_codeword {
                    bail!(
                        "Folded codewords accross queries not consistent with codeword provided."
                    );
                }
            }

            let decoding = decode_mle_ext(early_code.clone(), config.rate);
            let mut az_fold = A.transpose_multiply_by_matrix(&decoding)?;
            let mut bz_fold = B.transpose_multiply_by_matrix(&decoding)?;
            let mut cz_fold = C.transpose_multiply_by_matrix(&decoding)?;

            for round in fri_rounds..rounds {
                early_code = fold(&early_code, round_challenges[round], &roots[round]);
                az_fold = az_fold.fold(round_challenges[round]);
                bz_fold = bz_fold.fold(round_challenges[round]);
                cz_fold = cz_fold.fold(round_challenges[round]);
            }

            final_fold = early_code[0];
            a_claim = az_fold[0];
            b_claim = bz_fold[0];
            c_claim = cz_fold[0];
        } else {
            final_fold = folded_codewords[0];
            for &final_val in &folded_codewords[1..] {
                if final_val != final_fold {
                    bail!("Final folds not consistent accross queries.");
                }
            }
        }

        let [a, b, c, z] = self.final_evals;

        if final_fold != z {
            anyhow::bail!(
                "Final claim verification failed: {:?} != {:?}",
                folded_codewords[0],
                current_claim
            );
        }

        // Final check: A(r)·B(r) - C(r) + \gamma.z(r) = final_claim
        if current_claim != a * b - c + gamma * z {
            return Err(anyhow!("Final Check Failed in SumCheck"));
        }

        if self.early_codeword.is_some() {
            if a_claim != a {
                return Err(anyhow!("a fold did not match sumcheck transcript"));
            }
            if b_claim != b {
                return Err(anyhow!("b fold did not match sumcheck transcript"));
            }
            if c_claim != c {
                return Err(anyhow!("c fold did not match sumcheck transcript"));
            }
        }
        Ok((round_challenges, self.final_evals))
    }
}

/// Retrieves codeword pairs from an encoding at query positions.
/// Uses slice splitting to eliminate manual offset calculation.
pub fn get_codewords_vec<F: Copy>(queries: &[usize], encoding: &[F]) -> Vec<Vec<F>> {
    let halfsize = encoding.len() >> 1;
    let (left, right) = encoding.split_at(halfsize);

    queries
        .iter()
        .copied()
        .map(|i| vec![left[i], right[i]])
        .collect()
}

/// Retrieves codeword pairs from an encoding at query positions.
/// Uses slice splitting to eliminate manual offset calculation.
pub fn get_first_round_codewords<F: Copy>(
    queries: &[usize],
    encoding: &[F],
    skip: usize,
) -> Vec<Vec<F>> {
    let partition_size = encoding.len() >> (skip + 1);
    let partitions = 1 << (skip + 1);

    println!("Partition size {partition_size}");
    println!("Partitions {partitions}");

    queries
        .iter()
        .copied()
        .map(|i| {
            let mut vec = Vec::new();

            for p in 0..partitions {
                vec.push(encoding[i + p * partition_size]);
            }
            vec
        })
        .collect()
}

pub struct BasefoldState {
    pub oracle_commitments: Vec<[u8; 32]>,
    pub merkle_trees: Vec<MerkleTree>,
    pub encodings: Vec<Vec<Fp4>>,
}

impl BasefoldState {
    pub fn update(&mut self, current_encoding: Vec<Fp4>, commit: bool) -> anyhow::Result<()> {
        if commit {
            let (oracle_commitment, merkle_tree) = commit_oracle(&current_encoding)?;
            self.oracle_commitments.push(oracle_commitment);
            self.merkle_trees.push(merkle_tree);
        }
        self.encodings.push(current_encoding);
        Ok(())
    }
}

impl Default for BasefoldState {
    fn default() -> Self {
        Self {
            oracle_commitments: Default::default(),
            merkle_trees: Default::default(),
            encodings: Default::default(),
        }
    }
}

/// Computes the univariate polynomial for sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w) - c(X,w)].
#[tracing::instrument(level = "debug", skip_all)]
pub fn compute_round<F>(
    a: &MLE<F>,
    b: &MLE<F>,
    c: &MLE<F>,
    z_transpose: &MLE<F>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    gamma: Fp4,
    round: usize,
) -> UnivariatePoly
where
    F: Field,
    Fp4: ExtensionField<F>,
{
    let eq_slice = &eq.coeffs()[..];
    let a_slice = &a.coeffs()[..];
    let b_slice = &b.coeffs()[..];
    let c_slice = &c.coeffs()[..];
    let z_slice: &[F] = &z_transpose.coeffs()[..];

    let (coeff_0, coeff_2) = (
        a_slice.par_chunks_exact(2),
        b_slice.par_chunks_exact(2),
        c_slice.par_chunks_exact(2),
        z_slice.par_chunks_exact(2),
        eq_slice,
    )
        .into_par_iter()
        .map(|(a, b, c, z, &eq)| {
            let val_0 = eq * (gamma * z[0] + a[0] * b[0] - c[0]);

            let val_2 = eq * (eval_at_infinity(a[0], a[1]) * eval_at_infinity(b[0], b[1]));
            (val_0, val_2)
        })
        .reduce(
            || (Fp4::ZERO, Fp4::ZERO),
            |(acc_0, acc_2), (g_0, g_2)| (acc_0 + g_0, acc_2 + g_2),
        );

    let mut round_coeffs = vec![coeff_0, Fp4::ZERO, coeff_2];

    // g(1): derived from sum-check constraint
    round_coeffs[1] =
        (current_claim - round_coeffs[0] * (Fp4::ONE - eq_point[round])) / eq_point[round];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

fn print_fp4(vector: &[Fp4]) {
    print!("\n [");

    for val in vector {
        print!("{val}, ");
    }

    print!("] \n");
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp, challenger::Challenger, pcs::Basefold, polynomial::MLE};
    use itertools::multizip;
    use p3_field::BasedVectorSpace;
    use p3_monty_31::dft::RecursiveDft;
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
    use std::collections::HashMap;

    #[test]
    fn batch_sum_check_proof_end_to_end() -> anyhow::Result<()> {
        const ROWS: usize = 1 << 6;
        const COLS: usize = ROWS;
        const WITNESS_COLS: usize = 1 << 3;
        let mut rng = StdRng::seed_from_u64(0);

        let mut A = HashMap::<(usize, usize), Fp>::new();
        let mut B = HashMap::<(usize, usize), Fp>::new();
        let mut C = HashMap::<(usize, usize), Fp>::new();

        let witness_entry = Fp::new(rng.r#gen::<u32>());
        let z = MLE::new(vec![witness_entry; COLS * WITNESS_COLS]);

        let z_const = witness_entry;

        for row in 0..ROWS {
            let a_val = Fp::new(rng.r#gen::<u32>());
            let b_val = Fp::new(rng.r#gen::<u32>());

            let col_a = rng.gen_range(0..COLS);
            let col_b = rng.gen_range(0..COLS);

            A.insert((row, col_a), a_val);
            B.insert((row, col_b), b_val);
            C.insert((row, row), a_val * b_val * z_const);
        }

        let A = SparseMLE::new(A)?;
        let B = SparseMLE::new(B)?;
        let C = SparseMLE::new(C)?;

        let a_eval = A.multiply_by_mle(&z)?;
        let b_eval = B.multiply_by_mle(&z)?;
        let c_eval = C.multiply_by_mle(&z)?;

        for (&a_i, &b_i, &c_i) in multizip((a_eval.coeffs(), b_eval.coeffs(), c_eval.coeffs())) {
            assert_eq!(c_i, a_i * b_i, "R1CS instance not satisfied");
        }

        let config = BaseFoldConfig::new().with_queries(4);
        let roots = Fp::roots_of_unity_table(1 << (z.n_vars() + 1));
        let (_, cols) = A.dimensions();
        assert_eq!(cols, B.dimensions().1);
        assert_eq!(cols, C.dimensions().1);

        assert!(
            z.coeffs().len() % cols == 0,
            "z must align with column dimension"
        );
        let matrix_rows = z.coeffs().len() / cols;
        let z_transposed = MLE::new(transpose_column_major(z.coeffs(), cols, matrix_rows));
        let (commitment, prover_data) = Basefold::commit(&z_transposed, &roots, &config)?;

        let mut prover_challenger = Challenger::new();
        let (proof, round_challenges) = BatchSumCheckProof::prove(
            &A,
            &B,
            &C,
            &z_transposed,
            &commitment,
            &prover_data,
            &roots,
            &config,
            &mut prover_challenger,
        )?;

        assert_eq!(round_challenges.len(), z.n_vars());

        let mut verifier_challenger = Challenger::new();
        let (verified_challenges, final_evals) = proof.verify(
            &A,
            &B,
            &C,
            commitment,
            &roots,
            &mut verifier_challenger,
            &config,
        )?;

        assert_eq!(verified_challenges, round_challenges);
        Ok(())
    }

    #[test]
    fn batch_sum_check_proof_early_stopping() -> anyhow::Result<()> {
        const ROWS: usize = 1 << 3;
        const COLS: usize = ROWS;
        const WITNESS_COLS: usize = 1 << 3;
        let mut rng = StdRng::seed_from_u64(0);

        let mut A = HashMap::<(usize, usize), Fp>::new();
        let mut B = HashMap::<(usize, usize), Fp>::new();
        let mut C = HashMap::<(usize, usize), Fp>::new();

        let witness_entry = Fp::new(rng.r#gen::<u32>());
        let z = MLE::new(vec![witness_entry; COLS * WITNESS_COLS]);

        let z_const = witness_entry;

        for row in 0..ROWS {
            let a_val = Fp::new(rng.r#gen::<u32>());
            let b_val = Fp::new(rng.r#gen::<u32>());

            let col_a = rng.gen_range(0..COLS);
            let col_b = rng.gen_range(0..COLS);

            A.insert((row, col_a), a_val);
            B.insert((row, col_b), b_val);

            C.insert((row, row), a_val * b_val * z_const);
        }

        let A = SparseMLE::new(A)?;
        let B = SparseMLE::new(B)?;
        let C = SparseMLE::new(C)?;

        let a_eval = A.multiply_by_mle(&z)?;
        let b_eval = B.multiply_by_mle(&z)?;
        let c_eval = C.multiply_by_mle(&z)?;

        for (&a_i, &b_i, &c_i) in multizip((a_eval.coeffs(), b_eval.coeffs(), c_eval.coeffs())) {
            assert_eq!(c_i, a_i * b_i, "R1CS instance not satisfied");
        }

        let config = BaseFoldConfig::new().with_queries(4).with_early_stopping(3);
        let roots = Fp::roots_of_unity_table(1 << (z.n_vars() + 1));
        let (_, cols) = A.dimensions();
        assert_eq!(cols, B.dimensions().1);
        assert_eq!(cols, C.dimensions().1);

        assert!(
            z.coeffs().len() % cols == 0,
            "z must align with column dimension"
        );
        let matrix_rows = z.coeffs().len() / cols;
        let z_transposed = MLE::new(transpose_column_major(z.coeffs(), cols, matrix_rows));
        let (commitment, prover_data) = Basefold::commit(&z_transposed, &roots, &config)?;

        let mut prover_challenger = Challenger::new();
        let (proof, round_challenges) = BatchSumCheckProof::prove(
            &A,
            &B,
            &C,
            &z_transposed,
            &commitment,
            &prover_data,
            &roots,
            &config,
            &mut prover_challenger,
        )?;

        assert_eq!(round_challenges.len(), z.n_vars());

        let mut verifier_challenger = Challenger::new();
        let (verified_challenges, final_evals) = proof.verify(
            &A,
            &B,
            &C,
            commitment,
            &roots,
            &mut verifier_challenger,
            &config,
        )?;

        assert_eq!(verified_challenges, round_challenges);
        Ok(())
    }
}
