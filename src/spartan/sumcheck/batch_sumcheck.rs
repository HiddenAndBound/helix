use p3_field::{ ExtensionField, Field, PrimeCharacteristicRing };
use rayon::{ iter::{ IntoParallelIterator, ParallelIterator }, slice::ParallelSlice };
use anyhow::anyhow;
use crate::{
    challenger::Challenger,
    eq::EqEvals,
    merkle_tree::{ MerklePath, MerkleTree },
    pcs::{
        prover::{ commit_oracle, fold_encoding_and_polynomial, update_queries },
        utils::{ fold, get_codewords, get_merkle_paths },
        verifier::{ check_query_consistency, fold_codewords, verify_paths },
        BaseFoldConfig,
        BasefoldCommitment,
        ProverData,
    },
    polynomial::MLE,
    sparse::SparseMLE,
    spartan::{ sumcheck::{ eval_at_infinity, eval_at_two }, univariate::UnivariatePoly },
    Fp,
    Fp4,
};

pub struct BatchSumCheckProof {
    round_proofs: Vec<UnivariatePoly>,
    final_evals: [Fp4; 4],
    oracle_commitments: Vec<[u8; 32]>,
    z_eval: Fp4,
    codewords: Vec<Vec<(Fp4, Fp4)>>,
    paths: Vec<Vec<MerklePath>>,
}

impl BatchSumCheckProof {
    pub fn new(
        round_proofs: Vec<UnivariatePoly>,
        final_evals: [Fp4; 4],
        oracle_commitments: Vec<[u8; 32]>,
        z_eval: Fp4,
        codewords: Vec<Vec<(Fp4, Fp4)>>,
        paths: Vec<Vec<MerklePath>>
    ) -> Self {
        Self {
            round_proofs,
            final_evals,
            oracle_commitments,
            z_eval,
            codewords,
            paths,
        }
    }

    /// Generates a sum-check proof for f(x) = A(x)·B(x) - C(x).
    ///
    /// Computes A·z, B·z, C·z then runs the sum-check protocol: for each round,
    /// computes a univariate polynomial, gets a random challenge, and folds.
    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z: &MLE<Fp>,
        z_commitment: &BasefoldCommitment,
        prover_data: &ProverData,
        roots: &[Vec<Fp>],
        config: &BaseFoldConfig,
        challenger: &mut Challenger
    ) -> anyhow::Result<(Self, Vec<Fp4>)> {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let (a, b, c) = (
            A.multiply_by_matrix(z.coeffs()).unwrap(),
            B.multiply_by_matrix(z.coeffs()).unwrap(),
            C.multiply_by_matrix(z.coeffs()).unwrap(),
        );
        let rounds = a.n_vars();

        assert!(rounds > 0, "MLEs need to be non empty");

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let z_eval = z.evaluate(&eq_point);

        let gamma = challenger.get_challenge();

        // gamma.c_0 + c_1
        let mut current_claim = gamma * z_eval;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        let mut a_fold = MLE::default();
        let mut b_fold = MLE::default();
        let mut c_fold = MLE::default();
        let mut z_fold = MLE::default();

        let mut state = BasefoldState::default();

        // Process remaining rounds (1 to n-1)
        for round in 0..rounds {
            let round_proof = match round {
                0 => compute_round(&a, &b, &c, &z, &eq, &eq_point, current_claim, gamma, round),
                _ =>
                    compute_round(
                        &a_fold,
                        &b_fold,
                        &c_fold,
                        &z_fold,
                        &eq,
                        &eq_point,
                        current_claim,
                        gamma,
                        round
                    ),
            };

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            let (current_encoding, current_poly_folded) = fold_encoding_and_polynomial(
                round,
                &prover_data.encoding,
                &state.encodings,
                round_challenge,
                roots,
                z,
                &z_fold
            );

            state.update(current_encoding)?;
            challenger.observe_commitment(
                state.oracle_commitments.last().expect("Will be non-empty after at least 1 fold")
            );

            (
                // Fold polynomials for next round
                a_fold,
                b_fold,
                c_fold,
                z_fold,
            ) = match round {
                0 =>
                    (
                        a.fold_in_place(round_challenge),
                        b.fold_in_place(round_challenge),
                        c.fold_in_place(round_challenge),
                        current_poly_folded,
                    ),
                _ =>
                    (
                        a_fold.fold_in_place(round_challenge),
                        b_fold.fold_in_place(round_challenge),
                        c_fold.fold_in_place(round_challenge),
                        current_poly_folded,
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
        let log_domain_size = (rounds as u32) + config.rate.trailing_zeros() - 1;
        let mut domain_size = 1 << log_domain_size;
        let mut queries = challenger.get_indices(log_domain_size, config.queries);

        let mut codewords = Vec::with_capacity(rounds);
        let mut paths = Vec::with_capacity(rounds);
        let BasefoldState { encodings, merkle_trees, .. } = state;
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
        Ok((
            BatchSumCheckProof::new(
                round_proofs,
                final_evals,
                state.oracle_commitments,
                z_eval,
                codewords,
                paths
            ),
            round_challenges,
        ))
    }

    /// Verifies the sum-check proof. Panics if verification fails.
    pub fn verify(
        &self,
        commitment: BasefoldCommitment,
        roots: &[Vec<Fp>],
        challenger: &mut Challenger,
        config: &BaseFoldConfig
    ) -> anyhow::Result<(Vec<Fp4>, [Fp4; 4])> {
        let rounds = self.round_proofs.len();

        let eq_point = challenger.get_challenges(rounds);

        let gamma = challenger.get_challenge();
        let mut current_claim = gamma * self.z_eval;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            let round_eval =
                (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO) +
                eq_point[round] * round_poly.evaluate(Fp4::ONE);
            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            if current_claim != round_eval {
                return Err(
                    anyhow!(
                        "OuterSumcheck round verification failed in round {round}, expected {current_claim} got {round_eval}"
                    )
                );
            }

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
            challenger.observe_commitment(&self.oracle_commitments[round]);
        }

        //Query Phase

        // The queries are always in the range 0..encoding.len()/2
        let log_query_range = (rounds as u32) + config.rate.trailing_zeros() - 1;
        let mut query_range = 1 << log_query_range;
        let mut queries = challenger.get_indices(log_query_range, config.queries);
        let mut folded_codewords = vec![Fp4::ZERO; config.queries];

        for round in 0..rounds {
            let halfsize = query_range >> 1;
            let oracle_commitment = match round {
                0 => commitment.commitment,
                _ => self.oracle_commitments[round - 1],
            };

            let (codewords, paths) = (&self.codewords[round], &self.paths[round]);

            check_query_consistency(
                &mut queries,
                &folded_codewords,
                &codewords,
                query_range,
                round
            )?;

            verify_paths(codewords, paths, &queries, oracle_commitment)?;

            fold_codewords(
                &mut folded_codewords,
                codewords,
                &queries,
                round_challenges[round],
                &roots[round]
            );

            query_range = halfsize;
        }

        let [a, b, c, z] = self.final_evals;

        if folded_codewords[0] != z {
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

        Ok((round_challenges, self.final_evals))
    }
}

pub struct BasefoldState {
    pub oracle_commitments: Vec<[u8; 32]>,
    pub merkle_trees: Vec<MerkleTree>,
    pub encodings: Vec<Vec<Fp4>>,
}

impl BasefoldState {
    pub fn update(&mut self, current_encoding: Vec<Fp4>) -> anyhow::Result<()> {
        let (oracle_commitment, merkle_tree) = commit_oracle(&current_encoding)?;
        self.oracle_commitments.push(oracle_commitment);
        self.merkle_trees.push(merkle_tree);
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
pub fn compute_round<F>(
    a: &MLE<F>,
    b: &MLE<F>,
    c: &MLE<F>,
    z: &MLE<F>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    gamma: Fp4,
    round: usize
)
    -> UnivariatePoly
    where F: Field, Fp4: ExtensionField<F>
{
    let eq_slice = &eq.coeffs()[..];
    let a_slice = &a.coeffs()[..];
    let b_slice = &b.coeffs()[..];
    let c_slice = &c.coeffs()[..];
    let z_slice: &[F] = &z.coeffs()[..];

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
            |(acc_0, acc_2), (g_0, g_2)| (acc_0 + g_0, acc_2 + g_2)
        );

    let mut round_coeffs = vec![coeff_0, Fp4::ZERO, coeff_2];

    // g(1): derived from sum-check constraint
    round_coeffs[1] =
        (current_claim - round_coeffs[0] * (Fp4::ONE - eq_point[round])) / eq_point[round];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ challenger::Challenger, pcs::Basefold, polynomial::MLE, Fp };
    use itertools::multizip;
    use rand::{ rngs::StdRng, Rng, SeedableRng };
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
        let (commitment, prover_data) = Basefold::commit(&z, &roots, &config)?;

        let mut prover_challenger = Challenger::new();
        let (proof, round_challenges) = BatchSumCheckProof::prove(
            &A,
            &B,
            &C,
            &z,
            &commitment,
            &prover_data,
            &roots,
            &config,
            &mut prover_challenger
        )?;

        assert_eq!(round_challenges.len(), z.n_vars());

        let mut verifier_challenger = Challenger::new();
        let (verified_challenges, final_evals) = proof.verify(
            commitment,
            &roots,
            &mut verifier_challenger,
            &config
        )?;

        assert_eq!(verified_challenges, round_challenges);
        Ok(())
    }
}
