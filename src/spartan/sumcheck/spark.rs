use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp,
    Fp4,
    challenger::Challenger,
    polynomial::MLE,
    spartan::{ spark::sparse::SparkMetadata, univariate::UnivariatePoly },
};

/// Sum-check proof for inner product constraints of the form:
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} ⟨A(w), B(w)⟩`
/// where A and B are vectors and ⟨·,·⟩ denotes the inner product.
#[derive(Debug, Clone, PartialEq)]
pub struct SparkSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [row(r), e_rx(r), e_ry(r)] at the random point r for each of the 3 matrics.
    final_evals: [Fp4; 9],
}

impl SparkSumCheckProof {
    /// Creates a new inner sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 9]) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a batched sum-check proof for sparse mle evaluation.
    /// Proves: row * e_rx * e_ry as a
    ///
    /// This verifies the evaluation claims from OuterSumCheck by proving the inner products.
    pub fn prove(
        metadatas: &[SparkMetadata; 3],
        oracle_pairs: &[(MLE<Fp4>, MLE<Fp4>); 3],
        evaluation_claims: [Fp4; 3],
        gamma: Fp4,
        challenger: &mut Challenger
    ) -> Self {
        let rounds = metadatas[0].val().n_vars();

        // Batch the evaluation claims with gamma powers
        let mut current_claim =
            gamma * evaluation_claims[0] +
            gamma.square() * evaluation_claims[1] +
            gamma.cube() * evaluation_claims[2];

        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Process first round
        let round_proof = compute_spark_first_round_batched(
            &metadatas[0].val(),
            &metadatas[1].val(),
            &metadatas[2].val(),
            &oracle_pairs[0].0,
            &oracle_pairs[0].1,
            &oracle_pairs[1].0,
            &oracle_pairs[1].1,
            &oracle_pairs[2].0,
            &oracle_pairs[2].1,
            gamma,
            current_claim,
            rounds
        );

        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold MLEs for the first time
        let mut val_a_folded = metadatas[0].val().fold_in_place(round_challenge);
        let mut val_b_folded = metadatas[1].val().fold_in_place(round_challenge);
        let mut val_c_folded = metadatas[2].val().fold_in_place(round_challenge);
        let mut e_rx_a_folded = oracle_pairs[0].0.fold_in_place(round_challenge);
        let mut e_ry_a_folded = oracle_pairs[0].1.fold_in_place(round_challenge);
        let mut e_rx_b_folded = oracle_pairs[1].0.fold_in_place(round_challenge);
        let mut e_ry_b_folded = oracle_pairs[1].1.fold_in_place(round_challenge);
        let mut e_rx_c_folded = oracle_pairs[2].0.fold_in_place(round_challenge);
        let mut e_ry_c_folded = oracle_pairs[2].1.fold_in_place(round_challenge);

        // Process remaining rounds
        for round in 1..rounds {
            let round_proof = compute_spark_round_batched(
                &val_a_folded,
                &val_b_folded,
                &val_c_folded,
                &e_rx_a_folded,
                &e_ry_a_folded,
                &e_rx_b_folded,
                &e_ry_b_folded,
                &e_rx_c_folded,
                &e_ry_c_folded,
                gamma,
                current_claim,
                round,
                rounds
            );

            round_proofs.push(round_proof.clone());
            challenger.observe_fp4_elems(&round_proof.coefficients());

            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold for the next round
            val_a_folded = val_a_folded.fold_in_place(round_challenge);
            val_b_folded = val_b_folded.fold_in_place(round_challenge);
            val_c_folded = val_c_folded.fold_in_place(round_challenge);
            e_rx_a_folded = e_rx_a_folded.fold_in_place(round_challenge);
            e_ry_a_folded = e_ry_a_folded.fold_in_place(round_challenge);
            e_rx_b_folded = e_rx_b_folded.fold_in_place(round_challenge);
            e_ry_b_folded = e_ry_b_folded.fold_in_place(round_challenge);
            e_rx_c_folded = e_rx_c_folded.fold_in_place(round_challenge);
            e_ry_c_folded = e_ry_c_folded.fold_in_place(round_challenge);
        }

        // Extract final evaluations
        let final_evals = [
            val_a_folded[0],
            e_rx_a_folded[0],
            e_ry_a_folded[0],
            val_b_folded[0],
            e_rx_b_folded[0],
            e_ry_b_folded[0],
            val_c_folded[0],
            e_rx_c_folded[0],
            e_ry_c_folded[0],
        ];

        SparkSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the inner sum-check proof. Panics if verification fails.
    pub fn verify(&self, evaluation_claims: [Fp4; 3], gamma: Fp4, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();

        // Recompute the batched claim
        let mut current_claim =
            gamma * evaluation_claims[0] +
            gamma.square() * evaluation_claims[1] +
            gamma.cube() * evaluation_claims[2];

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = g_i(0) + g_i(1)
            assert_eq!(
                current_claim,
                round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
        }

        // Final check: batched evaluation of final values must match the final claim
        let [val_a, e_rx_a, e_ry_a, val_b, e_rx_b, e_ry_b, val_c, e_rx_c, e_ry_c] =
            self.final_evals;

        let final_eval_a = val_a * e_rx_a * e_ry_a;
        let final_eval_b = val_b * e_rx_b * e_ry_b;
        let final_eval_c = val_c * e_rx_c * e_ry_c;

        let expected_claim =
            gamma * final_eval_a + gamma.square() * final_eval_b + gamma.cube() * final_eval_c;

        assert_eq!(current_claim, expected_claim);
    }
}

/// Computes the univariate polynomial for batched inner product sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [(γ·a(X,w) + γ²·b(X,w) + γ³·c(X,w)) * z(X,w)].
pub fn compute_spark_round_batched(
    val_a: &MLE<Fp4>,
    val_b: &MLE<Fp4>,
    val_c: &MLE<Fp4>,
    e_rx_a: &MLE<Fp4>,
    e_ry_a: &MLE<Fp4>,
    e_rx_b: &MLE<Fp4>,
    e_ry_b: &MLE<Fp4>,
    e_rx_c: &MLE<Fp4>,
    e_ry_c: &MLE<Fp4>,
    gamma: Fp4,
    current_claim: Fp4,
    round: usize,
    rounds: usize
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - round - 1) {
        // Terms for g(0)
        let term_a_0 = val_a[i << 1] * e_rx_a[i << 1] * e_ry_a[i << 1];
        let term_b_0 = val_b[i << 1] * e_rx_b[i << 1] * e_ry_b[i << 1];
        let term_c_0 = val_c[i << 1] * e_rx_c[i << 1] * e_ry_c[i << 1];
        round_coeffs[0] += gamma * term_a_0 + gamma_squared * term_b_0 + gamma_cubed * term_c_0;

        // Terms for g(2)
        let val_a_2 = val_a[i << 1] + val_a[(i << 1) | 1].double();
        let val_b_2 = val_b[i << 1] + val_b[(i << 1) | 1].double();
        let val_c_2 = val_c[i << 1] + val_c[(i << 1) | 1].double();

        let e_rx_a_2 = e_rx_a[i << 1] + e_rx_a[(i << 1) | 1].double();
        let e_ry_a_2 = e_ry_a[i << 1] + e_ry_a[(i << 1) | 1].double();
        let e_rx_b_2 = e_rx_b[i << 1] + e_rx_b[(i << 1) | 1].double();
        let e_ry_b_2 = e_ry_b[i << 1] + e_ry_b[(i << 1) | 1].double();
        let e_rx_c_2 = e_rx_c[i << 1] + e_rx_c[(i << 1) | 1].double();
        let e_ry_c_2 = e_ry_c[i << 1] + e_ry_c[(i << 1) | 1].double();

        let term_a_2 = val_a_2 * e_rx_a_2 * e_ry_a_2;
        let term_b_2 = val_b_2 * e_rx_b_2 * e_ry_b_2;
        let term_c_2 = val_c_2 * e_rx_c_2 * e_ry_c_2;
        round_coeffs[2] += gamma * term_a_2 + gamma_squared * term_b_2 + gamma_cubed * term_c_2;
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first batched inner sum-check round.
/// Since bound matrices are already in Fp4, we work directly with Fp4.
pub fn compute_spark_first_round_batched(
    val_a: &MLE<Fp>,
    val_b: &MLE<Fp>,
    val_c: &MLE<Fp>,
    e_rx_a: &MLE<Fp4>,
    e_ry_a: &MLE<Fp4>,
    e_rx_b: &MLE<Fp4>,
    e_ry_b: &MLE<Fp4>,
    e_rx_c: &MLE<Fp4>,
    e_ry_c: &MLE<Fp4>,
    gamma: Fp4,
    current_claim: Fp4,
    rounds: usize
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - 1) {
        // Terms for g(0)
        let term_a_0 = Fp4::from(val_a[i << 1]) * e_rx_a[i << 1] * e_ry_a[i << 1];
        let term_b_0 = Fp4::from(val_b[i << 1]) * e_rx_b[i << 1] * e_ry_b[i << 1];
        let term_c_0 = Fp4::from(val_c[i << 1]) * e_rx_c[i << 1] * e_ry_c[i << 1];
        round_coeffs[0] += gamma * term_a_0 + gamma_squared * term_b_0 + gamma_cubed * term_c_0;

        // Terms for g(2)
        let val_a_2 = Fp4::from(val_a[i << 1]) + Fp4::from(val_a[(i << 1) | 1]).double();
        let val_b_2 = Fp4::from(val_b[i << 1]) + Fp4::from(val_b[(i << 1) | 1]).double();
        let val_c_2 = Fp4::from(val_c[i << 1]) + Fp4::from(val_c[(i << 1) | 1]).double();

        let e_rx_a_2 = e_rx_a[i << 1] + e_rx_a[(i << 1) | 1].double();
        let e_ry_a_2 = e_ry_a[i << 1] + e_ry_a[(i << 1) | 1].double();
        let e_rx_b_2 = e_rx_b[i << 1] + e_rx_b[(i << 1) | 1].double();
        let e_ry_b_2 = e_ry_b[i << 1] + e_ry_b[(i << 1) | 1].double();
        let e_rx_c_2 = e_rx_c[i << 1] + e_rx_c[(i << 1) | 1].double();
        let e_ry_c_2 = e_ry_c[i << 1] + e_ry_c[(i << 1) | 1].double();

        let term_a_2 = val_a_2 * e_rx_a_2 * e_ry_a_2;
        let term_b_2 = val_b_2 * e_rx_b_2 * e_ry_b_2;
        let term_c_2 = val_c_2 * e_rx_c_2 * e_ry_c_2;
        round_coeffs[2] += gamma * term_a_2 + gamma_squared * term_b_2 + gamma_cubed * term_c_2;
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}
