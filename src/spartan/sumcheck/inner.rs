use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4, challenger::Challenger, polynomial::MLE, spartan::univariate::UnivariatePoly,
};

/// Sum-check proof for inner product constraints of the form:
/// `f(x₁, ..., xₙ) = ∑_{w∈{0,1}ⁿ} ⟨A(w), B(w)⟩`
/// where A and B are vectors and ⟨·,·⟩ denotes the inner product.
#[derive(Debug, Clone, PartialEq)]
pub struct InnerSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r), C(r) and Z(r)] at the random point r.
    final_evals: [Fp4; 4],
}

impl InnerSumCheckProof {
    /// Creates a new inner sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 4]) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    /// Generates a batched sum-check proof for inner product claims.
    /// Proves: (γ₀ A_bound(y) + γ₀² B_bound(y) + γ₀³ C_bound(y)) · Z(y) = batched_claim
    ///
    /// This verifies the evaluation claims from OuterSumCheck by proving the inner products.
    pub fn prove(
        a_bound: &MLE<Fp4>,
        b_bound: &MLE<Fp4>,
        c_bound: &MLE<Fp4>,
        outer_claims: [Fp4; 3],
        gamma: Fp4,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        // Use the bound matrices from outer sumcheck
        let rounds = a_bound.n_vars();

        let mut current_claim = gamma * outer_claims[0]
            + gamma.square() * outer_claims[1]
            + gamma.cube() * outer_claims[2];
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately - note: bound matrices are already in Fp4
        let round_proof = compute_inner_first_round_batched(
            &a_bound,
            &b_bound,
            &c_bound,
            gamma,
            &z,
            current_claim,
            rounds,
        );

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a_bound.fold_in_place(round_challenge);
        let mut b_fold = b_bound.fold_in_place(round_challenge);
        let mut c_fold = c_bound.fold_in_place(round_challenge);
        let mut z_fold = z.fold_in_place(round_challenge);

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_inner_round_batched(
                &a_fold,
                &b_fold,
                &c_fold,
                gamma,
                &z_fold,
                current_claim,
                round,
                rounds,
            );

            challenger.observe_fp4_elems(&round_proof.coefficients());
            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            a_fold = a_fold.fold_in_place(round_challenge);
            b_fold = b_fold.fold_in_place(round_challenge);
            c_fold = c_fold.fold_in_place(round_challenge);
            z_fold = z_fold.fold_in_place(round_challenge);
        }

        // Extract final evaluations A_bound(r), B_bound(r), C_bound(r), Z(r)
        let final_evals = [a_fold[0], b_fold[0], c_fold[0], z_fold[0]];

        InnerSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the inner sum-check proof. Panics if verification fails.
    pub fn verify(&self, outer_claims: [Fp4; 3], gamma: Fp4, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();

        let [a_claim, b_claim, c_claim] = outer_claims;
        let mut current_claim = gamma * a_claim + gamma.square() * b_claim + gamma.cube() * c_claim;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        let [a, b, c, z] = self.final_evals;
        // Final check: (γ·A_bound(r) + γ²·B_bound(r) + γ³·C_bound(r)) · Z(r) = final_claim
        assert_eq!(
            current_claim,
            (gamma * a + gamma.square() * b + gamma.cube() * c) * z
        )
    }
}

/// Computes the univariate polynomial for batched inner product sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [(γ·a(X,w) + γ²·b(X,w) + γ³·c(X,w)) * z(X,w)].
pub fn compute_inner_round_batched(
    a_bound: &MLE<Fp4>,
    b_bound: &MLE<Fp4>,
    c_bound: &MLE<Fp4>,
    gamma: Fp4,
    z: &MLE<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set first variable to 0
        round_coeffs[0] += (gamma * a_bound[i << 1]
            + gamma_squared * b_bound[i << 1]
            + gamma_cubed * c_bound[i << 1])
            * Fp4::from(z[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += (gamma * (a_bound[i << 1] + a_bound[i << 1 | 1].double())
            + gamma_squared * (b_bound[i << 1] + b_bound[i << 1 | 1].double())
            + gamma_cubed * (c_bound[i << 1] + c_bound[i << 1 | 1].double()))
            * (Fp4::from(z[i << 1]) + Fp4::from(z[i << 1 | 1]).double());
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first batched inner sum-check round.
/// Since bound matrices are already in Fp4, we work directly with Fp4.
pub fn compute_inner_first_round_batched(
    a_bound: &MLE<Fp4>,
    b_bound: &MLE<Fp4>,
    c_bound: &MLE<Fp4>,
    gamma: Fp4,
    z: &MLE<Fp>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0 and 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let gamma_squared = gamma.square();
    let gamma_cubed = gamma.cube();

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0

        round_coeffs[0] += (gamma * a_bound[i << 1]
            + gamma_squared * b_bound[i << 1]
            + gamma_cubed * c_bound[i << 1])
            * Fp4::from(z[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += (gamma * (a_bound[i << 1] + a_bound[i << 1 | 1].double())
            + gamma_squared * (b_bound[i << 1] + b_bound[i << 1 | 1].double())
            + gamma_cubed * (c_bound[i << 1] + c_bound[i << 1 | 1].double()))
            * (Fp4::from(z[i << 1]) + Fp4::from(z[i << 1 | 1]).double());
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = current_claim - round_coeffs[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}
