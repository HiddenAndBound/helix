use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    polynomial::MLE,
    spartan::{spark::sparse::SparseMLE, univariate::UnivariatePoly},
};

/// Sum-check proof demonstrating that f(x₁, ..., xₙ) = A(x)·B(x) - C(x) sums to zero
/// over the boolean hypercube {0,1}ⁿ.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterSumCheckProof {
    /// Univariate polynomials for each round of the sum-check protocol.
    pub round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations [A(r), B(r), C(r)] at the random point r.
    pub final_evals: [Fp4; 3],
}

impl OuterSumCheckProof {
    /// Creates a new sum-check proof from round polynomials and final evaluations.
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: [Fp4; 3]) -> Self {
        Self {
            round_proofs,
            final_evals,
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
        challenger: &mut Challenger,
    ) -> Self {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        // Handle first round separately (uses base field Fp for efficiency)
        let round_proof = compute_first_round(&a, &b, &c, &eq, &eq_point, current_claim, rounds);

        // Process first round proof
        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);
        current_claim = round_proof.evaluate(round_challenge);

        // Fold polynomials by fixing first variable to challenge
        let mut a_fold = a.fold_in_place(round_challenge);
        let mut b_fold = b.fold_in_place(round_challenge);
        let mut c_fold = c.fold_in_place(round_challenge);
        eq.fold_in_place();

        // Process remaining rounds (1 to n-1)
        for round in 1..rounds {
            let round_proof = compute_round(
                &a_fold,
                &b_fold,
                &c_fold,
                &eq,
                &eq_point,
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
            eq.fold_in_place();
        }

        // Extract final evaluations A(r), B(r), C(r)
        let final_evals = [a_fold[0], b_fold[0], c_fold[0]];

        OuterSumCheckProof::new(round_proofs, final_evals)
    }

    /// Verifies the sum-check proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut Challenger) {
        let rounds = self.round_proofs.len();
        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = Fp4::ZERO;
        let mut round_challenges = Vec::new();

        // Verify each round of the sum-check protocol
        for round in 0..rounds {
            let round_poly = &self.round_proofs[round];

            // Check sum-check relation: current_claim = (1-r_i) * g_i(0) + r_i * g_i(1)
            assert_eq!(
                current_claim,
                (Fp4::ONE - eq_point[round]) * round_poly.evaluate(Fp4::ZERO)
                    + eq_point[round] * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            round_challenges.push(challenge);
        }

        // Final check: A(r)·B(r) - C(r) = final_claim
        assert_eq!(
            current_claim,
            self.final_evals[0] * self.final_evals[1] - self.final_evals[2]
        )
    }
}

/// Computes the univariate polynomial for sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w) - c(X,w)].
pub fn compute_round(
    a: &MLE<Fp4>,
    b: &MLE<Fp4>,
    c: &MLE<Fp4>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += eq[i]
            * ((a[i << 1] + a[i << 1 | 1].double()) * (b[i << 1] + b[i << 1 | 1].double())
                - (c[i << 1] + c[i << 1 | 1].double()));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

/// Computes the univariate polynomial for the first sum-check round.
/// Uses base field (Fp) arithmetic for efficiency, outputs in extension field (Fp4).
pub fn compute_first_round(
    a: &MLE<Fp>,
    b: &MLE<Fp>,
    c: &MLE<Fp>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    rounds: usize,
) -> UnivariatePoly {
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - 1) {
        // g(0): set first variable to 0 (base field Fp promoted to Fp4)
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] += (a[i << 1] + a[i << 1 | 1].double())
            * (b[i << 1] + b[i << 1 | 1].double())
            - (c[i << 1] + c[i << 1 | 1].double());
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] = (current_claim - round_coeffs[0] * (Fp4::ONE + eq_point[0])) / eq_point[0];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}
