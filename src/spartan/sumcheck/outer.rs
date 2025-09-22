use std::{ any, collections::HashMap };

use anyhow::{ anyhow, bail };
use itertools::multizip;
use p3_field::{ ExtensionField, Field, PackedValue, PrimeCharacteristicRing };
use rand::{ Rng, SeedableRng, rngs::StdRng };

use crate::{
    Fp,
    Fp4,
    challenger::Challenger,
    eq::EqEvals,
    polynomial::MLE,
    spartan::{ spark::sparse::SparseMLE, univariate::UnivariatePoly },
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
        challenger: &mut Challenger
    ) -> (Self, Vec<Fp4>) {
        // Compute A·z, B·z, C·z (sparse matrix-MLE multiplications)
        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        assert!(rounds > 0, "MLEs need to be non empty");

        // Get random evaluation point from challenger (Fiat-Shamir)
        let eq_point = challenger.get_challenges(rounds);

        // Initialize equality polynomial eq(x, r) for rounds 1..n
        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;
        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();

        let mut a_fold = MLE::default();
        let mut b_fold = MLE::default();
        let mut c_fold = MLE::default();

        // Process remaining rounds (1 to n-1)
        for round in 0..rounds {
            let round_proof = match round {
                0 => compute_round(&a, &b, &c, &eq, &eq_point, current_claim, round, rounds),
                _ =>
                    compute_round(
                        &a_fold,
                        &b_fold,
                        &c_fold,
                        &eq,
                        &eq_point,
                        current_claim,
                        round,
                        rounds
                    ),
            };

            challenger.observe_fp4_elems(&round_proof.coefficients());

            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);
            current_claim = round_proof.evaluate(round_challenge);

            // Fold polynomials for next round
            (a_fold, b_fold, c_fold) = match round {
                0 =>
                    (
                        a.fold_in_place(round_challenge),
                        b.fold_in_place(round_challenge),
                        c.fold_in_place(round_challenge),
                    ),
                _ =>
                    (
                        a_fold.fold_in_place(round_challenge),
                        b_fold.fold_in_place(round_challenge),
                        c_fold.fold_in_place(round_challenge),
                    ),
            };

            eq.fold_in_place();
            round_proofs.push(round_proof);
        }

        // Extract final evaluations A(r), B(r), C(r)
        let final_evals = [a_fold[0], b_fold[0], c_fold[0]];

        (OuterSumCheckProof::new(round_proofs, final_evals), round_challenges)
    }

    /// Verifies the sum-check proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut Challenger) -> anyhow::Result<(Vec<Fp4>, [Fp4; 3])> {
        let rounds = self.round_proofs.len();

        let eq_point = challenger.get_challenges(rounds);

        let mut current_claim = Fp4::ZERO;
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
        }

        // Final check: A(r)·B(r) - C(r) = final_claim
        if current_claim != self.final_evals[0] * self.final_evals[1] - self.final_evals[2] {
            return Err(anyhow!("Final Check Failed in OuterSumCheck"));
        }

        Ok((round_challenges, self.final_evals))
    }
}

/// Computes the univariate polynomial for sum-check rounds 1 to n-1.
/// Returns g(X) = ∑_{w∈{0,1}^{n-round-1}} eq(w) * [a(X,w) * b(X,w) - c(X,w)].
pub fn compute_round<F>(
    a: &MLE<F>,
    b: &MLE<F>,
    c: &MLE<F>,
    eq: &EqEvals,
    eq_point: &Vec<Fp4>,
    current_claim: Fp4,
    round: usize,
    rounds: usize
)
    -> UnivariatePoly
    where F: Field, Fp4: ExtensionField<F>
{
    // Use Gruen's optimization: compute evaluations at X = 0, 1, 2
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    for i in 0..1 << (rounds - round - 1) {
        // g(0): set current variable to 0
        round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1]);

        // g(2): use multilinear polynomial identity
        round_coeffs[2] +=
            eq[i] *
            ((a[(i << 1) | 1].double() - a[i << 1]) * (b[(i << 1) | 1].double() - b[i << 1]) -
                (c[(i << 1) | 1].double() - c[i << 1]));
    }

    // g(1): derived from sum-check constraint
    round_coeffs[1] =
        (current_claim - round_coeffs[0] * (Fp4::ONE - eq_point[round])) / eq_point[round];

    let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();
    round_proof.interpolate().unwrap();

    round_proof
}

#[test]
fn outer_sum_check_test() -> anyhow::Result<()> {
    // This is also the number of nonlinear constraints.
    const ROWS: usize = 1 << 10;
    const COLS: usize = ROWS;
    let mut rng = StdRng::seed_from_u64(0);
    let mut A = HashMap::<(usize, usize), Fp>::new();
    let mut B = HashMap::<(usize, usize), Fp>::new();
    let mut C = HashMap::<(usize, usize), Fp>::new();
    let mut z = MLE::new(Fp::new_array([rng.r#gen::<u32>(); COLS as usize]).to_vec());

    let z_const = z[0];
    let z_inv_const = z[0].inverse();
    // (a*b - c = 0 => c = a*b)
    for j in 0..ROWS {
        let a = Fp::new(rng.r#gen());
        let b = Fp::new(rng.r#gen());

        let i_0 = rng.gen_range(0..COLS);
        let i_1 = rng.gen_range(0..COLS);

        A.insert((j, i_0), a);
        B.insert((j, i_1), b);

        // Trivial equation a.b/z_inv_const
        C.insert((j, j), a * b * z_const);
    }

    let A = SparseMLE::new(A)?;
    let B = SparseMLE::new(B)?;
    let C = SparseMLE::new(C)?;

    let a = A.multiply_by_mle(&z)?;
    let b = B.multiply_by_mle(&z)?;
    let c = C.multiply_by_mle(&z)?;

    for (&a_i, &b_i, &c_i) in multizip((a.coeffs(), b.coeffs(), c.coeffs())) {
        if c_i != a_i * b_i {
            bail!("R1CS instance not satisfied");
        }
    }

    let mut challenger = Challenger::new();
    let (proof, _) = OuterSumCheckProof::prove(&A, &B, &C, &z, &mut challenger);

    let mut challenger = Challenger::new();
    proof.verify(&mut challenger)?;
    Ok(())
}
