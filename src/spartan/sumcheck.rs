use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    eq::EqEvals,
    polynomial::MLE,
    spartan::{sparse::SparseMLE, univariate::UnivariatePoly},
};
use std::fmt;

pub struct OuterSumCheckProof {
    round_proofs: Vec<UnivariatePoly>,
    final_evals: Vec<Fp4>,
}

impl OuterSumCheckProof {
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_evals: Vec<Fp4>) -> Self {
        Self {
            round_proofs,
            final_evals,
        }
    }

    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) -> Self {
        //First compute A.z, B.z and C.z

        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );
        let rounds = a.n_vars();

        let eq_point = challenger.get_challenges(rounds);

        let mut eq = EqEvals::gen_from_point(&eq_point[1..]);

        let mut current_claim = Fp4::ZERO;

        let mut round_proofs = Vec::new();
        let mut round_challenges = Vec::new();
        let mut round_coeffs = vec![Fp4::ZERO; 2];

        for i in 0..1 << (rounds - 1) {
            round_coeffs[0] += eq[i] * (a[i << 1] * b[i << 1] - c[i << 1])
        }

        round_coeffs[1] = (current_claim - current_claim * (Fp4::ONE + eq_point[0])) / eq_point[0];

        let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();

        round_proof.interpolate_in_place().unwrap();

        round_proofs.push(round_proof.clone());
        challenger.observe_fp4_elems(&round_proof.coefficients());

        let round_challenge = challenger.get_challenge();
        round_challenges.push(round_challenge);

        current_claim = round_proof.eval_at(round_challenge);

        let mut a_fold = a.fold_in_place(round_challenge);
        let mut b_fold = b.fold_in_place(round_challenge);
        let mut c_fold = c.fold_in_place(round_challenge);
        eq.fold_in_place();
        for round in 1..rounds {
            let mut round_coeffs = vec![Fp4::ZERO; 2];
            for i in 0..1 << (rounds - round - 1) {
                round_coeffs[0] += eq[i] * (a_fold[i << 1] * b_fold[i << 1] - c_fold[i << 1])
            }

            round_coeffs[1] =
                (current_claim - current_claim * (Fp4::ONE + eq_point[round])) / eq_point[round];

            let mut round_proof = UnivariatePoly::new(round_coeffs).unwrap();

            round_proof.interpolate_in_place().unwrap();

            challenger.observe_fp4_elems(&round_proof.coefficients());

            let round_challenge = challenger.get_challenge();
            round_challenges.push(round_challenge);

            current_claim = round_proof.eval_at(round_challenge);

            a_fold = a_fold.fold_in_place(round_challenge);
            b_fold = b_fold.fold_in_place(round_challenge);
            c_fold = c_fold.fold_in_place(round_challenge);
            eq.fold_in_place();
        }

        let final_evals = vec![a_fold[0], b_fold[1], c_fold[2]];

        OuterSumCheckProof::new(round_proofs, final_evals)
    }
}
