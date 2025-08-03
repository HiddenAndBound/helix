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
    final_claims: Vec<Fp4>,
}

impl OuterSumCheckProof {
    pub fn new(round_proofs: Vec<UnivariatePoly>, final_claims: Vec<Fp4>) -> Self {
        Self {
            round_proofs,
            final_claims,
        }
    }

    pub fn prove(
        A: &SparseMLE,
        B: &SparseMLE,
        C: &SparseMLE,
        z: &MLE<Fp>,
        challenger: &mut Challenger,
    ) {
        //First compute A.z, B.z and C.z

        let (a, b, c) = (
            A.multiply_by_mle(z).unwrap(),
            B.multiply_by_mle(z).unwrap(),
            C.multiply_by_mle(z).unwrap(),
        );

        let rounds = a.n_vars();

        let eq_point = challenger.get_challenges(rounds - 1);

        let eq = EqEvals::gen_from_point(&eq_point);

        let current_claim = Fp4::ZERO;
        for round in 0..rounds {
            let mut round_coeffs = [Fp4::ZERO; 2];
            for i in 0..1 << (rounds - round - 1) {
                round_coeffs[0] += eq[i] * (a[i << 1] * b[i << i] - c[i << 1])
            }
            
        }
    }
}
