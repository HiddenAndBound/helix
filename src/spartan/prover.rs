use crate::Fp4;

pub struct SpartanProof {
    outer_sumcheck_proof: OuterSumCheckProof,
}

pub struct OuterSumCheckProof {
    round_proofs: Vec<UnivariatePoly>,
    final_claims: Vec<Fp4>,
}

