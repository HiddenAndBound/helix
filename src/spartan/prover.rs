//! Spartan zkSNARK prover implementation.
//!
//! Spartan provides zero-knowledge proofs for R1CS instances without trusted setup.
//! Uses sum-check protocols for efficient proving with logarithmic verification time.

use crate::{
    Fp4,
    challenger::Challenger,
    spartan::{
        R1CSInstance,
        sumcheck::{InnerSumCheckProof, OuterSumCheckProof},
    },
};

/// Spartan zkSNARK proof for an R1CS instance.
#[derive(Debug, Clone, PartialEq)]
pub struct SpartanProof {
    /// The outer sum-check proof demonstrating R1CS constraint satisfaction.
    outer_sumcheck_proof: OuterSumCheckProof,
    inner_sumcheck_proof: InnerSumCheckProof,
}

impl SpartanProof {
    /// Creates a new Spartan proof from an outer sum-check proof.
    pub fn new(
        outer_sumcheck_proof: OuterSumCheckProof,
        inner_sumcheck_proof: InnerSumCheckProof,
    ) -> Self {
        Self {
            outer_sumcheck_proof,
            inner_sumcheck_proof,
        }
    }

    /// Returns a reference to the outer sum-check proof.
    pub fn outer_sumcheck_proof(&self) -> &OuterSumCheckProof {
        &self.outer_sumcheck_proof
    }

    pub fn inner_sumcheck_proof(&self) -> &InnerSumCheckProof {
        &self.inner_sumcheck_proof
    }

    pub fn prove(instance: R1CSInstance, challenger: &mut Challenger) -> Self {
        let z = &instance.witness_mle();
        let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

        // Phase 1: OuterSumCheck - proves R1CS constraint satisfaction
        // Generates evaluation claims A(r_x), B(r_x), C(r_x) at random point r_x
        let outer_sum_check = OuterSumCheckProof::prove(A, B, C, z, challenger);

        // Extract the random point from outer sumcheck to bind first half of variables
        let outer_claims = outer_sum_check.final_evals.clone();

        // Use the random challenge from outer sumcheck to compute bound matrices
        // This gives us A_bound(y) = A(r_x, y), B_bound(y) = B(r_x, y), C_bound(y) = C(r_x, y)
        let r_x_point: Vec<Fp4> =
            challenger.get_challenges(A.dimensions().0.trailing_zeros() as usize);
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

        // Phase 2: InnerSumCheck - proves evaluation claims using bound matrices
        // Verifies: (γ·A_bound(y) + γ²·B_bound(y) + γ³·C_bound(y)) · Z(y) = batched_claim
        let gamma = challenger.get_challenge(); // Random batching challenge
        let inner_sum_check = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            [outer_claims[0], outer_claims[1], outer_claims[2]],
            gamma,
            z,
            challenger,
        );

        SpartanProof::new(outer_sum_check, inner_sum_check)
    }
    /// Verifies the Spartan proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut crate::challenger::Challenger) {
        // Phase 1: Verify the outer sum-check proof
        // This ensures R1CS constraints are satisfied
        self.outer_sumcheck_proof.verify(challenger);

        // Phase 2: Verify the inner sum-check proof
        // This ensures evaluation claims from outer sumcheck are correct
        self.inner_sumcheck_proof.verify(challenger);

        // Note: In a complete Spartan implementation, additional steps would include:
        // - Polynomial commitment opening verifications (SparkSumCheck)
        // - Consistency checks between proof phases
        // - Final point evaluation verification
    }
}
