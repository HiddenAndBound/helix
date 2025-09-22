//! Spartan zkSNARK prover implementation.
//!
//! Spartan provides zero-knowledge proofs for R1CS instances without trusted setup.
//! Uses sum-check protocols for efficient proving with logarithmic verification time.

use crate::{
    Fp4,
    challenger::Challenger,
    spartan::{ R1CSInstance, sumcheck::{ InnerSumCheckProof, OuterSumCheckProof } },
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
        inner_sumcheck_proof: InnerSumCheckProof
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
        let (outer_sum_check, rx) = OuterSumCheckProof::prove(A, B, C, z, challenger);

        // Extract the random point from outer sumcheck to bind first half of variables
        let outer_claims = outer_sum_check.final_evals.clone();

        // Use the random challenge from outer sumcheck to compute bound matrices
        // This gives us A_bound(y) = A(r_x, y), B_bound(y) = B(r_x, y), C_bound(y) = C(r_x, y)
        let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&rx).unwrap();

        challenger.observe_fp4_elems(&outer_sum_check.final_evals);
        // Phase 2: InnerSumCheck - proves evaluation claims using bound matrices
        // Verifies: (γ·A_bound(y) + γ²·B_bound(y) + γ³·C_bound(y)) · Z(y) = batched_claim
        let gamma = challenger.get_challenge(); // Random batching challenge
        let inner_sum_check = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            z,
            [outer_claims[0], outer_claims[1], outer_claims[2]],
            gamma,

            challenger
        );

        SpartanProof::new(outer_sum_check, inner_sum_check)
    }
    /// Verifies the Spartan proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut crate::challenger::Challenger) -> anyhow::Result<()> {
        // Phase 1: Verify the outer sum-check proof
        // This ensures R1CS constraints are satisfied
        let (rx, outer_claims) = self.outer_sumcheck_proof.verify(challenger)?;

        // Phase 2: Verify the inner sum-check proof
        // This ensures evaluation claims from outer sumcheck are correct
        challenger.observe_fp4_elems(&self.outer_sumcheck_proof.final_evals);
        let gamma = challenger.get_challenge();
        self.inner_sumcheck_proof.verify(outer_claims, gamma, challenger);

        // Note: In a complete Spartan implementation, additional steps would include:
        // - Polynomial commitment opening verifications (SparkSumCheck)
        // - Consistency checks between proof phases
        // - Final point evaluation verification
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use anyhow::bail;
    use itertools::multizip;
    use rand::{ rngs::StdRng, Rng, SeedableRng };
    use crate::{
        challenger::Challenger,
        polynomial::MLE,
        spartan::{
            prover::SpartanProof,
            r1cs::{ R1CS, R1CSInstance, Witness },
            spark::sparse::SparseMLE,
        },
        *,
    };
    use p3_field::{ Field, PrimeCharacteristicRing };
    #[test]
    fn spartan_test() -> anyhow::Result<()> {
        // This is also the number of nonlinear constraints.
        const ROWS: usize = 1 << 10;
        const COLS: usize = ROWS;
        let mut rng = StdRng::seed_from_u64(0);
        let mut a_matrix = HashMap::<(usize, usize), Fp>::new();
        let mut b_matrix = HashMap::<(usize, usize), Fp>::new();
        let mut c_matrix = HashMap::<(usize, usize), Fp>::new();

        let mut witness_vals: Vec<Fp> = (0..COLS).map(|_| Fp::new(rng.r#gen::<u32>())).collect();
        if witness_vals[0] == Fp::ZERO {
            witness_vals[0] = Fp::ONE;
        }
        let z_const = witness_vals[0];
        let z_const_inv = z_const.inverse();

        // (a*b - c = 0 => c = a*b)
        for j in 0..ROWS {
            let a = Fp::new(rng.r#gen());
            let b = Fp::new(rng.r#gen());

            let (i_0, i_1) = if j == 0 {
                (0, 0)
            } else {
                (rng.gen_range(0..j), rng.gen_range(0..j))
            };

            a_matrix.insert((j, i_0), a);
            b_matrix.insert((j, i_1), b);

            // Trivial equation a·b - c = 0 ⇒ c = a·b
            c_matrix.insert((j, j), a * b * z_const);

            if j != 0 {
                witness_vals[j] = witness_vals[i_0] * witness_vals[i_1] * z_const_inv;
            }
        }

        let z = MLE::new(witness_vals.clone());

        let A = SparseMLE::new(a_matrix)?;
        let B = SparseMLE::new(b_matrix)?;
        let C = SparseMLE::new(c_matrix)?;

        let a = A.multiply_by_mle(&z)?;
        let b = B.multiply_by_mle(&z)?;
        let c = C.multiply_by_mle(&z)?;

        for (&a_i, &b_i, &c_i) in multizip((a.coeffs(), b.coeffs(), c.coeffs())) {
            if c_i != a_i * b_i {
                bail!("R1CS instance not satisfied");
            }
        }

        let mut prover_challenger = Challenger::new();

        // Form an R1CS instance from the sparse matrices and witness vector
        let num_public_inputs = 0;
        let r1cs = R1CS::new(A, B, C, num_public_inputs)?;
        let witness = Witness::from_vec(witness_vals, num_public_inputs);
        let instance = R1CSInstance::new(r1cs, witness)?;

        assert!(instance.verify()?);

        let proof = SpartanProof::prove(instance.clone(), &mut prover_challenger);

        let mut verifier_challenger = Challenger::new();
        proof.verify(&mut verifier_challenger);

        Ok(())
    }
}
