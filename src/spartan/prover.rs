//! Spartan zkSNARK prover implementation.
//!
//! Spartan provides zero-knowledge proofs for R1CS instances without trusted setup.
//! Uses sum-check protocols for efficient proving with logarithmic verification time.

use crate::{Fp4, spartan::sumcheck::OuterSumCheckProof};

/// Spartan zkSNARK proof for an R1CS instance.
#[derive(Debug, Clone, PartialEq)]
pub struct SpartanProof {
    /// The outer sum-check proof demonstrating R1CS constraint satisfaction.
    outer_sumcheck_proof: OuterSumCheckProof,
}

impl SpartanProof {
    /// Creates a new Spartan proof from an outer sum-check proof.
    pub fn new(outer_sumcheck_proof: OuterSumCheckProof) -> Self {
        Self {
            outer_sumcheck_proof,
        }
    }

    /// Returns a reference to the outer sum-check proof.
    pub fn outer_sumcheck_proof(&self) -> &OuterSumCheckProof {
        &self.outer_sumcheck_proof
    }

    /// Verifies the Spartan proof. Panics if verification fails.
    pub fn verify(&self, challenger: &mut crate::challenger::Challenger) {
        // Verify the outer sum-check proof
        self.outer_sumcheck_proof.verify(challenger);

        // TODO: Add additional verification steps for a complete Spartan proof:
        // - Inner sum-check verifications
        // - Polynomial commitment verifications
        // - Final consistency checks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        challenger::Challenger,
        polynomial::MLE,
        spartan::{sparse::SparseMLE, sumcheck::OuterSumCheckProof, univariate::UnivariatePoly},
        utils::{Fp, Fp4},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use std::collections::HashMap;

    /// Helper function to create a test challenger
    fn create_test_challenger() -> crate::challenger::Challenger {
        Challenger::new()
    }

    /// Helper function to create a simple outer sumcheck proof for testing
    fn create_test_outer_sumcheck_proof() -> OuterSumCheckProof {
        let round_proofs = vec![
            UnivariatePoly::from_coeffs(Fp4::ZERO, Fp4::ONE),
            UnivariatePoly::from_coeffs(Fp4::ONE, Fp4::ZERO),
        ];
        let final_evals = vec![Fp4::ZERO, Fp4::ONE, Fp4::ZERO];

        OuterSumCheckProof::new(round_proofs, final_evals)
    }

    /// Helper function to create test sparse matrices and witness
    fn create_test_r1cs() -> (SparseMLE, SparseMLE, SparseMLE, MLE<Fp>) {
        // Create simple constraint system: A * z ⊙ B * z = C * z
        let mut a_coeffs = HashMap::new();
        a_coeffs.insert((0, 0), BabyBear::ONE);
        a_coeffs.insert((1, 1), BabyBear::from_u32(2));
        let matrix_a = SparseMLE::new(a_coeffs).unwrap();

        let mut b_coeffs = HashMap::new();
        b_coeffs.insert((0, 0), BabyBear::from_u32(3));
        b_coeffs.insert((1, 1), BabyBear::from_u32(4));
        let matrix_b = SparseMLE::new(b_coeffs).unwrap();

        let mut c_coeffs = HashMap::new();
        c_coeffs.insert((0, 0), BabyBear::from_u32(3)); // 1 * 3 = 3
        c_coeffs.insert((1, 1), BabyBear::from_u32(8)); // 2 * 4 = 8
        let matrix_c = SparseMLE::new(c_coeffs).unwrap();

        let witness = MLE::new(vec![BabyBear::ONE, BabyBear::ONE]);

        (matrix_a, matrix_b, matrix_c, witness)
    }

    #[test]
    fn test_spartan_proof_new() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof = SpartanProof::new(outer_proof.clone());

        assert_eq!(*proof.outer_sumcheck_proof(), outer_proof);
    }

    #[test]
    fn test_spartan_proof_outer_sumcheck_proof_accessor() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof = SpartanProof::new(outer_proof.clone());

        let accessed_proof = proof.outer_sumcheck_proof();
        assert_eq!(*accessed_proof, outer_proof);
    }

    #[test]
    #[should_panic]
    fn test_spartan_proof_verify_with_invalid_proof() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof = SpartanProof::new(outer_proof);
        let mut challenger = create_test_challenger();

        // This test expects the verification to panic since we're using an invalid proof
        // In a real scenario, we'd create a valid proof from real R1CS data
        proof.verify(&mut challenger);
    }

    #[test]
    fn test_spartan_proof_integration_with_real_sumcheck() {
        let (matrix_a, matrix_b, matrix_c, witness) = create_test_r1cs();
        let mut prover_challenger = create_test_challenger();

        // Generate a real outer sumcheck proof
        let outer_proof = OuterSumCheckProof::prove(
            &matrix_a,
            &matrix_b,
            &matrix_c,
            &witness,
            &mut prover_challenger,
        );

        // Create Spartan proof with the real sumcheck proof
        let spartan_proof = SpartanProof::new(outer_proof);

        // Verify the Spartan proof
        let mut verifier_challenger = create_test_challenger();
        spartan_proof.verify(&mut verifier_challenger);

        // If we reach here, verification passed
        assert!(true);
    }

    #[test]
    fn test_spartan_proof_clone() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof1 = SpartanProof::new(outer_proof);
        let proof2 = proof1.clone();

        assert_eq!(proof1, proof2);
    }

    #[test]
    fn test_spartan_proof_debug() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof = SpartanProof::new(outer_proof);

        // Test that Debug is implemented
        let debug_str = format!("{:?}", proof);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("SpartanProof"));
    }

    #[test]
    fn test_spartan_proof_equality() {
        let outer_proof1 = create_test_outer_sumcheck_proof();
        let outer_proof2 = create_test_outer_sumcheck_proof();

        let proof1 = SpartanProof::new(outer_proof1);
        let proof2 = SpartanProof::new(outer_proof2);

        // Should be equal since they use the same test data
        assert_eq!(proof1, proof2);
    }

    #[test]
    fn test_spartan_proof_with_different_outer_proofs() {
        let outer_proof1 = create_test_outer_sumcheck_proof();

        // Create a different outer proof
        let round_proofs2 = vec![
            UnivariatePoly::from_coeffs(Fp4::ONE, Fp4::ONE),
            UnivariatePoly::from_coeffs(Fp4::ZERO, Fp4::ZERO),
        ];
        let final_evals2 = vec![Fp4::ONE, Fp4::ZERO, Fp4::ONE];
        let outer_proof2 = OuterSumCheckProof::new(round_proofs2, final_evals2);

        let proof1 = SpartanProof::new(outer_proof1);
        let proof2 = SpartanProof::new(outer_proof2);

        // Should be different
        assert_ne!(proof1, proof2);
    }

    #[test]
    fn test_spartan_proof_structure() {
        let outer_proof = create_test_outer_sumcheck_proof();
        let proof = SpartanProof::new(outer_proof.clone());

        // Test that the proof contains the expected outer sumcheck proof
        assert_eq!(*proof.outer_sumcheck_proof(), outer_proof);

        // Test that the proof structure is as expected
        // (In a full implementation, we'd test additional components)
        // Note: We can only test public interface since fields are private
    }
}
