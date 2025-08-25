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

        if rounds == 0 {
            return OuterSumCheckProof::new(vec![], [a[0].into(), b[0].into(), c[0].into()]);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        spartan::spark::sparse::SparseMLE,
        utils::{challenger::Challenger, polynomial::MLE},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing};
    use std::collections::HashMap;

    /// Creates a random R1CS instance (A, B, C, z) where (Az) ∘ (Bz) = Cz
    ///
    /// # Arguments
    /// * `n_vars` - Number of variables (log₂ of vector/matrix dimensions)
    /// * `seed` - Optional seed for deterministic randomness
    ///
    /// # Returns
    /// Tuple of (A, B, C, z) where all matrices are sparse and witness is dense
    fn create_random_r1cs_instance(
        n_vars: usize,
        seed: Option<u64>,
    ) -> (SparseMLE, SparseMLE, SparseMLE, MLE<BabyBear>) {
        let size = 1 << n_vars; // 2^n_vars
        let mut rng_state = seed.unwrap_or(12345);

        // Simple linear congruential generator for deterministic randomness
        let mut next_rand = || -> u32 {
            rng_state = (rng_state.wrapping_mul(1664525).wrapping_add(1013904223)) & 0xFFFFFFFF;
            (rng_state & 0x7FFFFFFF) as u32 // Keep positive
        };

        // Generate random sparse matrix A (about 30% density)
        let mut a_coeffs = HashMap::new();
        for i in 0..size {
            for j in 0..size {
                if next_rand() % 10 < 3 {
                    // 30% chance
                    let val = (next_rand() % 100) + 1; // Values 1-100
                    a_coeffs.insert((i, j), BabyBear::from_u32(val));
                }
            }
        }
        // Ensure at least some entries exist
        if a_coeffs.is_empty() {
            a_coeffs.insert((0, 0), BabyBear::ONE);
        }

        // Generate random sparse matrix B (about 30% density)
        let mut b_coeffs = HashMap::new();
        for i in 0..size {
            for j in 0..size {
                if next_rand() % 10 < 3 {
                    // 30% chance
                    let val = (next_rand() % 100) + 1; // Values 1-100
                    b_coeffs.insert((i, j), BabyBear::from_u32(val));
                }
            }
        }
        // Ensure at least some entries exist
        if b_coeffs.is_empty() {
            b_coeffs.insert((0, 0), BabyBear::ONE);
        }

        let a = SparseMLE::new(a_coeffs).unwrap();
        let b = SparseMLE::new(b_coeffs).unwrap();

        // Generate random witness vector z (all non-zero to avoid division issues)
        let mut z_coeffs = Vec::with_capacity(size);
        for _ in 0..size {
            let val = (next_rand() % 50) + 1; // Values 1-50
            z_coeffs.push(BabyBear::from_u32(val));
        }
        let z = MLE::new(z_coeffs);

        // Compute Az and Bz
        let az = a.multiply_by_mle(&z).unwrap();
        let bz = b.multiply_by_mle(&z).unwrap();

        // Compute (Az) ∘ (Bz) (Hadamard product)
        let hadamard_product: Vec<BabyBear> = az
            .coeffs()
            .iter()
            .zip(bz.coeffs().iter())
            .map(|(&a_val, &b_val)| a_val * b_val)
            .collect();

        // Construct matrix C such that Cz = (Az) ∘ (Bz)
        // For each row i: C[i,:] * z = hadamard_product[i]
        // We'll create a simple diagonal-like structure where C[i,i] = hadamard_product[i] / z[i]
        let mut c_coeffs = HashMap::new();
        for i in 0..size {
            if !z.coeffs()[i].is_zero() {
                let c_val = hadamard_product[i] / z.coeffs()[i];
                if !c_val.is_zero() {
                    c_coeffs.insert((i, i), c_val);
                }
            }
        }

        // Ensure C is not empty
        if c_coeffs.is_empty() {
            c_coeffs.insert((0, 0), BabyBear::ONE);
        }

        let c = SparseMLE::new(c_coeffs).unwrap();

        (a, b, c, z)
    }

    /// Creates a simple deterministic R1CS instance for testing
    fn create_simple_r1cs_instance(
        n_vars: usize,
    ) -> (SparseMLE, SparseMLE, SparseMLE, MLE<BabyBear>) {
        let size = 1 << n_vars;

        // Create identity matrices for A and B
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        for i in 0..size {
            a_coeffs.insert((i, i), BabyBear::ONE);
            b_coeffs.insert((i, i), BabyBear::ONE);
        }

        let a = SparseMLE::new(a_coeffs).unwrap();
        let b = SparseMLE::new(b_coeffs).unwrap();

        // Create simple witness: z[i] = i + 1
        let z_coeffs: Vec<BabyBear> = (1..=size).map(|i| BabyBear::from_u32(i as u32)).collect();
        let z = MLE::new(z_coeffs);

        // Since A = B = I, we have Az = Bz = z
        // So (Az) ∘ (Bz) = z ∘ z = z²
        // We need Cz = z², so C should be diag(z[i])
        let mut c_coeffs = HashMap::new();
        for i in 0..size {
            c_coeffs.insert((i, i), z.coeffs()[i]);
        }

        let c = SparseMLE::new(c_coeffs).unwrap();

        (a, b, c, z)
    }

    /// Validates that the R1CS constraint (Az) ∘ (Bz) = Cz holds
    fn validate_r1cs_constraint(
        a: &SparseMLE,
        b: &SparseMLE,
        c: &SparseMLE,
        z: &MLE<BabyBear>,
    ) -> bool {
        let az = a.multiply_by_mle(z).unwrap();
        let bz = b.multiply_by_mle(z).unwrap();
        let cz = c.multiply_by_mle(z).unwrap();

        // Check (Az) ∘ (Bz) = Cz element-wise
        for i in 0..az.len() {
            let hadamard_val = az.coeffs()[i] * bz.coeffs()[i];
            if hadamard_val != cz.coeffs()[i] {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_prover_verifier_flow() {
        // Test with a small 1-variable instance (2x2 matrices)
        let (a, b, c, z) = create_simple_r1cs_instance(1);

        // Verify the R1CS constraint holds
        assert!(
            validate_r1cs_constraint(&a, &b, &c, &z),
            "R1CS constraint (Az) ∘ (Bz) = Cz should hold"
        );

        // Run the prover-verifier protocol
        let mut prover_challenger = Challenger::new();
        let proof = OuterSumCheckProof::prove(&a, &b, &c, &z, &mut prover_challenger);

        // Verify the proof with a fresh challenger (same random tape)
        let mut verifier_challenger = Challenger::new();
        proof.verify(&mut verifier_challenger);

        println!("✅ Simple prover-verifier flow test passed!");
        println!("   Proof contains {} rounds", proof.round_proofs.len());
    }

    #[test]
    fn test_prover_verifier_flow_2vars() {
        // Test with a 2-variable instance (4x4 matrices)
        let (a, b, c, z) = create_simple_r1cs_instance(2);

        // Verify the R1CS constraint holds
        assert!(
            validate_r1cs_constraint(&a, &b, &c, &z),
            "R1CS constraint should hold for 2-variable instance"
        );

        // Run the prover-verifier protocol
        let mut prover_challenger = Challenger::new();
        let proof = OuterSumCheckProof::prove(&a, &b, &c, &z, &mut prover_challenger);

        // Verify the proof
        let mut verifier_challenger = Challenger::new();
        proof.verify(&mut verifier_challenger);

        println!("✅ 2-variable prover-verifier flow test passed!");
    }

    #[test]
    fn test_prover_verifier_flow_random() {
        // Test with random instances
        for n_vars in 1..=2 {
            let (a, b, c, z) = create_random_r1cs_instance(n_vars, Some(42 + n_vars as u64));

            // Verify the R1CS constraint holds
            assert!(
                validate_r1cs_constraint(&a, &b, &c, &z),
                "R1CS constraint should hold for random {}-variable instance",
                n_vars
            );

            // Run the prover-verifier protocol
            let mut prover_challenger = Challenger::new();
            let proof = OuterSumCheckProof::prove(&a, &b, &c, &z, &mut prover_challenger);

            // Verify the proof
            let mut verifier_challenger = Challenger::new();
            proof.verify(&mut verifier_challenger);

            println!("✅ Random {}-variable prover-verifier test passed!", n_vars);
        }
    }

    #[test]
    fn test_helper_functions() {
        // Test the helper functions work correctly

        // Test simple instance
        let (a, b, c, z) = create_simple_r1cs_instance(1);
        assert_eq!(a.dimensions(), (2, 2));
        assert_eq!(z.len(), 2);
        assert!(validate_r1cs_constraint(&a, &b, &c, &z));

        // Test random instance
        let (a, b, c, z) = create_random_r1cs_instance(1, Some(123));
        assert_eq!(a.dimensions(), (2, 2));
        assert_eq!(z.len(), 2);
        assert!(validate_r1cs_constraint(&a, &b, &c, &z));

        println!("✅ Helper function tests passed!");
    }
}
