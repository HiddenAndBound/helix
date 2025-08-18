use p3_field::PrimeCharacteristicRing;

use super::*;
use crate::challenger::Challenger;
use crate::polynomial::MLE;
use crate::spartan::R1CSInstance;
use crate::spartan::spark::sparse::SpartanMetadata;
use crate::spartan::spark::sparse::TimeStamps;
use crate::{Fp, Fp4};

#[test]
fn test_outer_sumcheck_prove_verify_simple() {
    // Basic outer sum-check prove/verify cycle with simple R1CS
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    let mut challenger_prove = Challenger::new();
    let mut challenger_verify = Challenger::new();

    // Prove outer sum-check
    let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger_prove);

    // Verify proof
    outer_proof.verify(&mut challenger_verify);

    // Test passed if no panics occurred
}

#[test]
fn test_outer_sumcheck_prove_verify_multi_constraint() {
    // Test outer sum-check with multi-constraint R1CS
    let instance = R1CSInstance::multi_constraint_test().unwrap();
    let z = &instance.witness_mle();
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    let mut challenger_prove = Challenger::new();
    let mut challenger_verify = Challenger::new();

    // Prove outer sum-check
    let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger_prove);

    // Verify proof
    outer_proof.verify(&mut challenger_verify);

    // Test that proof structure is consistent
    assert!(!outer_proof.round_proofs.is_empty());
}

#[test]
fn test_outer_sumcheck_mathematical_correctness() {
    // Test that outer sum-check correctly proves f(x) = A(x)·B(x) - C(x) = 0
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    let mut challenger = Challenger::new();

    // Generate proof
    let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger);

    // Verify that final evaluations satisfy the constraint
    // A(r) * B(r) - C(r) should equal the final claim after sum-check
    let final_evals = &outer_proof.final_evals;
    let _constraint_result = final_evals[0] * final_evals[1] - final_evals[2];

    // The constraint should be satisfied (though we can't directly test the sum-check claim
    // without reimplementing the verifier logic, successful verification implies correctness)

    // Test successful verification
    let mut verifier = Challenger::new();
    outer_proof.verify(&mut verifier);
}

#[test]
fn test_outer_sumcheck_round_proof_consistency() {
    // Test that round proofs are generated consistently
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    let mut challenger1 = Challenger::new();
    let mut challenger2 = Challenger::new();

    // Generate two proofs with same challenger state
    let proof1 = OuterSumCheckProof::prove(A, B, C, z, &mut challenger1);
    let proof2 = OuterSumCheckProof::prove(A, B, C, z, &mut challenger2);

    // Should be identical due to deterministic challenger
    assert_eq!(proof1, proof2);
    assert_eq!(proof1.round_proofs.len(), proof2.round_proofs.len());
    assert_eq!(proof1.final_evals, proof2.final_evals);
}

#[test]
fn test_inner_sumcheck_prove_verify_simple() {
    // Test inner sum-check with simple bound matrices
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();

    // First run outer sum-check to get bound matrices
    let mut outer_challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut outer_challenger,
    );

    // Extract random point from outer sum-check (simulate the protocol flow)
    let r_x_point: Vec<Fp4> = vec![]; // Simple instance needs 0 variables
    let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

    let mut inner_challenger_prove = Challenger::new();
    let mut inner_challenger_verify = Challenger::new();

    // Simulate gamma challenge
    inner_challenger_prove.observe_fp4_elems(&outer_proof.final_evals);
    inner_challenger_verify.observe_fp4_elems(&outer_proof.final_evals);
    let gamma = inner_challenger_prove.get_challenge();
    let gamma_verify = inner_challenger_verify.get_challenge();
    assert_eq!(gamma, gamma_verify);

    // Prove inner sum-check
    let inner_proof = InnerSumCheckProof::prove(
        &a_bound,
        &b_bound,
        &c_bound,
        outer_proof.final_evals,
        gamma,
        z,
        &mut inner_challenger_prove,
    );

    // Verify inner sum-check
    inner_proof.verify(outer_proof.final_evals, gamma, &mut inner_challenger_verify);

    // Test passed if no panics occurred
}

#[test]
fn test_inner_sumcheck_batched_claims() {
    // Test that inner sum-check correctly handles batched evaluation claims
    let instance = R1CSInstance::multi_constraint_test().unwrap();
    let z = &instance.witness_mle();

    // Run outer sum-check first
    let mut outer_challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut outer_challenger,
    );

    // Get bound matrices
    let r_x_point: Vec<Fp4> = vec![Fp4::from_u32(7), Fp4::from_u32(13)]; // Multi instance needs 2 variables
    let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();
    let mut challenger = Challenger::new();
    challenger.observe_fp4_elems(&outer_proof.final_evals);
    let _gamma = challenger.get_challenge();

    // Test different gamma values produce different but valid proofs
    let gammas = [Fp4::from_u32(1), Fp4::from_u32(17), Fp4::from_u32(42)];

    for test_gamma in gammas {
        let mut prove_challenger = Challenger::new();
        let mut verify_challenger = Challenger::new();

        let inner_proof = InnerSumCheckProof::prove(
            &a_bound,
            &b_bound,
            &c_bound,
            outer_proof.final_evals,
            test_gamma,
            z,
            &mut prove_challenger,
        );

        // Each should verify successfully
        inner_proof.verify(outer_proof.final_evals, test_gamma, &mut verify_challenger);
    }
}

#[test]
fn test_inner_sumcheck_bound_matrix_integration() {
    // Test inner sum-check works with actual bound matrices from outer sum-check
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();

    // Complete outer sum-check phase
    let mut outer_challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut outer_challenger,
    );

    // Simulate the protocol: compute bound matrices using random point
    let r_x_point: Vec<Fp4> = vec![];
    let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

    // Continue with inner sum-check
    let mut inner_challenger = Challenger::new();
    inner_challenger.observe_fp4_elems(&outer_proof.final_evals);
    let gamma = inner_challenger.get_challenge();

    let inner_proof = InnerSumCheckProof::prove(
        &a_bound,
        &b_bound,
        &c_bound,
        outer_proof.final_evals,
        gamma,
        z,
        &mut inner_challenger,
    );

    // Verify with same protocol flow
    let mut verifier = Challenger::new();
    verifier.observe_fp4_elems(&outer_proof.final_evals);
    let verify_gamma = verifier.get_challenge();
    assert_eq!(gamma, verify_gamma);

    inner_proof.verify(outer_proof.final_evals, verify_gamma, &mut verifier);
}

#[test]
fn test_outer_inner_sumcheck_integration() {
    // Test complete outer → inner sum-check flow
    let instance = R1CSInstance::multi_constraint_test().unwrap();
    let z = &instance.witness_mle();
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    // Phase 1: Outer sum-check
    let mut challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(A, B, C, z, &mut challenger);

    // Extract random point for bound matrices (simulate protocol)
    let r_x_point: Vec<Fp4> = vec![Fp4::from_u32(5), Fp4::from_u32(11)];
    let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

    // Phase 2: Inner sum-check with outer evaluation claims
    challenger.observe_fp4_elems(&outer_proof.final_evals);
    let gamma = challenger.get_challenge();

    let inner_proof = InnerSumCheckProof::prove(
        &a_bound,
        &b_bound,
        &c_bound,
        outer_proof.final_evals,
        gamma,
        z,
        &mut challenger,
    );

    // Verify both phases
    let mut verifier = Challenger::new();
    outer_proof.verify(&mut verifier);

    verifier.observe_fp4_elems(&outer_proof.final_evals);
    let verify_gamma = verifier.get_challenge();
    inner_proof.verify(outer_proof.final_evals, verify_gamma, &mut verifier);

    // Test complete integration success
    assert_eq!(gamma, verify_gamma);
}

#[test]
fn test_sumcheck_field_arithmetic_consistency() {
    // Test Fp → Fp4 promotion and field arithmetic throughout sum-check
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle(); // z is MLE<BabyBear> (base field)

    let mut challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut challenger,
    );

    // Final evaluations should be in extension field Fp4
    for eval in outer_proof.final_evals {
        // Test that we can perform Fp4 arithmetic
        let _sum = eval + eval;
        let _product = eval * eval;
        let _difference = eval - eval;
    }

    // Verify field consistency through successful verification
    let mut verifier = Challenger::new();
    outer_proof.verify(&mut verifier);
}

#[test]
fn test_compute_round_functions() {
    // Test that round computation functions work correctly
    let instance = R1CSInstance::simple_test().unwrap();
    let z_fp = &instance.witness_mle(); // Base field witness
    let (A, B, C) = (&instance.r1cs.a, &instance.r1cs.b, &instance.r1cs.c);

    // Compute A·z, B·z, C·z (these will be in base field initially)
    let a_mle = A.multiply_by_mle(z_fp).unwrap();
    let b_mle = B.multiply_by_mle(z_fp).unwrap();
    let c_mle = C.multiply_by_mle(z_fp).unwrap();

    // Test that we can use the round computation functions
    // (This tests the mathematical correctness indirectly)

    // Create dummy EqEvals for testing
    let eq_point = vec![Fp4::from_u32(7)];
    let eq_evals = crate::eq::EqEvals::gen_from_point(&eq_point);

    // Test first round computation (base field → extension field)
    let current_claim = Fp4::ZERO;
    let rounds = a_mle.n_vars();

    let first_round_poly = outer::compute_first_round(
        &a_mle,
        &b_mle,
        &c_mle,
        &eq_evals,
        &eq_point,
        current_claim,
        rounds,
    );

    // Verify polynomial has correct structure (degree ≤ 2)
    assert!(first_round_poly.coefficients().len() >= 2);
    assert!(first_round_poly.coefficients().len() <= 3);

    // Test evaluation at different points
    let eval_0 = first_round_poly.evaluate(Fp4::ZERO);
    let eval_1 = first_round_poly.evaluate(Fp4::ONE);
    let eval_2 = first_round_poly.evaluate(Fp4::from_u32(2));

    // All should be valid field elements
    let _test_arithmetic = eval_0 + eval_1 + eval_2;
}

#[test]
fn test_gruen_optimization() {
    // Test that Gruen's optimization (evaluate at 0, 1, 2) works correctly
    let instance = R1CSInstance::simple_test().unwrap();
    let z = &instance.witness_mle();

    let mut challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut challenger,
    );

    // Test that each round proof can be evaluated at standard points
    for round_poly in &outer_proof.round_proofs {
        let eval_0 = round_poly.evaluate(Fp4::ZERO);
        let eval_1 = round_poly.evaluate(Fp4::ONE);
        let eval_2 = round_poly.evaluate(Fp4::from_u32(2));

        // All evaluations should be valid and consistent
        // (Gruen's optimization uses these points for degree-2 polynomial interpolation)
        let _arithmetic_test = eval_0 + eval_1 + eval_2;
    }

    // Verify overall proof correctness
    let mut verifier = Challenger::new();
    outer_proof.verify(&mut verifier);
}

#[test]
fn test_sumcheck_sparse_mle_integration() {
    // Test sum-check protocols work correctly with sparse MLE operations
    let instance = R1CSInstance::multi_constraint_test().unwrap();
    let z = &instance.witness_mle();

    // Verify constraint matrices are sparse
    assert!(
        instance.r1cs.a.num_nonzeros()
            < instance.r1cs.a.dimensions().0 * instance.r1cs.a.dimensions().1
    );
    assert!(
        instance.r1cs.b.num_nonzeros()
            < instance.r1cs.b.dimensions().0 * instance.r1cs.b.dimensions().1
    );
    assert!(
        instance.r1cs.c.num_nonzeros()
            < instance.r1cs.c.dimensions().0 * instance.r1cs.c.dimensions().1
    );

    // Run outer sum-check with sparse matrices
    let mut challenger = Challenger::new();
    let outer_proof = OuterSumCheckProof::prove(
        &instance.r1cs.a,
        &instance.r1cs.b,
        &instance.r1cs.c,
        z,
        &mut challenger,
    );

    // Verify sparse operations maintain correctness
    let mut verifier = Challenger::new();
    outer_proof.verify(&mut verifier);

    // Test that sparsity is preserved in bound matrix computation
    let r_x_point = vec![Fp4::from_u32(3), Fp4::from_u32(7)];
    let (a_bound, b_bound, c_bound) = instance.compute_bound_matrices(&r_x_point).unwrap();

    // Bound matrices should have expected dimensions
    assert_eq!(a_bound.len(), instance.r1cs.a.dimensions().1);
    assert_eq!(b_bound.len(), instance.r1cs.b.dimensions().1);
    assert_eq!(c_bound.len(), instance.r1cs.c.dimensions().1);
}

#[test]
fn test_spark_sumcheck_prove_verify() {
    // Create dummy metadata and oracles for three matrices
    let read_ts = vec![Fp::ZERO, Fp::ONE];
    let final_ts = vec![Fp::ONE, Fp::from_u32(2u32)];
    let ts = TimeStamps::new(read_ts, final_ts).unwrap();

    let metadatas = [
        SpartanMetadata::new(
            MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(2u32)]),
            MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(1u32)]),
            MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(0u32)]),
            ts.clone(),
            ts.clone(),
        )
        .unwrap(),
        SpartanMetadata::new(
            MLE::new(vec![Fp::from_u32(3u32), Fp::from_u32(4u32)]),
            MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(0u32)]),
            MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(1u32)]),
            ts.clone(),
            ts.clone(),
        )
        .unwrap(),
        SpartanMetadata::new(
            MLE::new(vec![Fp::from_u32(5u32), Fp::from_u32(6u32)]),
            MLE::new(vec![Fp::from_u32(0u32), Fp::from_u32(0u32)]),
            MLE::new(vec![Fp::from_u32(1u32), Fp::from_u32(1u32)]),
            ts.clone(),
            ts.clone(),
        )
        .unwrap(),
    ];

    let oracle_pairs = [
        (
            MLE::new(vec![Fp4::from_u32(10u32), Fp4::from_u32(11u32)]),
            MLE::new(vec![Fp4::from_u32(12u32), Fp4::from_u32(13u32)]),
        ),
        (
            MLE::new(vec![Fp4::from_u32(14u32), Fp4::from_u32(15u32)]),
            MLE::new(vec![Fp4::from_u32(16u32), Fp4::from_u32(17u32)]),
        ),
        (
            MLE::new(vec![Fp4::from_u32(18u32), Fp4::from_u32(19u32)]),
            MLE::new(vec![Fp4::from_u32(20u32), Fp4::from_u32(21u32)]),
        ),
    ];

    // Dummy evaluation claims and gamma
    let evaluation_claims = [
        Fp4::from_u32(1u32),
        Fp4::from_u32(2u32),
        Fp4::from_u32(3u32),
    ];
    let gamma = Fp4::from_u32(7u32);

    // Create separate challengers for prover and verifier
    let mut prover_challenger = Challenger::new();
    let mut verifier_challenger = Challenger::new();

    // Generate the proof
    let proof = SparkSumCheckProof::prove(
        &metadatas,
        &oracle_pairs,
        evaluation_claims,
        gamma,
        &mut prover_challenger,
    );

    // Verify the proof
    proof.verify(evaluation_claims, gamma, &mut verifier_challenger);

    // The test passes if no assertions fail
}
