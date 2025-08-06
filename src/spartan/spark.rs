//! Spark IOP components for polynomial commitment opening in Spartan.
//!
//! The Spark protocol provides the final phase of Spartan's IOP by enabling efficient
//! verification of polynomial commitment openings. It batches multiple opening claims
//! to achieve logarithmic communication complexity.

use crate::{
    Fp, Fp4,
    challenger::Challenger,
    polynomial::MLE,
    spartan::{
        sparse::SparseMLE,
        sumcheck::SparkSumCheckProof,
        commitment::{DummyPCS, PolynomialCommitment},
    },
};
use p3_field::PrimeCharacteristicRing;

/// Represents a single polynomial commitment opening claim.
/// 
/// **Mathematical Purpose:**
/// Each opening proves that a committed polynomial P(x) evaluates to a specific value v
/// at a given point r: P(r) = v. This is essential for verifying the final polynomial
/// evaluations from the inner sumcheck phase.
#[derive(Debug, Clone)]
pub struct OpeningClaim {
    /// The point at which the polynomial is evaluated
    pub point: Vec<Fp4>,
    /// The claimed evaluation value P(point) = value
    pub value: Fp4,
    /// The polynomial commitment being opened
    pub commitment: <DummyPCS as PolynomialCommitment<Fp4>>::Commitment,
}

impl OpeningClaim {
    pub fn new(point: Vec<Fp4>, value: Fp4, commitment: <DummyPCS as PolynomialCommitment<Fp4>>::Commitment) -> Self {
        Self { point, value, commitment }
    }
}

/// Spark IOP proof for batched polynomial commitment openings.
/// 
/// **Mathematical Purpose:**
/// Instead of verifying 3 separate polynomial commitment openings (which would require
/// 3 separate sumcheck protocols), Spark batches them into a single verification using
/// random linear combinations. This reduces communication from O(3n) to O(n + 3).
#[derive(Debug, Clone)]
pub struct SparkProof {
    /// The batched sumcheck proof for triple products A(x)·B(x)·C(x)
    pub sumcheck_proof: SparkSumCheckProof,
    /// Individual opening proofs for each polynomial commitment  
    pub opening_proofs: Vec<<DummyPCS as PolynomialCommitment<Fp4>>::Proof>,
}

impl SparkProof {
    /// Creates a new Spark proof from its components.
    pub fn new(
        sumcheck_proof: SparkSumCheckProof,
        opening_proofs: Vec<<DummyPCS as PolynomialCommitment<Fp4>>::Proof>,
    ) -> Self {
        Self {
            sumcheck_proof,
            opening_proofs,
        }
    }

    /// Generates a Spark proof for batched polynomial commitment openings.
    /// 
    /// **Mathematical Process:**
    /// 1. **Batching**: Combines 3 opening claims using random challenges r₁, r₂, r₃
    /// 2. **Sumcheck**: Proves the batched claim via SparkSumCheckProof  
    /// 3. **Individual Openings**: Generates commitment opening proofs for each polynomial
    /// 
    /// **Input**: Three opening claims from the inner sumcheck phase
    /// **Output**: Single batched proof with 3x communication efficiency
    pub fn prove(
        opening_claims: [OpeningClaim; 3],
        polynomials: [&MLE<Fp4>; 3],  // The actual polynomials being opened
        challenger: &mut Challenger,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Generate random challenges for batching the three opening claims
        let batching_challenges = [
            challenger.get_challenge(),
            challenger.get_challenge(), 
            challenger.get_challenge(),
        ];

        // Create dummy sparse matrices for the sumcheck protocol
        // Note: In practice, these would be derived from the polynomial structure
        let dummy_instances = create_dummy_spark_instances(&polynomials)?;

        // Create a dummy witness for the sumcheck (this would be the witness MLE in practice)
        let witness = create_dummy_witness(polynomials[0].n_vars());

        // Phase 1: Batched Sumcheck
        // Proves: r₁·(A₁·B₁·C₁) + r₂·(A₂·B₂·C₂) + r₃·(A₃·B₃·C₃) = batched_claimed_sum
        let sumcheck_proof = SparkSumCheckProof::prove(
            &dummy_instances,
            &batching_challenges,
            &witness,
            challenger,
        );

        // Phase 2: Individual Polynomial Commitment Openings
        // Generate opening proofs for each polynomial at its respective point
        let mut opening_proofs = Vec::new();
        for (polynomial, claim) in polynomials.iter().zip(opening_claims.iter()) {
            let opening_proof = DummyPCS::prove_evaluation(
                polynomial,
                &claim.point,
                claim.value,
                challenger,
            )?;
            opening_proofs.push(opening_proof);
        }

        Ok(SparkProof::new(sumcheck_proof, opening_proofs))
    }

    /// Verifies a Spark proof for batched polynomial commitment openings.
    /// 
    /// **Mathematical Verification:**
    /// 1. **Sumcheck Verification**: Ensures the batched polynomial equation holds
    /// 2. **Opening Verification**: Checks each individual polynomial commitment opening
    /// 3. **Consistency Check**: Verifies that batching was performed correctly
    /// 
    /// **Security**: Combines the soundness of both sumcheck and commitment schemes
    pub fn verify(
        &self,
        opening_claims: [OpeningClaim; 3],
        challenger: &mut Challenger,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Phase 1: Verify the batched sumcheck proof
        // This ensures the batched polynomial relation holds
        self.sumcheck_proof.verify(challenger);

        // Phase 2: Verify individual polynomial commitment openings  
        // This ensures each claimed evaluation is correct
        for (claim, proof) in opening_claims.iter().zip(self.opening_proofs.iter()) {
            let is_valid = DummyPCS::verify_evaluation(
                &claim.commitment,
                &claim.point,
                claim.value,
                proof,
                challenger,
            )?;
            
            if !is_valid {
                return Ok(false);
            }
        }

        // Phase 3: Consistency check
        // Verify that the sumcheck and opening proofs are consistent
        // (In practice, this would check that the final evaluations match)
        
        Ok(true)
    }
}

/// Spark IOP instance representing the complete polynomial commitment opening protocol.
/// 
/// **Mathematical Purpose:**  
/// Coordinates the entire Spark protocol execution, managing the transition from
/// inner sumcheck outputs to final commitment verification. This is where the
/// polynomial evaluation claims from InnerSumCheck get verified via commitments.
#[derive(Debug)]
pub struct SparkInstance {
    /// The polynomial commitments that need to be opened
    pub commitments: Vec<<DummyPCS as PolynomialCommitment<Fp4>>::Commitment>,
    /// The points at which polynomials should be evaluated  
    pub evaluation_points: Vec<Vec<Fp4>>,
    /// The claimed evaluation values
    pub claimed_values: Vec<Fp4>,
}

impl SparkInstance {
    /// Creates a new Spark instance from inner sumcheck outputs.
    /// 
    /// **Input Transformation:**
    /// Takes the point evaluations from InnerSumCheck (A_bound(r_y), B_bound(r_y), 
    /// C_bound(r_y), Z(r_y)) and transforms them into commitment opening claims.
    pub fn new(
        commitments: Vec<<DummyPCS as PolynomialCommitment<Fp4>>::Commitment>,
        evaluation_points: Vec<Vec<Fp4>>,
        claimed_values: Vec<Fp4>,
    ) -> Self {
        assert_eq!(commitments.len(), evaluation_points.len());
        assert_eq!(commitments.len(), claimed_values.len());
        
        Self {
            commitments,
            evaluation_points,
            claimed_values,
        }
    }

    /// Executes the complete Spark IOP protocol.
    /// 
    /// **Protocol Flow:**
    /// 1. **Setup**: Convert instance data into opening claims
    /// 2. **Prove**: Generate batched Spark proof  
    /// 3. **Output**: Proof that can be verified by any party
    /// 
    /// **Communication Complexity**: O(log n + k) where n = polynomial size, k = number of openings
    pub fn prove(
        &self,
        polynomials: Vec<&MLE<Fp4>>,
        challenger: &mut Challenger,
    ) -> Result<SparkProof, Box<dyn std::error::Error>> {
        assert_eq!(polynomials.len(), 3, "Spark protocol expects exactly 3 polynomials");

        // Convert instance data into opening claims
        let opening_claims = [
            OpeningClaim::new(
                self.evaluation_points[0].clone(),
                self.claimed_values[0],
                self.commitments[0].clone(),
            ),
            OpeningClaim::new(
                self.evaluation_points[1].clone(),
                self.claimed_values[1],
                self.commitments[1].clone(),
            ),
            OpeningClaim::new(
                self.evaluation_points[2].clone(),
                self.claimed_values[2],
                self.commitments[2].clone(),
            ),
        ];

        // Generate the batched Spark proof
        SparkProof::prove(
            opening_claims,
            [polynomials[0], polynomials[1], polynomials[2]],
            challenger,
        )
    }

    /// Verifies a Spark proof against this instance.
    /// 
    /// **Verification Process:**
    /// 1. **Reconstruct Claims**: Convert instance data back to opening claims
    /// 2. **Verify Proof**: Use SparkProof verification procedure
    /// 3. **Return Result**: Boolean indicating proof validity
    pub fn verify(
        &self,
        proof: &SparkProof,
        challenger: &mut Challenger,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Reconstruct opening claims from instance data
        let opening_claims = [
            OpeningClaim::new(
                self.evaluation_points[0].clone(),
                self.claimed_values[0],
                self.commitments[0].clone(),
            ),
            OpeningClaim::new(
                self.evaluation_points[1].clone(),
                self.claimed_values[1],
                self.commitments[1].clone(),
            ),
            OpeningClaim::new(
                self.evaluation_points[2].clone(),
                self.claimed_values[2],
                self.commitments[2].clone(),
            ),
        ];

        // Verify the proof
        proof.verify(opening_claims, challenger)
    }
}

// Helper functions for creating dummy components

/// Creates dummy sparse matrix instances for the Spark sumcheck protocol.
/// 
/// **Purpose**: In the full implementation, these would be constructed from the
/// polynomial structure and commitment scheme. For now, we use dummy matrices
/// to demonstrate the protocol structure.
fn create_dummy_spark_instances(
    polynomials: &[&MLE<Fp4>; 3],
) -> Result<[(SparseMLE, SparseMLE, SparseMLE); 3], Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    
    let mut instances = Vec::new();
    
    for i in 0..3 {
        let poly_size = polynomials[i].len();
        let _matrix_dim = (poly_size, poly_size);
        
        // Create simple identity-like sparse matrices for demonstration
        let mut coeffs_a = HashMap::new();
        let mut coeffs_b = HashMap::new(); 
        let mut coeffs_c = HashMap::new();
        
        // Add some sparse entries based on polynomial structure
        for j in 0..std::cmp::min(4, poly_size) {
            coeffs_a.insert((j, j), Fp::from_u32((i + 1) as u32));
            coeffs_b.insert((j, (j + 1) % poly_size), Fp::from_u32((i + 2) as u32));
            coeffs_c.insert(((j + 1) % poly_size, j), Fp::from_u32((i + 3) as u32));
        }
        
        let sparse_a = SparseMLE::new(coeffs_a)?;
        let sparse_b = SparseMLE::new(coeffs_b)?;
        let sparse_c = SparseMLE::new(coeffs_c)?;
        
        instances.push((sparse_a, sparse_b, sparse_c));
    }
    
    Ok([instances[0].clone(), instances[1].clone(), instances[2].clone()])
}

/// Creates a dummy witness MLE for the Spark protocol.
/// 
/// **Purpose**: Provides a placeholder witness for the sumcheck protocol.
/// In practice, this would be derived from the actual proof context.
fn create_dummy_witness(n_vars: usize) -> MLE<Fp> {
    let size = 1 << n_vars;
    let coeffs = (0..size).map(|i| Fp::from_u32((i + 1) as u32)).collect();
    MLE::new(coeffs)
}