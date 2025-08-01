//! Sumcheck protocol implementation
//!
//! This module provides the core infrastructure for the sumcheck protocol,
//! including constraint polynomials, univariate polynomials, and error handling.

use p3_field::PrimeCharacteristicRing;

use crate::{
    Fp, Fp4,
    challenger::{self, Challenger},
    eq::EqEvals,
    polynomial::MLE,
};

// Re-export core components
pub mod constraint;
pub mod error;
pub mod univariate;

// Re-export commonly used items
pub use constraint::{ConstraintPolynomial, ClosureConstraint};
pub use error::SumCheckError;
pub use univariate::UnivariatePolynomial;

/// Struct holding a complete sumcheck proof
pub struct SumCheckProof {
    /// Round proofs for each variable
    round_proofs: Vec<SumCheckRoundProof>,
    /// Final evaluation claims
    final_claims: Vec<Fp4>,
    /// Number of variables in the sumcheck
    num_variables: usize,
}

impl SumCheckProof {
    /// Creates a new SumCheckProof
    pub fn new(round_proofs: Vec<SumCheckRoundProof>, final_claims: Vec<Fp4>, num_variables: usize) -> Self {
        Self {
            round_proofs,
            final_claims,
            num_variables,
        }
    }

    /// Proves a sumcheck instance for the given MLEs and constraint
    pub fn prove<F, C>(
        mles: &[MLE<F>],
        constraint: &C,
        claimed_sum: Fp4,
        challenger: &mut Challenger,
    ) -> Result<Self, SumCheckError>
    where
        F: PrimeCharacteristicRing + Clone,
        Fp4: From<F>,
        C: ConstraintPolynomial<F>,
    {
        // Validate inputs
        if mles.len() != constraint.num_mles() {
            return Err(SumCheckError::InvalidNumMles {
                expected: constraint.num_mles(),
                actual: mles.len(),
            });
        }

        let num_vars = mles[0].n_vars();
        if num_vars != constraint.num_variables() {
            return Err(SumCheckError::VariableCountMismatch {
                expected: constraint.num_variables(),
                actual: num_vars,
            });
        }

        for mle in mles {
            if mle.n_vars() != num_vars {
                return Err(SumCheckError::VariableCountMismatch {
                    expected: num_vars,
                    actual: mle.n_vars(),
                });
            }
        }

        let mut round_proofs = Vec::with_capacity(num_vars);
        let mut current_mles: Vec<MLE<Fp4>> = mles.iter().map(|mle| {
            // Convert MLE<F> to MLE<Fp4> for evaluation
            MLE::new(mle.coeffs().iter().map(|c| Fp4::from(c.clone())).collect())
        }).collect();

        let mut current_sum = claimed_sum;
        let mut challenges = Vec::with_capacity(num_vars);

        // Round-by-round sumcheck protocol
        for round in 0..num_vars {
            // Evaluate constraint over hypercube points for current round
            let mut evaluations = Vec::with_capacity(constraint.degree() + 1);
            for x in 0..=constraint.degree() {
                let point = Fp4::from_u32(x as u32);
                challenges.push(point);
                // Convert MLE<Fp4> to MLE<F> references by reconstructing the references
                let mle_refs: Vec<&MLE<F>> = current_mles.iter()
                    .map(|mle_fp4| {
                        // Convert MLE<Fp4> to MLE<F> by reconstructing the reference
                        unsafe { &*(mle_fp4 as *const MLE<Fp4> as *const MLE<F>) }
                    })
                    .collect();
                let eval = constraint.evaluate(&mle_refs, &challenges);
                evaluations.push(eval);
                challenges.pop();
            }

            // Interpolate univariate polynomial
            let points: Vec<_> = (0..=constraint.degree())
                .map(|x| Fp4::from_u32(x as u32))
                .collect();
            let poly = <crate::sumcheck::univariate::UnivariatePolynomial<Fp4>>::interpolate(&points, &evaluations)?;
            let round_poly = SumCheckRoundProof::new(poly.coefficients().to_vec());
            round_proofs.push(round_poly);

            // Get challenge from verifier
            let challenge = challenger.get_challenge();
            challenges.push(challenge);

            // Fold all MLEs with the challenge
            current_mles = current_mles.into_iter()
                .map(|mle| mle.fold_in_place(challenge))
                .collect();
        }

        // Final evaluation claims
        let final_claims = challenges;

        Ok(Self {
            round_proofs,
            final_claims,
            num_variables: num_vars,
        })
    }
}

pub struct SumCheckRoundProof {
    coeffs: Vec<Fp4>,
}

impl SumCheckRoundProof {
    pub fn new(coeffs: Vec<Fp4>) -> Self {
        Self { coeffs }
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn eval(&self, point: Fp4) -> Fp4 {
        let mut eval = Fp4::ZERO;
        for &coeff in &self.coeffs {
            eval = coeff + eval * point
        }
        eval
    }

    /// Creates a round proof from evaluations by interpolating a univariate polynomial
    pub fn from_evaluations(evals: &[Fp4]) -> Result<Self, SumCheckError> {
        let points: Vec<_> = (0..evals.len())
            .map(|x| Fp4::from_u32(x as u32))
            .collect();
        let poly = UnivariatePolynomial::interpolate(&points, evals)?;
        Ok(Self::new(poly.coefficients().to_vec()))
    }
}

/// Helper function to create a constraint polynomial from a closure
pub fn constraint_from_closure<F>(
    closure: impl Fn(&[&MLE<F>], &[Fp4]) -> Fp4 + Clone + Send + Sync,
    degree: usize,
    num_vars: usize,
    num_mles: usize,
) -> ClosureConstraint<F, impl Fn(&[&MLE<F>], &[Fp4]) -> Fp4 + Clone + Send + Sync>
where
    F: PrimeCharacteristicRing + Clone,
{
    ClosureConstraint::new(closure, degree, num_vars, num_mles)
}


/// Helper function to interpolate a univariate polynomial from points and values
pub fn interpolate_univariate(
    points: &[Fp4],
    values: &[Fp4],
) -> Result<UnivariatePolynomial<Fp4>, SumCheckError> {
    <crate::sumcheck::univariate::UnivariatePolynomial<Fp4>>::interpolate(points, values)
        .map_err(|e| SumCheckError::InterpolationError(e.to_string()))
}