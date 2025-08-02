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
pub use constraint::{ClosureConstraint, ConstraintPolynomial};
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
    pub fn new(
        round_proofs: Vec<SumCheckRoundProof>,
        final_claims: Vec<Fp4>,
        num_variables: usize,
    ) -> Self {
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
        C: ConstraintPolynomial,
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
        let mut current_mles: Vec<MLE<Fp4>> = mles
            .iter()
            .map(|mle| {
                // Convert MLE<F> to MLE<Fp4> for evaluation
                MLE::new(mle.coeffs().iter().map(|c| Fp4::from(c.clone())).collect())
            })
            .collect();

        let mut current_sum = claimed_sum;
        let mut challenges = Vec::with_capacity(num_vars);

        // Round-by-round sumcheck protocol
        for round in 0..num_vars {
            let mut coeffs = vec![Fp4::ZERO; constraint.degree()];
            for i in 0..1 << num_vars - round - 1 {
                let eval_0: Vec<Fp4> = current_mles
                    .iter()
                    .map(|mle| mle.coeffs()[i << 1])
                    .collect();
                let eval_1: Vec<Fp4> = current_mles
                    .iter()
                    .map(|mle| mle.coeffs()[(i << 1) | 1])
                    .collect();
                let eval_2: Vec<Fp4> = eval_0
                    .iter()
                    .zip(eval_1.iter())
                    .map(|(x0, x1)| x0.double() + *x1)
                    .collect();

                coeffs[0] += eval_0.into_iter().map(|x| x).product::<Fp4>();
                coeffs[1] += eval_1.into_iter().map(|x| x).product::<Fp4>();
                coeffs[2] += eval_2.into_iter().map(|x| x).product::<Fp4>();
            }

            let round_proof = SumCheckRoundProof::from_evaluations(&coeffs).expect("Should work");

            challenger.observe_fp4_elems(&round_proof.coeffs);

            let challenge = challenger.get_challenge();

            current_mles = current_mles
                .into_iter()
                .map(|mle| mle.fold_in_place(challenge))
                .collect();

            challenges.push(challenge);
            round_proofs.push(round_proof);
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
        let points: Vec<Fp4> = (0..evals.len()).map(|x| Fp4::from_u32(x as u32)).collect();
        let poly = UnivariatePolynomial::<Fp4>::interpolate(&points, evals)?;
        Ok(Self::new(poly.coefficients().to_vec()))
    }
}
