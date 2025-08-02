use crate::utils::{Fp4, polynomial::MLE};
use p3_field::PrimeCharacteristicRing;
use std::marker::PhantomData;

/// Trait for defining custom constraint polynomials in the sumcheck protocol
pub trait ConstraintPolynomial {
    /// Evaluate the constraint polynomial at a given point
    ///
    /// # Arguments
    /// * `mles` - Slice of MLE references to evaluate
    /// * `point` - Point in the extension field to evaluate at
    ///
    /// # Returns
    /// The evaluation result in Fp4
    fn evaluate(&self, poly_evals: &[Fp4]) -> Fp4;

    /// Maximum degree of the constraint polynomial
    fn degree(&self) -> usize;

    /// Number of variables in the constraint
    fn num_variables(&self) -> usize;

    /// Number of MLE inputs required
    fn num_mles(&self) -> usize;
}

/// A constraint polynomial defined by a closure function
pub struct ClosureConstraint<C>
where
    C: Fn(&[Fp4]) -> Fp4,
{
    closure: C,
    degree: usize,
    num_variables: usize,
    num_mles: usize,
}

impl<C> ClosureConstraint<C>
where
    C: Fn(&[Fp4]) -> Fp4,
{
    /// Create a new closure-based constraint polynomial
    ///
    /// # Arguments
    /// * `closure` - Function that evaluates the constraint
    /// * `degree` - Maximum degree of the polynomial
    /// * `num_variables` - Number of variables in the constraint
    /// * `num_mles` - Number of MLE inputs required
    pub fn new(closure: C, degree: usize, num_variables: usize, num_mles: usize) -> Self {
        Self {
            closure,
            degree,
            num_variables,
            num_mles,
        }
    }
}

impl<C> ConstraintPolynomial for ClosureConstraint<C>
where
    C: Fn(&[Fp4]) -> Fp4,
{
    fn evaluate(&self, poly_evals: &[Fp4]) -> Fp4 {
        (self.closure)(poly_evals)
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }

    fn num_mles(&self) -> usize {
        self.num_mles
    }
}
