use std::marker::PhantomData;
use p3_field::PrimeCharacteristicRing;
use crate::utils::{Fp4, polynomial::MLE};

/// Trait for defining custom constraint polynomials in the sumcheck protocol
pub trait ConstraintPolynomial<F: PrimeCharacteristicRing + Clone> {
    /// Evaluate the constraint polynomial at a given point
    /// 
    /// # Arguments
    /// * `mles` - Slice of MLE references to evaluate
    /// * `point` - Point in the extension field to evaluate at
    /// 
    /// # Returns
    /// The evaluation result in Fp4
    fn evaluate(&self, mles: &[&MLE<F>], point: &[Fp4]) -> Fp4;
    
    /// Maximum degree of the constraint polynomial
    fn degree(&self) -> usize;
    
    /// Number of variables in the constraint
    fn num_variables(&self) -> usize;
    
    /// Number of MLE inputs required
    fn num_mles(&self) -> usize;
}

/// A constraint polynomial defined by a closure function
pub struct ClosureConstraint<F, C> 
where 
    F: PrimeCharacteristicRing + Clone,
    C: Fn(&[&MLE<F>], &[Fp4]) -> Fp4,
{
    closure: C,
    degree: usize,
    num_variables: usize,
    num_mles: usize,
    _phantom: PhantomData<F>,
}

impl<F, C> ClosureConstraint<F, C>
where 
    F: PrimeCharacteristicRing + Clone,
    C: Fn(&[&MLE<F>], &[Fp4]) -> Fp4,
{
    /// Create a new closure-based constraint polynomial
    /// 
    /// # Arguments
    /// * `closure` - Function that evaluates the constraint
    /// * `degree` - Maximum degree of the polynomial
    /// * `num_variables` - Number of variables in the constraint
    /// * `num_mles` - Number of MLE inputs required
    pub fn new(
        closure: C,
        degree: usize,
        num_variables: usize,
        num_mles: usize,
    ) -> Self {
        Self {
            closure,
            degree,
            num_variables,
            num_mles,
            _phantom: PhantomData,
        }
    }
}

impl<F, C> ConstraintPolynomial<F> for ClosureConstraint<F, C>
where 
    F: PrimeCharacteristicRing + Clone,
    C: Fn(&[&MLE<F>], &[Fp4]) -> Fp4,
{
    fn evaluate(&self, mles: &[&MLE<F>], point: &[Fp4]) -> Fp4 {
        (self.closure)(mles, point)
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

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_closure_constraint_creation() {
        let constraint = ClosureConstraint::new(
            |mles: &[&MLE<BabyBear>], point: &[Fp4]| {
                let a = mles[0].evaluate(point);
                let b = mles[1].evaluate(point);
                a * b
            },
            2, // degree
            3, // num_variables
            2, // num_mles
        );

        assert_eq!(constraint.degree(), 2);
        assert_eq!(constraint.num_variables(), 3);
        assert_eq!(constraint.num_mles(), 2);
    }

    #[test]
    fn test_constraint_evaluation() {
        // Create simple MLEs for testing
        let coeffs_a = vec![
            BabyBear::from_u32(1), BabyBear::from_u32(2),
            BabyBear::from_u32(3), BabyBear::from_u32(4),
        ];
        let coeffs_b = vec![
            BabyBear::from_u32(5), BabyBear::from_u32(6),
            BabyBear::from_u32(7), BabyBear::from_u32(8),
        ];
        
        let mle_a = MLE::new(coeffs_a);
        let mle_b = MLE::new(coeffs_b);
        let mles = vec![&mle_a, &mle_b];

        // Create multiplication constraint: a * b
        let constraint = ClosureConstraint::new(
            |mles: &[&MLE<BabyBear>], point: &[Fp4]| {
                let a = mles[0].evaluate(point);
                let b = mles[1].evaluate(point);
                a * b
            },
            2, // degree
            2, // num_variables
            2, // num_mles
        );

        // Test evaluation at a point
        let point = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
        let result = constraint.evaluate(&mles, &point);
        
        // Verify the result is the product of individual evaluations
        let a_eval = mle_a.evaluate(&point);
        let b_eval = mle_b.evaluate(&point);
        let expected = a_eval * b_eval;
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_complex_constraint() {
        // Test a more complex constraint: (a + b) * c - d
        let coeffs_a = vec![BabyBear::from_u32(1), BabyBear::from_u32(2)];
        let coeffs_b = vec![BabyBear::from_u32(3), BabyBear::from_u32(4)];
        let coeffs_c = vec![BabyBear::from_u32(5), BabyBear::from_u32(6)];
        let coeffs_d = vec![BabyBear::from_u32(7), BabyBear::from_u32(8)];
        
        let mle_a = MLE::new(coeffs_a);
        let mle_b = MLE::new(coeffs_b);
        let mle_c = MLE::new(coeffs_c);
        let mle_d = MLE::new(coeffs_d);
        let mles = vec![&mle_a, &mle_b, &mle_c, &mle_d];

        let constraint = ClosureConstraint::new(
            |mles: &[&MLE<BabyBear>], point: &[Fp4]| {
                let a = mles[0].evaluate(point);
                let b = mles[1].evaluate(point);
                let c = mles[2].evaluate(point);
                let d = mles[3].evaluate(point);
                (a + b) * c - d
            },
            2, // degree
            1, // num_variables
            4, // num_mles
        );

        let point = vec![Fp4::from_u32(7)];
        let result = constraint.evaluate(&mles, &point);
        
        // Manually compute expected result
        let a_eval = mle_a.evaluate(&point);
        let b_eval = mle_b.evaluate(&point);
        let c_eval = mle_c.evaluate(&point);
        let d_eval = mle_d.evaluate(&point);
        let expected = (a_eval + b_eval) * c_eval - d_eval;
        
        assert_eq!(result, expected);
    }
}