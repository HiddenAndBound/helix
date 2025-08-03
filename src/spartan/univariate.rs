use super::error::{SumCheckError, SumCheckResult};
use crate::Fp4;
use std::fmt;

/// Univariate polynomial for sumcheck round polynomials of degree 1.
///
/// Represents f(x) = ax + b where coefficients are stored as [a, b].
/// Used in Spartan's sumcheck protocol for round polynomial evaluations.
#[derive(Debug, Clone, PartialEq)]
pub struct UnivariatePoly {
    /// Coefficients [a, b] representing f(x) = ax + b
    coeffs: Vec<Fp4>,
}

impl UnivariatePoly {
    /// Creates a new degree-1 univariate polynomial from coefficients [a, b].
    ///
    /// # Arguments
    /// * `coeffs` - Coefficient vector [a, b] for polynomial f(x) = ax + b
    ///
    /// # Returns
    /// * `SumCheckResult<Self>` - The polynomial or validation error
    ///
    /// # Errors
    /// * Returns `ValidationError` if coeffs length is not exactly 2
    pub fn new(coeffs: Vec<Fp4>) -> SumCheckResult<Self> {
        if coeffs.len() != 2 {
            return Err(SumCheckError::ValidationError(format!(
                "Degree-1 polynomial requires exactly 2 coefficients, got {}",
                coeffs.len()
            )));
        }

        Ok(UnivariatePoly { coeffs })
    }

    /// Creates a new degree-1 polynomial from coefficient values a and b.
    ///
    /// # Arguments
    /// * `a` - Linear coefficient (slope)
    /// * `b` - Constant coefficient (y-intercept)
    ///
    /// # Returns
    /// * Polynomial representing f(x) = ax + b
    pub fn from_coeffs(a: Fp4, b: Fp4) -> Self {
        UnivariatePoly { coeffs: vec![a, b] }
    }

    /// Interpolates from evaluation points [f(0), f(1)] to coefficients [a, b] in place.
    ///
    /// Given f(x) = ax + b:
    /// - f(0) = b
    /// - f(1) = a + b
    ///
    /// Therefore: a = f(1) - f(0), b = f(0)
    ///
    /// # Arguments
    /// * `evals` - Mutable vector containing [f(0), f(1)] evaluations
    ///
    /// # Returns
    /// * `SumCheckResult<()>` - Success or validation error
    ///
    /// # Errors
    /// * Returns `ValidationError` if evals length is not exactly 2
    ///
    /// # Post-condition
    /// * `evals` is mutated to contain [a, b] coefficients
    pub fn interpolate_in_place(&mut self) -> SumCheckResult<()> {
        let f_0 = self.coeffs[0]; // f(0) = b
        let f_1 = self.coeffs[1]; // f(1) = a + b

        let b = f_1 - f_0; // Linear coefficient: a = f(1) - f(0)
        let a = f_0; // Constant coefficient: b = f(0)

        // Mutate in place: [f(0), f(1)] → [a, b]
        self.coeffs[0] = a;
        self.coeffs[1] = b;

        Ok(())
    }

    /// Evaluates the polynomial at a given Fp4 point.
    ///
    /// Computes f(x) = ax + b for the given x value.
    ///
    /// # Arguments
    /// * `x` - The point at which to evaluate the polynomial
    ///
    /// # Returns
    /// * `Fp4` - The polynomial evaluation f(x)
    pub fn eval_at(&self, x: Fp4) -> Fp4 {
        let a = self.coeffs[0]; // Linear coefficient
        let b = self.coeffs[1]; // Constant coefficient

        a * x + b // f(x) = ax + b
    }

    /// Returns the degree of the polynomial (always 1 for this implementation).
    pub fn degree(&self) -> usize {
        1
    }

    /// Returns a reference to the coefficient vector [a, b].
    pub fn coefficients(&self) -> &Vec<Fp4> {
        &self.coeffs
    }

    /// Returns the linear coefficient (a).
    pub fn linear_coeff(&self) -> Fp4 {
        self.coeffs[0]
    }

    /// Returns the constant coefficient (b).
    pub fn constant_coeff(&self) -> Fp4 {
        self.coeffs[1]
    }
}

impl fmt::Display for UnivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.coeffs[0];
        let b = self.coeffs[1];
        write!(f, "{}*x + {}", a, b)
    }
}

#[cfg(test)]
mod univariate_tests {
    use crate::Fp4;

    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_new_polynomial() {
        let coeffs = vec![Fp4::from_u32(3), Fp4::from_u32(7)]; // 3x + 7
        let poly = UnivariatePoly::new(coeffs).unwrap();

        assert_eq!(poly.degree(), 1);
        assert_eq!(poly.linear_coeff(), Fp4::from_u32(3));
        assert_eq!(poly.constant_coeff(), Fp4::from_u32(7));
    }

    #[test]
    fn test_new_polynomial_invalid_length() {
        let coeffs = vec![Fp4::from_u32(1)]; // Only 1 coefficient
        let result = UnivariatePoly::new(coeffs);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SumCheckError::ValidationError(_)
        ));
    }

    #[test]
    fn test_from_coeffs() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(5), Fp4::from_u32(11));

        assert_eq!(poly.linear_coeff(), Fp4::from_u32(5));
        assert_eq!(poly.constant_coeff(), Fp4::from_u32(11));
    }

    #[test]
    fn test_interpolate_in_place() {
        // For f(x) = 3x + 7:
        // f(0) = 7, f(1) = 10
        // Should interpolate to [a=3, b=7]
        let mut evals = vec![Fp4::from_u32(7), Fp4::from_u32(10)];

        UnivariatePoly::interpolate_in_place(&mut evals).unwrap();

        assert_eq!(evals[0], Fp4::from_u32(3)); // a = f(1) - f(0) = 10 - 7 = 3
        assert_eq!(evals[1], Fp4::from_u32(7)); // b = f(0) = 7
    }

    #[test]
    fn test_interpolate_in_place_invalid_length() {
        let mut evals = vec![Fp4::from_u32(1), Fp4::from_u32(2), Fp4::from_u32(3)];
        let result = UnivariatePoly::interpolate_in_place(&mut evals);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SumCheckError::ValidationError(_)
        ));
    }

    #[test]
    fn test_eval_at() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(3), Fp4::from_u32(7)); // 3x + 7

        assert_eq!(poly.eval_at(Fp4::ZERO), Fp4::from_u32(7)); // f(0) = 7
        assert_eq!(poly.eval_at(Fp4::ONE), Fp4::from_u32(10)); // f(1) = 10
        assert_eq!(poly.eval_at(Fp4::from_u32(2)), Fp4::from_u32(13)); // f(2) = 13
    }

    #[test]
    fn test_interpolation_roundtrip() {
        // Start with polynomial 5x + 11
        let original_poly = UnivariatePoly::from_coeffs(Fp4::from_u32(5), Fp4::from_u32(11));

        // Evaluate at 0 and 1
        let f_0 = original_poly.eval_at(Fp4::ZERO);
        let f_1 = original_poly.eval_at(Fp4::ONE);
        let mut evals = vec![f_0, f_1];

        // Interpolate back to coefficients
        UnivariatePoly::interpolate_in_place(&mut evals).unwrap();

        // Should recover original coefficients
        assert_eq!(evals[0], Fp4::from_u32(5)); // Linear coefficient
        assert_eq!(evals[1], Fp4::from_u32(11)); // Constant coefficient
    }

    #[test]
    fn test_display() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(3), Fp4::from_u32(7));
        let display = format!("{}", poly);
        assert!(display.contains("x"));
        assert!(display.contains("+"));
    }
}
