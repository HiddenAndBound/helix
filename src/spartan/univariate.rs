use super::error::{ SumCheckError, SumCheckResult };
use crate::Fp4;
use p3_field::PrimeCharacteristicRing;
use std::fmt;

/// Univariate polynomials of degree 1-2 with degree-based coefficient indexing.
/// Degree 1: f(x) = a + bx, coeffs [a, b]. Degree 2: f(x) = a + bx + cx², coeffs [a, b, c].
#[derive(Debug, Clone, PartialEq)]
pub struct UnivariatePoly {
    /// Coefficients in degree order: coeffs[i] = coefficient of x^i
    coeffs: Vec<Fp4>,
}

impl UnivariatePoly {
    /// Creates polynomial from coefficients in degree order [a, b] or [a, b, c].
    pub fn new(coeffs: Vec<Fp4>) -> SumCheckResult<Self> {
        if coeffs.len() < 2 || coeffs.len() > 3 {
            return Err(
                SumCheckError::ValidationError(
                    format!(
                        "Polynomial requires 2-3 coefficients for degree 1-2, got {}",
                        coeffs.len()
                    )
                )
            );
        }

        Ok(UnivariatePoly { coeffs })
    }

    /// Creates degree-1 polynomial f(x) = a + bx from coefficients.
    pub fn from_coeffs(a: Fp4, b: Fp4) -> Self {
        UnivariatePoly { coeffs: vec![a, b] }
    }

    /// Creates degree-2 polynomial f(x) = a + bx + cx² from coefficients.
    pub fn from_coeffs_deg2(a: Fp4, b: Fp4, c: Fp4) -> Self {
        UnivariatePoly {
            coeffs: vec![a, b, c],
        }
    }

    /// Interpolates from evaluation points to coefficients in place.
    /// Degree 1: [f(0), f(1)] → [a, b]. Degree 2: [f(0), f(1), f(2)] → [a, b, c].
    pub fn interpolate(&mut self) -> SumCheckResult<()> {
        match self.coeffs.len() {
            2 => {
                let f_0 = self.coeffs[0]; // f(0) = a
                let f_1 = self.coeffs[1]; // f(1) = a + b

                let a = f_0; // Constant coefficient
                let b = f_1 - f_0; // Linear coefficient

                self.coeffs[0] = a;
                self.coeffs[1] = b;
                Ok(())
            }
            3 => {
                let f_0 = self.coeffs[0]; // f(0) = a
                let f_1 = self.coeffs[1]; // f(1) = a + b + c
                let f_2 = self.coeffs[2]; // f(2) = c

                self.coeffs[0] = f_0;
                self.coeffs[1] = f_1 - f_0 - f_2;
                self.coeffs[2] = f_2;
                Ok(())
            }
            _ =>
                Err(
                    SumCheckError::ValidationError(
                        "Interpolation requires 2-3 evaluation points".to_string()
                    )
                ),
        }
    }

    /// Evaluates the polynomial using Horner's method.
    pub fn evaluate(&self, x: Fp4) -> Fp4 {
        match self.coeffs.len() {
            2 => {
                // f(x) = a + bx
                self.coeffs[0] + self.coeffs[1] * x
            }
            3 => {
                // f(x) = a + x(b + cx) using Horner's method
                self.coeffs[0] + x * (self.coeffs[1] + self.coeffs[2] * x)
            }
            _ => panic!("Invalid polynomial degree"),
        }
    }

    /// Returns the degree of the polynomial.
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Returns a reference to the coefficient vector.
    pub fn coefficients(&self) -> &Vec<Fp4> {
        &self.coeffs
    }

    /// Returns the constant coefficient (a).
    pub fn constant_coeff(&self) -> Fp4 {
        self.coeffs[0]
    }

    /// Returns the linear coefficient (b).
    pub fn linear_coeff(&self) -> Fp4 {
        self.coeffs[1]
    }

    /// Returns the quadratic coefficient (c) if degree 2.
    pub fn quadratic_coeff(&self) -> Option<Fp4> {
        if self.coeffs.len() > 2 { Some(self.coeffs[2]) } else { None }
    }
}

impl fmt::Display for UnivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.coeffs.len() {
            2 => {
                let a = self.coeffs[0];
                let b = self.coeffs[1];
                write!(f, "{} + {}*x", a, b)
            }
            3 => {
                let a = self.coeffs[0];
                let b = self.coeffs[1];
                let c = self.coeffs[2];
                write!(f, "{} + {}*x + {}*x^2", a, b, c)
            }
            _ => write!(f, "Invalid polynomial"),
        }
    }
}

#[cfg(test)]
mod univariate_tests {
    use crate::Fp4;

    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_new_polynomial() {
        let coeffs = vec![Fp4::from_u32(3), Fp4::from_u32(7)]; // 3 + 7x
        let poly = UnivariatePoly::new(coeffs).unwrap();

        assert_eq!(poly.degree(), 1);
        assert_eq!(poly.constant_coeff(), Fp4::from_u32(3));
        assert_eq!(poly.linear_coeff(), Fp4::from_u32(7));
    }

    #[test]
    fn test_new_polynomial_invalid_length() {
        let coeffs = vec![Fp4::from_u32(1)]; // Only 1 coefficient
        let result = UnivariatePoly::new(coeffs);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SumCheckError::ValidationError(_)));

        let coeffs = vec![Fp4::from_u32(1), Fp4::from_u32(2), Fp4::from_u32(3), Fp4::from_u32(4)]; // Too many
        let result = UnivariatePoly::new(coeffs);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_coeffs() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(5), Fp4::from_u32(11));

        assert_eq!(poly.constant_coeff(), Fp4::from_u32(5));
        assert_eq!(poly.linear_coeff(), Fp4::from_u32(11));
    }

    #[test]
    fn test_interpolate() {
        // For f(x) = 7 + 3x:
        // f(0) = 7, f(1) = 10
        // Should interpolate to [a=7, b=3]
        let evals = vec![Fp4::from_u32(7), Fp4::from_u32(10)];
        let mut poly = UnivariatePoly::new(evals).unwrap();

        poly.interpolate().unwrap();

        assert_eq!(poly.coeffs[0], Fp4::from_u32(7)); // a = f(0) = 7
        assert_eq!(poly.coeffs[1], Fp4::from_u32(3)); // b = f(1) - f(0) = 10 - 7 = 3
    }

    #[test]
    fn test_interpolate_degree2() {
        // For f(x) = 1 + 2x + 3x^2:
        // f(0) = 1, f(1) = 6, f(2) = 17
        let evals = vec![Fp4::from_u32(1), Fp4::from_u32(6), Fp4::from_u32(17)];
        let mut poly = UnivariatePoly::new(evals).unwrap();

        poly.interpolate().unwrap();

        assert_eq!(poly.coeffs[0], Fp4::from_u32(1)); // a = 1
        assert_eq!(poly.coeffs[1], Fp4::from_u32(2)); // b = 2
        assert_eq!(poly.coeffs[2], Fp4::from_u32(3)); // c = 3
    }

    #[test]
    fn test_evaluate() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(3), Fp4::from_u32(7)); // 3 + 7x

        assert_eq!(poly.evaluate(Fp4::ZERO), Fp4::from_u32(3)); // f(0) = 3
        assert_eq!(poly.evaluate(Fp4::ONE), Fp4::from_u32(10)); // f(1) = 10
        assert_eq!(poly.evaluate(Fp4::from_u32(2)), Fp4::from_u32(17)); // f(2) = 17
    }

    #[test]
    fn test_interpolation_roundtrip() {
        // Start with polynomial 11 + 5x
        let original_poly = UnivariatePoly::from_coeffs(Fp4::from_u32(11), Fp4::from_u32(5));

        // Evaluate at 0 and 1
        let f_0 = original_poly.evaluate(Fp4::ZERO);
        let f_1 = original_poly.evaluate(Fp4::ONE);
        let evals = vec![f_0, f_1];

        // Create polynomial from evaluations and interpolate back to coefficients
        let mut eval_poly = UnivariatePoly::new(evals).unwrap();
        eval_poly.interpolate().unwrap();

        // Should recover original coefficients
        assert_eq!(eval_poly.coeffs[0], Fp4::from_u32(11)); // Constant coefficient
        assert_eq!(eval_poly.coeffs[1], Fp4::from_u32(5)); // Linear coefficient
    }

    #[test]
    fn test_display() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(3), Fp4::from_u32(7));
        let display = format!("{}", poly);
        assert!(display.contains("x"));
        assert!(display.contains("+"));
    }

    #[test]
    fn test_from_coeffs_deg2() {
        let poly = UnivariatePoly::from_coeffs_deg2(
            Fp4::from_u32(1),
            Fp4::from_u32(2),
            Fp4::from_u32(3)
        );

        assert_eq!(poly.degree(), 2);
        assert_eq!(poly.constant_coeff(), Fp4::from_u32(1));
        assert_eq!(poly.linear_coeff(), Fp4::from_u32(2));
        assert_eq!(poly.quadratic_coeff(), Some(Fp4::from_u32(3)));
    }

    #[test]
    fn test_evaluate_degree2() {
        let poly = UnivariatePoly::from_coeffs_deg2(
            Fp4::from_u32(1),
            Fp4::from_u32(2),
            Fp4::from_u32(3)
        ); // 1 + 2x + 3x^2

        assert_eq!(poly.evaluate(Fp4::ZERO), Fp4::from_u32(1)); // f(0) = 1
        assert_eq!(poly.evaluate(Fp4::ONE), Fp4::from_u32(6)); // f(1) = 1 + 2 + 3 = 6
        assert_eq!(poly.evaluate(Fp4::from_u32(2)), Fp4::from_u32(17)); // f(2) = 1 + 4 + 12 = 17
    }

    #[test]
    fn test_interpolation_roundtrip_degree2() {
        let original_poly = UnivariatePoly::from_coeffs_deg2(
            Fp4::from_u32(5),
            Fp4::from_u32(7),
            Fp4::from_u32(2)
        );

        let f_0 = original_poly.evaluate(Fp4::ZERO);
        let f_1 = original_poly.evaluate(Fp4::ONE);
        let f_2 = original_poly.evaluate(Fp4::from_u32(2));
        let evals = vec![f_0, f_1, f_2];

        let mut eval_poly = UnivariatePoly::new(evals).unwrap();
        eval_poly.interpolate().unwrap();

        assert_eq!(eval_poly.coeffs[0], Fp4::from_u32(5)); // Constant coefficient
        assert_eq!(eval_poly.coeffs[1], Fp4::from_u32(7)); // Linear coefficient
        assert_eq!(eval_poly.coeffs[2], Fp4::from_u32(2)); // Quadratic coefficient
    }

    #[test]
    fn test_quadratic_coeff_none_for_degree1() {
        let poly = UnivariatePoly::from_coeffs(Fp4::from_u32(1), Fp4::from_u32(2));
        assert_eq!(poly.quadratic_coeff(), None);
    }
}
