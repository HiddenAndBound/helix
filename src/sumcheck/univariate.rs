use crate::{Fp, Fp4};
use p3_field::{Field, PrimeCharacteristicRing};
use std::ops::{Add, Div, Mul};

/// A univariate polynomial represented by its coefficients
#[derive(Debug, Clone, PartialEq)]
pub struct UnivariatePolynomial<F> {
    coefficients: Vec<F>,
}

impl<F: Clone + PartialEq + Default + Add<Output = F> + Mul<Output = F>> UnivariatePolynomial<F> {
    /// Create a new polynomial from coefficients (constant term first)
    pub fn new(coefficients: Vec<F>) -> Self {
        let mut coeffs = coefficients;
        // Remove trailing zeros
        while coeffs.len() > 1 && coeffs.last().unwrap() == &F::default() {
            coeffs.pop();
        }
        Self {
            coefficients: coeffs,
        }
    }

    /// Create a zero polynomial
    pub fn zero() -> Self {
        Self::new(vec![F::default()])
    }

    /// Create a constant polynomial
    pub fn constant(c: F) -> Self {
        Self::new(vec![c])
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Evaluate the polynomial at a given point
    pub fn evaluate(&self, x: F) -> F {
        let mut result = F::default();
        for coeff in self.coefficients.iter().rev() {
            result = result * x.clone() + coeff.clone();
        }
        result
    }

    /// Get coefficients as a slice
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }
}

/// Specialized implementation for Fp
impl<F> UnivariatePolynomial<F>
where
    F: PrimeCharacteristicRing
        + Clone
        + PartialEq
        + Default
        + Add<Output = F>
        + Mul<Output = F>
        + Div<Output = F>,
{
    /// Interpolate a polynomial from evaluations at points {0, 1, ..., d} to coefficients
    /// For a degree d polynomial, takes d+1 evaluations and returns coefficients
    pub fn interpolate(evaluations: &[F]) -> Option<Self> {
        if evaluations.is_empty() {
            return None;
        }

        let degree = evaluations.len() - 1;
        let coefficients = match degree {
            0 => {
                // Degree 0: f(x) = c, so f(0) = c
                vec![evaluations[0].clone()]
            }
            1 => {
                // Degree 1: f(x) = a + bx
                // f(0) = a, f(1) = a + b
                // So a = f(0), b = f(1) - f(0)
                vec![
                    evaluations[0].clone(),
                    evaluations[1].clone() - evaluations[0].clone(),
                ]
            }
            2 => {
                // Degree 2: f(x) = a + bx + cx²
                // f(0) = a, f(1) = a + b + c, f(2) = a + 2b + 4c
                // Using Lagrange interpolation formulas
                let f0 = evaluations[0].clone();
                let f1 = evaluations[1].clone();
                let f2 = evaluations[2].clone();

                // a = f(0)
                let a = f0.clone();
                // Solving the system for b and c
                // f(1) - f(0) = b + c
                // f(2) - f(0) = 2b + 4c
                // From first: c = f(1) - f(0) - b
                // Substitute: f(2) - f(0) = 2b + 4(f(1) - f(0) - b) = 2b + 4f(1) - 4f(0) - 4b = 4f(1) - 4f(0) - 2b
                // So: f(2) - f(0) - 4f(1) + 4f(0) = -2b
                // b = (4f(1) - 4f(0) - f(2) + f(0)) / 2 = (4f(1) - 3f(0) - f(2)) / 2
                let two = F::ONE + F::ONE;
                let three = two.clone() + F::ONE;
                let four = two.clone() + two.clone();

                let b = (four * f1.clone() - three * f0.clone() - f2.clone()) / two;
                let c = f1 - f0 - b.clone();

                vec![a, b, c]
            }
            3 => {
                // Degree 3: f(x) = a + bx + cx² + dx³
                let f0 = evaluations[0].clone();
                let f1 = evaluations[1].clone();
                let f2 = evaluations[2].clone();
                let f3 = evaluations[3].clone();

                let one = F::ONE;
                let two = one.clone() + one.clone();
                let three = two.clone() + one;
                let four = two.clone() + two.clone();
                let six = three.clone() + three.clone();

                // Using Newton's divided differences
                // For points 0,1,2,3 with values f0,f1,f2,f3
                // Newton form: f(x) = f[0] + f[0,1]x + f[0,1,2]x(x-1) + f[0,1,2,3]x(x-1)(x-2)
                // Convert to standard form: a + bx + cx² + dx³

                // First order divided differences
                let f01 = f1.clone() - f0.clone(); // f[0,1] = (f1-f0)/(1-0) = f1-f0
                let f12 = f2.clone() - f1; // f[1,2] = (f2-f1)/(2-1) = f2-f1  
                let f23 = f3.clone() - f2; // f[2,3] = (f3-f2)/(3-2) = f3-f2

                // Second order divided differences
                let f012 = (f12.clone() - f01.clone()) / two.clone(); // f[0,1,2] = (f[1,2]-f[0,1])/(2-0) = (f12-f01)/2
                let f123 = (f23 - f12) / two.clone(); // f[1,2,3] = (f[2,3]-f[1,2])/(3-1) = (f23-f12)/2

                // Third order divided difference
                let f0123 = (f123 - f012.clone()) / three.clone(); // f[0,1,2,3] = (f[1,2,3]-f[0,1,2])/(3-0) = (f123-f012)/3

                // Convert Newton form to standard form
                // f(x) = f0 + f01*x + f012*x*(x-1) + f0123*x*(x-1)*(x-2)
                // f(x) = f0 + f01*x + f012*(x²-x) + f0123*(x³-3x²+2x)
                // f(x) = f0 + f01*x + f012*x² - f012*x + f0123*x³ - 3*f0123*x² + 2*f0123*x
                // f(x) = f0 + (f01 - f012 + 2*f0123)*x + (f012 - 3*f0123)*x² + f0123*x³

                let a = f0; // constant term
                let b = f01 - f012.clone() + two.clone() * f0123.clone(); // coefficient of x
                let c = f012 - three.clone() * f0123.clone(); // coefficient of x²
                let d = f0123; // coefficient of x³

                vec![a, b, c, d]
            }
            4 => {
                // Degree 4: f(x) = a + bx + cx² + dx³ + ex⁴
                let f0 = evaluations[0].clone();
                let f1 = evaluations[1].clone();
                let f2 = evaluations[2].clone();
                let f3 = evaluations[3].clone();
                let f4 = evaluations[4].clone();

                let one = F::ONE;
                let two = one.clone() + one.clone();
                let three = two.clone() + one;
                let four = two.clone() + two.clone();
                let six = three.clone() + three.clone();

                // Using Newton's divided differences
                // For points 0,1,2,3,4 with values f0,f1,f2,f3,f4
                // Newton form: f(x) = f[0] + f[0,1]x + f[0,1,2]x(x-1) + f[0,1,2,3]x(x-1)(x-2) + f[0,1,2,3,4]x(x-1)(x-2)(x-3)
                // Convert to standard form: a + bx + cx² + dx³ + ex⁴

                // First order divided differences
                let f01 = f1.clone() - f0.clone(); // f[0,1]
                let f12 = f2.clone() - f1; // f[1,2]  
                let f23 = f3.clone() - f2; // f[2,3]
                let f34 = f4.clone() - f3; // f[3,4]

                // Second order divided differences
                let f012 = (f12.clone() - f01.clone()) / two.clone(); // f[0,1,2]
                let f123 = (f23.clone() - f12) / two.clone(); // f[1,2,3]
                let f234 = (f34 - f23) / two.clone(); // f[2,3,4]

                // Third order divided differences
                let f0123 = (f123.clone() - f012.clone()) / three.clone(); // f[0,1,2,3]
                let f1234 = (f234 - f123) / three.clone(); // f[1,2,3,4]

                // Fourth order divided difference
                let f01234 = (f1234 - f0123.clone()) / four.clone(); // f[0,1,2,3,4]

                // Convert Newton form to standard form step by step
                // f(x) = f0 + f01*x + f012*x*(x-1) + f0123*x*(x-1)*(x-2) + f01234*x*(x-1)*(x-2)*(x-3)

                // Let's expand each term systematically:
                // Term 1: f0 = f0
                // Term 2: f01*x = f01*x
                // Term 3: f012*x*(x-1) = f012*x² - f012*x
                // Term 4: f0123*x*(x-1)*(x-2) = f0123*x*(x²-3x+2) = f0123*x³ - 3*f0123*x² + 2*f0123*x
                // Term 5: f01234*x*(x-1)*(x-2)*(x-3) = f01234*x*(x³-6x²+11x-6) = f01234*x⁴ - 6*f01234*x³ + 11*f01234*x² - 6*f01234*x

                // Collecting all terms by degree:
                // x⁰: f0
                // x¹: f01 - f012 + 2*f0123 - 6*f01234
                // x²: f012 - 3*f0123 + 11*f01234
                // x³: f0123 - 6*f01234
                // x⁴: f01234

                let eleven = six.clone() + three.clone() + two.clone();

                let a = f0; // constant term
                let b =
                    f01 - f012.clone() + two.clone() * f0123.clone() - six.clone() * f01234.clone(); // coefficient of x
                let c = f012 - three.clone() * f0123.clone() + eleven * f01234.clone(); // coefficient of x²
                let d = f0123 - six.clone() * f01234.clone(); // coefficient of x³
                let e = f01234; // coefficient of x⁴

                vec![a, b, c, d, e]
            }
            _ => return None, // Only support up to degree 4
        };

        Some(Self::new(coefficients))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp, Fp4};

    #[test]
    fn test_polynomial_creation() {
        let poly =
            UnivariatePolynomial::new(vec![Fp::from_u32(1), Fp::from_u32(2), Fp::from_u32(3)]);
        assert_eq!(poly.degree(), 2);
        assert_eq!(
            poly.coefficients(),
            &[Fp::from_u32(1), Fp::from_u32(2), Fp::from_u32(3)]
        );
    }

    #[test]
    fn test_polynomial_evaluation() {
        let poly =
            UnivariatePolynomial::new(vec![Fp::from_u32(1), Fp::from_u32(2), Fp::from_u32(3)]);

        // Evaluate at x = 2: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let result = poly.evaluate(Fp::from_u32(2));
        assert_eq!(result, Fp::from_u32(17));
    }

    #[test]
    fn test_interpolate_degree_0() {
        // f(x) = 5, so f(0) = 5
        let evaluations = vec![Fp::from_u32(5)];
        let poly = UnivariatePolynomial::<Fp>::interpolate(&evaluations).unwrap();

        assert_eq!(poly.coefficients(), &[Fp::from_u32(5)]);
        assert_eq!(poly.evaluate(Fp::from_u32(0)), Fp::from_u32(5));
        assert_eq!(poly.evaluate(Fp::from_u32(42)), Fp::from_u32(5));
    }

    #[test]
    fn test_interpolate_degree_1() {
        // f(x) = 2 + 3x, so f(0) = 2, f(1) = 5
        let evaluations = vec![Fp::from_u32(2), Fp::from_u32(5)];
        let poly = UnivariatePolynomial::<Fp>::interpolate(&evaluations).unwrap();

        // Expected coefficients: [2, 3] (f(0) = 2, f(1) - f(0) = 3)
        assert_eq!(poly.coefficients(), &[Fp::from_u32(2), Fp::from_u32(3)]);
        assert_eq!(poly.evaluate(Fp::from_u32(0)), Fp::from_u32(2));
        assert_eq!(poly.evaluate(Fp::from_u32(1)), Fp::from_u32(5));
        assert_eq!(poly.evaluate(Fp::from_u32(2)), Fp::from_u32(8)); // 2 + 3*2 = 8
    }

    #[test]
    fn test_interpolate_degree_2() {
        // f(x) = 1 + 2x + 3x², so f(0) = 1, f(1) = 6, f(2) = 17
        let evaluations = vec![Fp::from_u32(1), Fp::from_u32(6), Fp::from_u32(17)];
        let poly = UnivariatePolynomial::<Fp>::interpolate(&evaluations).unwrap();

        assert_eq!(
            poly.coefficients(),
            &[Fp::from_u32(1), Fp::from_u32(2), Fp::from_u32(3)]
        );
        assert_eq!(poly.evaluate(Fp::from_u32(0)), Fp::from_u32(1));
        assert_eq!(poly.evaluate(Fp::from_u32(1)), Fp::from_u32(6));
        assert_eq!(poly.evaluate(Fp::from_u32(2)), Fp::from_u32(17));
        assert_eq!(poly.evaluate(Fp::from_u32(3)), Fp::from_u32(34)); // 1 + 6 + 27 = 34
    }

    #[test]
    fn test_interpolate_degree_3() {
        // f(x) = 1 + x + x² + x³, so f(0) = 1, f(1) = 4, f(2) = 15, f(3) = 40
        let evaluations = vec![
            Fp::from_u32(1),
            Fp::from_u32(4),
            Fp::from_u32(15),
            Fp::from_u32(40),
        ];
        let poly = UnivariatePolynomial::<Fp>::interpolate(&evaluations).unwrap();

        assert_eq!(
            poly.coefficients(),
            &[
                Fp::from_u32(1),
                Fp::from_u32(1),
                Fp::from_u32(1),
                Fp::from_u32(1)
            ]
        );
        assert_eq!(poly.evaluate(Fp::from_u32(0)), Fp::from_u32(1));
        assert_eq!(poly.evaluate(Fp::from_u32(1)), Fp::from_u32(4));
        assert_eq!(poly.evaluate(Fp::from_u32(2)), Fp::from_u32(15));
        assert_eq!(poly.evaluate(Fp::from_u32(3)), Fp::from_u32(40));
    }

    #[test]
    fn test_interpolate_degree_4() {
        // f(x) = 1 + x⁴, so f(0) = 1, f(1) = 2, f(2) = 17, f(3) = 82, f(4) = 257
        let evaluations = vec![
            Fp::from_u32(1),
            Fp::from_u32(2),
            Fp::from_u32(17),
            Fp::from_u32(82),
            Fp::from_u32(257),
        ];
        let poly = UnivariatePolynomial::<Fp>::interpolate(&evaluations).unwrap();

        assert_eq!(
            poly.coefficients(),
            &[
                Fp::from_u32(1),
                Fp::from_u32(0),
                Fp::from_u32(0),
                Fp::from_u32(0),
                Fp::from_u32(1)
            ]
        );
        assert_eq!(poly.evaluate(Fp::from_u32(0)), Fp::from_u32(1));
        assert_eq!(poly.evaluate(Fp::from_u32(1)), Fp::from_u32(2));
        assert_eq!(poly.evaluate(Fp::from_u32(2)), Fp::from_u32(17));
        assert_eq!(poly.evaluate(Fp::from_u32(3)), Fp::from_u32(82));
        assert_eq!(poly.evaluate(Fp::from_u32(4)), Fp::from_u32(257));
    }

    #[test]
    fn test_interpolate_unsupported_degree() {
        // Degree 5 is not supported
        let evaluations = vec![Fp::from_u32(1); 6];
        let result = UnivariatePolynomial::<Fp>::interpolate(&evaluations);
        assert!(result.is_none());
    }

    #[test]
    fn test_interpolate_empty() {
        let evaluations: Vec<Fp> = vec![];
        let result = UnivariatePolynomial::<Fp>::interpolate(&evaluations);
        assert!(result.is_none());
    }
}
