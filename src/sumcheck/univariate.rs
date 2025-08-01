use std::ops::{Add, Mul};
use crate::{Fp, Fp4};

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
        Self { coefficients: coeffs }
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

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(max_len);
        
        for i in 0..max_len {
            let a = if i < self.coefficients.len() { &self.coefficients[i] } else { &F::default() };
            let b = if i < other.coefficients.len() { &other.coefficients[i] } else { &F::default() };
            result.push(a.clone() + b.clone());
        }
        
        Self::new(result)
    }

    /// Multiply two polynomials
    pub fn mul(&self, other: &Self) -> Self {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Self::zero();
        }
        
        let mut result = vec![F::default(); self.coefficients.len() + other.coefficients.len() - 1];
        
        for (i, a) in self.coefficients.iter().enumerate() {
            for (j, b) in other.coefficients.iter().enumerate() {
                result[i + j] = result[i + j].clone() + a.clone() * b.clone();
            }
        }
        
        Self::new(result)
    }

    /// Get coefficients as a slice
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }
}

/// Specialized implementation for Fp
impl UnivariatePolynomial<Fp> {
    /// Interpolate a polynomial from points and values using Lagrange interpolation
    pub fn interpolate(points: &[Fp], values: &[Fp]) -> Result<Self, String> {
        if points.len() != values.len() {
            return Err("Points and values must have the same length".to_string());
        }
        
        if points.is_empty() {
            return Ok(Self::zero());
        }
        
        let n = points.len();
        let mut coeffs = vec![Fp::ZERO; n];
        
        for i in 0..n {
            let mut basis = vec![Fp::ONE];
            
            // Compute Lagrange basis polynomial for point i
            for j in 0..n {
                if i != j {
                    let mut new_basis = vec![Fp::ZERO; basis.len() + 1];
                    let denom = points[i] - points[j];
                    let denom_inv = denom.inverse().ok_or("Division by zero")?;
                    
                    // Multiply by (x - x_j) / (x_i - x_j)
                    for k in 0..basis.len() {
                        new_basis[k] = new_basis[k] + basis[k] * (-points[j]) * denom_inv;
                        new_basis[k + 1] = new_basis[k + 1] + basis[k] * denom_inv;
                    }
                    
                    basis = new_basis;
                }
            }
            
            // Add y_i * L_i(x) to the result
            for (k, coeff) in basis.iter().enumerate() {
                coeffs[k] = coeffs[k] + values[i] * coeff;
            }
        }
        
        Ok(Self::new(coeffs))
    }
}

/// Specialized implementation for Fp4
impl UnivariatePolynomial<Fp4> {
    /// Interpolate a polynomial from points and values using Lagrange interpolation
    pub fn interpolate(points: &[Fp4], values: &[Fp4]) -> Result<Self, String> {
        if points.len() != values.len() {
            return Err("Points and values must have the same length".to_string());
        }
        
        if points.is_empty() {
            return Ok(Self::zero());
        }
        
        let n = points.len();
        let mut coeffs = vec![Fp4::ZERO; n];
        
        for i in 0..n {
            let mut basis = vec![Fp4::ONE];
            
            // Compute Lagrange basis polynomial for point i
            for j in 0..n {
                if i != j {
                    let mut new_basis = vec![Fp4::ZERO; basis.len() + 1];
                    let denom = points[i] - points[j];
                    let denom_inv = denom.inverse().ok_or("Division by zero")?;
                    
                    // Multiply by (x - x_j) / (x_i - x_j)
                    for k in 0..basis.len() {
                        new_basis[k] = new_basis[k] + basis[k] * (-points[j]) * denom_inv;
                        new_basis[k + 1] = new_basis[k + 1] + basis[k] * denom_inv;
                    }
                    
                    basis = new_basis;
                }
            }
            
            // Add y_i * L_i(x) to the result
            for (k, coeff) in basis.iter().enumerate() {
                coeffs[k] = coeffs[k] + values[i] * coeff;
            }
        }
        
        Ok(Self::new(coeffs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp, Fp4};

    #[test]
    fn test_polynomial_creation() {
        let poly = UnivariatePolynomial::new(vec![Fp::from(1), Fp::from(2), Fp::from(3)]);
        assert_eq!(poly.degree(), 2);
        assert_eq!(poly.coefficients(), &[Fp::from(1), Fp::from(2), Fp::from(3)]);
    }

    #[test]
    fn test_polynomial_evaluation() {
        let poly = UnivariatePolynomial::new(vec![Fp::from(1), Fp::from(2), Fp::from(3)]);
        
        // Evaluate at x = 2: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let result = poly.evaluate(Fp::from(2));
        assert_eq!(result, Fp::from(17));
    }

    #[test]
    fn test_polynomial_addition() {
        let poly1 = UnivariatePolynomial::new(vec![Fp::from(1), Fp::from(2)]);
        let poly2 = UnivariatePolynomial::new(vec![Fp::from(3), Fp::from(4), Fp::from(5)]);
        
        let sum = poly1.add(&poly2);
        assert_eq!(sum.coefficients(), &[Fp::from(4), Fp::from(6), Fp::from(5)]);
    }

    #[test]
    fn test_polynomial_multiplication() {
        let poly1 = UnivariatePolynomial::new(vec![Fp::from(1), Fp::from(2)]);
        let poly2 = UnivariatePolynomial::new(vec![Fp::from(3), Fp::from(4)]);
        
        let product = poly1.mul(&poly2);
        assert_eq!(product.coefficients(), &[Fp::from(3), Fp::from(10), Fp::from(8)]);
    }

    #[test]
    fn test_polynomial_interpolation() {
        let points = vec![Fp::from(1), Fp::from(2), Fp::from(3)];
        let values = vec![Fp::from(1), Fp::from(4), Fp::from(9)];
        
        let poly = UnivariatePolynomial::interpolate(&points, &values).unwrap();
        
        // Verify interpolation
        for (point, expected) in points.iter().zip(values.iter()) {
            let result = poly.evaluate(*point);
            assert_eq!(result, *expected);
        }
    }
}