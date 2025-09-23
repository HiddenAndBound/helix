use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use std::ops::{Add, Index, Mul, Range};

use crate::utils::{Fp4, eq::EqEvals};

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MLE<F: PrimeCharacteristicRing + Field + Clone> {
    coeffs: Vec<F>,
}

impl<F: PrimeCharacteristicRing + Clone + Field> MLE<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        Self { coeffs }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn n_vars(&self) -> usize {
        self.coeffs.len().trailing_zeros() as usize
    }

    /// Evaluates the MLE at the given point
    pub fn evaluate(&self, point: &[Fp4]) -> Fp4
    where
        Fp4: ExtensionField<F>,
    {
        assert_eq!(
            point.len(),
            self.n_vars(),
            "Dimensions of point must match MLE variables"
        );

        let eq = EqEvals::gen_from_point(point);

        eq.coeffs()
            .iter()
            .zip(self.coeffs.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    /// Folds the MLE by binding the lowest variable to a challenge value.
    /// Automatically promotes to Fp4 if needed, consuming self and returning MLE<Fp4>.
    ///
    /// For MLE f(x₀, x₁, ..., xₙ₋₁), this computes:
    /// g(x₁, ..., xₙ₋₁) = f(challenge, x₁, ..., xₙ₋₁)
    ///                  = (1 - challenge) * f(0, x₁, ..., xₙ₋₁) + challenge * f(1, x₁, ..., xₙ₋₁)
    pub fn fold_in_place(&self, r: Fp4) -> MLE<Fp4>
    where
        Fp4: ExtensionField<F> + Mul<F, Output = Fp4>,
    {
        if self.coeffs.len() == 1 {
            // Base case: 0-variable polynomial, promote to Fp4 and return
            return MLE::new(vec![Fp4::from(self.coeffs[0].clone())]);
        }

        let half_len = self.coeffs.len() >> 1;
        let mut folded_coeffs = Vec::with_capacity(half_len);

        // For each coefficient pair (low, high) where low corresponds to x₀=0 and high to x₀=1
        // In hypercube layout, we pair coefficients that differ only in the lowest bit
        for i in 0..half_len {
            // Compute (1 - challenge) * low + challenge * high
            let folded = r * (self[(i << 1) | 1] - self[i << 1]) + self[i << 1];
            folded_coeffs.push(folded);
        }

        MLE::new(folded_coeffs)
    }

    /// Returns a reference to the coefficient vector (for testing/debugging)
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Creates MLE from vector (convenience method)
    pub fn from_vector(vector: Vec<F>) -> Self {
        Self::new(vector)
    }

    /// Creates zero MLE with given number of variables
    pub fn zero(n_vars: usize) -> Self {
        let len = 1 << n_vars;
        Self::new(vec![F::ZERO; len])
    }

    /// Creates constant MLE
    pub fn constant(value: F, n_vars: usize) -> Self {
        let len = 1 << n_vars;
        Self::new(vec![value; len])
    }

    /// Computes partial evaluation (binds variables from left)
    pub fn partial_evaluate(&mut self, point: &[Fp4], num_vars: usize) -> MLE<Fp4>
    where
        Fp4: ExtensionField<F>,
    {
        assert!(num_vars <= self.n_vars(), "Too many variables to bind");

        let mut current = self.fold_in_place(point[0]);

        for &challenge in point.iter().take(num_vars).skip(1) {
            current = current.fold_in_place(challenge);
        }

        current
    }

    /// Computes dot product with another MLE
    pub fn dot_product(&self, other: &Self) -> F
    where
        F: Mul<Output = F> + Add<Output = F> + Clone,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Dimension mismatch for dot product"
        );

        self.coeffs()
            .iter()
            .zip(other.coeffs().iter())
            .map(|(a, b)| a.clone() * b.clone())
            .fold(F::ZERO, |acc, x| acc + x)
    }

    /// Adds another MLE element-wise
    pub fn add(&self, other: &Self) -> Self
    where
        F: Add<Output = F> + Clone,
    {
        assert_eq!(self.len(), other.len(), "Dimension mismatch for addition");

        let coeffs = self
            .coeffs()
            .iter()
            .zip(other.coeffs().iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Self::new(coeffs)
    }

    /// Scales by a scalar
    pub fn scale(&self, scalar: F) -> Self
    where
        F: Mul<Output = F> + Clone,
    {
        let coeffs = self
            .coeffs()
            .iter()
            .map(|coeff| coeff.clone() * scalar.clone())
            .collect();

        Self::new(coeffs)
    }
}

impl<F: PrimeCharacteristicRing + Field + Clone> Index<usize> for MLE<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<F: PrimeCharacteristicRing + Field + Clone> Index<Range<usize>> for MLE<F> {
    type Output = [F];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.coeffs[range]
    }
}

#[cfg(test)]
mod tests {
    use crate::Fp;

    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    //Tests that folding, and inner product return the same value which should be the evaluation.
    #[test]
    fn test_eval_vs_fold() {
        let n_vars = 3;
        let mut rng = StdRng::seed_from_u64(0);
        let point: Vec<Fp4> = (0..n_vars).map(|_| Fp4::from_u128(rng.r#gen())).collect();
        let mut mle = MLE::from_vector(
            (0..1 << n_vars)
                .map(|_| Fp::from_u32(rng.r#gen()))
                .collect(),
        );

        let claimed_eval = mle.evaluate(&point);
        let folded_eval = mle.partial_evaluate(&point, n_vars)[0];

        assert_eq!(claimed_eval, folded_eval)
    }
    #[test]
    fn test_fold_single_variable() {
        // Test edge case: single variable MLE
        let coeffs = vec![BabyBear::from_u32(7), BabyBear::from_u32(11)];
        let mle = MLE::new(coeffs.clone());
        let challenge = Fp4::from_u32(13);

        let folded = mle.fold_in_place(challenge);

        // Should return constant polynomial (0 variables)
        assert_eq!(folded.n_vars(), 0);
        assert_eq!(folded.coeffs().len(), 1);

        // Result should be (1-13)*7 + 13*11 = -12*7 + 13*11 = -84 + 143 = 59
        let expected = (Fp4::ONE - challenge) * Fp4::from_u32(7) + challenge * Fp4::from_u32(11);
        assert_eq!(folded.coeffs()[0], expected);
    }

    #[test]
    fn test_fold_constant_polynomial() {
        // Test edge case: constant polynomial (0 variables)
        let coeffs = vec![BabyBear::from_u32(42)];
        let mle = MLE::new(coeffs.clone());
        let challenge = Fp4::from_u32(17);

        let folded = mle.fold_in_place(challenge);

        // Should remain constant polynomial (promoted to Fp4)
        assert_eq!(folded.n_vars(), 0);
        assert_eq!(folded.coeffs().len(), 1);
        assert_eq!(folded.coeffs()[0], Fp4::from_u32(42));
    }

    #[test]
    fn test_base_field_vs_extension_field() {
        // Test that BabyBear and Fp4 MLEs produce consistent results
        let coeffs_base = vec![
            BabyBear::from_u32(10),
            BabyBear::from_u32(20),
            BabyBear::from_u32(30),
            BabyBear::from_u32(40),
        ];
        let coeffs_ext = vec![
            Fp4::from_u32(10),
            Fp4::from_u32(20),
            Fp4::from_u32(30),
            Fp4::from_u32(40),
        ];

        let mle_base = MLE::new(coeffs_base);
        let mle_ext = MLE::new(coeffs_ext);
        let challenge = Fp4::from_u32(7);

        let folded_base = mle_base.fold_in_place(challenge);
        let folded_ext = mle_ext.fold_in_place(challenge);

        // Results should be identical
        assert_eq!(folded_base.coeffs(), folded_ext.coeffs());
        assert_eq!(folded_base.n_vars(), folded_ext.n_vars());
    }

    #[test]
    fn test_multiple_folds() {
        // Test multiple consecutive folds (sum-check protocol simulation)
        let coeffs = vec![
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
        ];
        let mle = MLE::new(coeffs);

        assert_eq!(mle.n_vars(), 2);

        // First fold: 2 vars -> 1 var
        let mle = mle.fold_in_place(Fp4::from_u32(5));
        assert_eq!(mle.n_vars(), 1);
        assert_eq!(mle.coeffs().len(), 2);

        // Second fold: 1 var -> 0 vars (constant)
        let mle = mle.fold_in_place(Fp4::from_u32(7));
        assert_eq!(mle.n_vars(), 0);
        assert_eq!(mle.coeffs().len(), 1);
    }

    #[test]
    fn test_folding_with_zero_challenge() {
        // Test folding with challenge = 0
        let coeffs = vec![
            BabyBear::from_u32(10), // f(0,0)
            BabyBear::from_u32(20), // f(1,0)
            BabyBear::from_u32(30), // f(0,1)
            BabyBear::from_u32(40), // f(1,1)
        ];
        let mle = MLE::new(coeffs);
        let folded = mle.fold_in_place(Fp4::ZERO);

        // With challenge = 0, we get g(x₁) where g(x₁) = f(0, x₁)
        // So g(0) = f(0,0) = 10 and g(1) = f(0,1) = 30
        assert_eq!(folded.coeffs()[0], Fp4::from_u32(10));
        assert_eq!(folded.coeffs()[1], Fp4::from_u32(30));
    }

    #[test]
    fn test_folding_with_one_challenge() {
        // Test folding with challenge = 1
        let coeffs = vec![
            BabyBear::from_u32(10),
            BabyBear::from_u32(20),
            BabyBear::from_u32(30),
            BabyBear::from_u32(40),
        ];
        let mle = MLE::new(coeffs);
        let folded = mle.fold_in_place(Fp4::ONE);

        // With challenge = 1, result should be [f(1,0), f(1,1)] = [20, 40]
        assert_eq!(folded.coeffs()[0], Fp4::from_u32(20));
        assert_eq!(folded.coeffs()[1], Fp4::from_u32(40));
    }

    #[test]
    fn test_mle_index_single_element() {
        let coeffs = vec![BabyBear::from_u32(42)];
        let mle = MLE::new(coeffs);

        assert_eq!(mle[0], BabyBear::from_u32(42));
    }

    #[test]
    fn test_mle_index_multiple_elements() {
        let coeffs = vec![
            BabyBear::from_u32(10),
            BabyBear::from_u32(20),
            BabyBear::from_u32(30),
            BabyBear::from_u32(40),
        ];
        let mle = MLE::new(coeffs);

        assert_eq!(mle[0], BabyBear::from_u32(10));
        assert_eq!(mle[1], BabyBear::from_u32(20));
        assert_eq!(mle[2], BabyBear::from_u32(30));
        assert_eq!(mle[3], BabyBear::from_u32(40));
    }

    #[test]
    fn test_mle_index_range() {
        let coeffs = vec![
            BabyBear::from_u32(10),
            BabyBear::from_u32(20),
            BabyBear::from_u32(30),
            BabyBear::from_u32(40),
        ];
        let mle = MLE::new(coeffs);

        let slice = &mle[1..3];
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], BabyBear::from_u32(20));
        assert_eq!(slice[1], BabyBear::from_u32(30));
    }

    #[test]
    fn test_mle_index_full_range() {
        let coeffs = vec![BabyBear::from_u32(100), BabyBear::from_u32(200)];
        let mle = MLE::new(coeffs);

        let full_slice = &mle[0..2];
        assert_eq!(full_slice.len(), 2);
        assert_eq!(full_slice[0], BabyBear::from_u32(100));
        assert_eq!(full_slice[1], BabyBear::from_u32(200));
    }

    #[test]
    fn test_mle_index_with_fp4() {
        let coeffs = vec![
            Fp4::from_u32(5),
            Fp4::from_u32(15),
            Fp4::from_u32(25),
            Fp4::from_u32(35),
        ];
        let mle = MLE::new(coeffs);

        assert_eq!(mle[0], Fp4::from_u32(5));
        assert_eq!(mle[3], Fp4::from_u32(35));

        let range_slice = &mle[1..3];
        assert_eq!(range_slice[0], Fp4::from_u32(15));
        assert_eq!(range_slice[1], Fp4::from_u32(25));
    }
}
