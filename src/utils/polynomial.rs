use p3_field::PrimeCharacteristicRing;

use crate::utils::{Fp4, eq::EqEvals};

pub struct MLE<F: PrimeCharacteristicRing + Clone> {
    coeffs: Vec<F>,
}

impl<F: PrimeCharacteristicRing + Clone> MLE<F> {
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
        Fp4: From<F>,
    {
        assert_eq!(
            point.len(),
            self.n_vars(),
            "Dimensions of point must match MLE variables"
        );

        let eq = EqEvals::gen_from_point(point);

        eq.coeffs
            .iter()
            .zip(self.coeffs.iter())
            .map(|(&x, y)| x * Fp4::from(y.clone()))
            .sum()
    }

    /// Folds the MLE by binding the lowest variable to a challenge value.
    /// Automatically promotes to Fp4 if needed, consuming self and returning MLE<Fp4>.
    ///
    /// For MLE f(x₀, x₁, ..., xₙ₋₁), this computes:
    /// g(x₁, ..., xₙ₋₁) = f(challenge, x₁, ..., xₙ₋₁)
    ///                  = (1 - challenge) * f(0, x₁, ..., xₙ₋₁) + challenge * f(1, x₁, ..., xₙ₋₁)
    pub fn fold_in_place(self, challenge: Fp4) -> MLE<Fp4>
    where
        Fp4: From<F>,
    {
        if self.coeffs.len() == 1 {
            // Base case: 0-variable polynomial, promote to Fp4 and return
            return MLE::new(vec![Fp4::from(self.coeffs[0].clone())]);
        }

        let half_len = self.coeffs.len() / 2;
        let mut folded_coeffs = Vec::with_capacity(half_len);

        // For each coefficient pair (low, high) where low corresponds to x₀=0 and high to x₀=1
        // In hypercube layout, we pair coefficients that differ only in the lowest bit
        for i in 0..half_len {
            let low_idx = i * 2; // Even indices: x₀=0
            let high_idx = i * 2 + 1; // Odd indices: x₀=1

            let low = Fp4::from(self.coeffs[low_idx].clone()); // f(..., x₀=0) promoted to Fp4
            let high = Fp4::from(self.coeffs[high_idx].clone()); // f(..., x₀=1) promoted to Fp4

            // Compute (1 - challenge) * low + challenge * high
            let folded = low * (Fp4::ONE - challenge) + high * challenge;
            folded_coeffs.push(folded);
        }

        MLE::new(folded_coeffs)
    }

    /// Returns a reference to the coefficient vector (for testing/debugging)
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_fold_two_variables() {
        // Test MLE with 2 variables: f(x₀, x₁) with coeffs [a₀₀, a₁₀, a₀₁, a₁₁]
        let coeffs = vec![
            BabyBear::from_u32(1), // f(0,0)
            BabyBear::from_u32(2), // f(1,0)
            BabyBear::from_u32(3), // f(0,1)
            BabyBear::from_u32(4), // f(1,1)
        ];
        let mle = MLE::new(coeffs);
        let challenge = Fp4::from_u32(5);

        let folded = mle.fold_in_place(challenge);

        assert_eq!(folded.n_vars(), 1);
        assert_eq!(folded.coeffs().len(), 2);

        // Folded should represent g(x₁) = f(5, x₁)
        // g(0) = (1-5)*f(0,0) + 5*f(1,0) = -4*1 + 5*2 = 6
        // g(1) = (1-5)*f(0,1) + 5*f(1,1) = -4*3 + 5*4 = 8
        let expected_g0 = (Fp4::ONE - challenge) * Fp4::from_u32(1) + challenge * Fp4::from_u32(2);
        let expected_g1 = (Fp4::ONE - challenge) * Fp4::from_u32(3) + challenge * Fp4::from_u32(4);

        assert_eq!(folded.coeffs()[0], expected_g0);
        assert_eq!(folded.coeffs()[1], expected_g1);
    }

    #[test]
    fn test_fold_with_fp4_coefficients() {
        // Test MLE that already has Fp4 coefficients
        let coeffs = vec![
            Fp4::from_u32(1), // f(0,0)
            Fp4::from_u32(2), // f(1,0)
            Fp4::from_u32(3), // f(0,1)
            Fp4::from_u32(4), // f(1,1)
        ];
        let mle = MLE::new(coeffs);
        let challenge = Fp4::from_u32(5);

        let folded = mle.fold_in_place(challenge);

        assert_eq!(folded.n_vars(), 1);
        assert_eq!(folded.coeffs().len(), 2);

        // Same expected results as base field case
        let expected_g0 = (Fp4::ONE - challenge) * Fp4::from_u32(1) + challenge * Fp4::from_u32(2);
        let expected_g1 = (Fp4::ONE - challenge) * Fp4::from_u32(3) + challenge * Fp4::from_u32(4);

        assert_eq!(folded.coeffs()[0], expected_g0);
        assert_eq!(folded.coeffs()[1], expected_g1);
    }

    #[test]
    fn test_fold_three_variables() {
        // Test MLE with 3 variables: f(x₀, x₁, x₂) with 8 coefficients
        let coeffs = vec![
            BabyBear::from_u32(1), // f(0,0,0)
            BabyBear::from_u32(2), // f(1,0,0)
            BabyBear::from_u32(3), // f(0,1,0)
            BabyBear::from_u32(4), // f(1,1,0)
            BabyBear::from_u32(5), // f(0,0,1)
            BabyBear::from_u32(6), // f(1,0,1)
            BabyBear::from_u32(7), // f(0,1,1)
            BabyBear::from_u32(8), // f(1,1,1)
        ];
        let mle = MLE::new(coeffs);
        let challenge = Fp4::from_u32(3);

        let folded = mle.fold_in_place(challenge);

        assert_eq!(folded.n_vars(), 2);
        assert_eq!(folded.coeffs().len(), 4);

        // Folded should represent g(x₁, x₂) = f(3, x₁, x₂)
        // g(0,0) = (1-3)*f(0,0,0) + 3*f(1,0,0) = -2*1 + 3*2 = 4
        // g(1,0) = (1-3)*f(0,1,0) + 3*f(1,1,0) = -2*3 + 3*4 = 6
        // g(0,1) = (1-3)*f(0,0,1) + 3*f(1,0,1) = -2*5 + 3*6 = 8
        // g(1,1) = (1-3)*f(0,1,1) + 3*f(1,1,1) = -2*7 + 3*8 = 10

        let one_minus_challenge = Fp4::ONE - challenge;
        assert_eq!(
            folded.coeffs()[0],
            one_minus_challenge * Fp4::from_u32(1) + challenge * Fp4::from_u32(2)
        );
        assert_eq!(
            folded.coeffs()[1],
            one_minus_challenge * Fp4::from_u32(3) + challenge * Fp4::from_u32(4)
        );
        assert_eq!(
            folded.coeffs()[2],
            one_minus_challenge * Fp4::from_u32(5) + challenge * Fp4::from_u32(6)
        );
        assert_eq!(
            folded.coeffs()[3],
            one_minus_challenge * Fp4::from_u32(7) + challenge * Fp4::from_u32(8)
        );
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
}
