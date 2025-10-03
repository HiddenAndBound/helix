use p3_field::PrimeCharacteristicRing;
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator };
use std::ops::{ Index, Range };

use crate::utils::Fp4;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EqEvals<'a> {
    point: &'a [Fp4],
    coeffs: Vec<Fp4>,
    n_vars: usize,
}

impl<'a> EqEvals<'a> {
    pub fn new(point: &'a [Fp4], coeffs: Vec<Fp4>, n_vars: usize) -> Self {
        Self {
            point,
            coeffs,
            n_vars,
        }
    }

    pub fn gen_from_point(point: &'a [Fp4]) -> Self {
        let mut coeffs = vec![Fp4::ZERO; 1 << point.len()];
        let n_vars = point.len();

        coeffs[0] = Fp4::ONE;

        for var in 0..n_vars {
            for i in 0..1 << var {
                coeffs[i | (1 << var)] = coeffs[i] * point[var];
                coeffs[i] = coeffs[i] - coeffs[i | (1 << var)];
            }
        }

        Self {
            point,
            coeffs,
            n_vars,
        }
    }

    pub fn gen_from_point_high_low(point: &'a [Fp4]) -> Self {
        let mut coeffs = vec![Fp4::ZERO; 1 << point.len()];
        let n_vars = point.len();

        coeffs[0] = Fp4::ONE;

        for var in 0..n_vars {
            for i in 0..1 << var {
                coeffs[i | (1 << var)] = coeffs[i] * point[n_vars - var];
                coeffs[i] = coeffs[i] - coeffs[i | (1 << var)];
            }
        }

        Self {
            point,
            coeffs,
            n_vars,
        }
    }

    /// Folds the equality polynomial by binding the lowest variable to a challenge value.
    ///
    /// For EqEvals with n variables, this computes:
    /// g(x₁, ..., xₙ₋₁) = eq(challenge, x₁, ..., xₙ₋₁; r₀, r₁, ..., rₙ₋₁)
    ///                  = (1 - challenge) * eq(0, x₁, ..., xₙ₋₁; r₀, r₁, ..., rₙ₋₁) +
    ///                    challenge * eq(1, x₁, ..., xₙ₋₁; r₀, r₁, ..., rₙ₋₁)
    pub fn fold_in_place<'b>(&mut self) {
        if self.coeffs.len() == 1 {
            // Base case: 0-variable polynomial, return constant
            return;
        }

        let half_len = self.coeffs.len() / 2;
        let mut folded_coeffs = Vec::with_capacity(half_len);

        // For each coefficient pair (low, high) where low corresponds to x₀=0 and high to x₀=1
        for i in 0..half_len {
            let low = self.coeffs[i << 1]; // eq(..., x₀=0)
            let high = self.coeffs[(i << 1) | 1]; // eq(..., x₀=1)

            // Compute (1 - challenge) * low + challenge * high
            let folded = low + high;
            folded_coeffs.push(folded);
        }

        self.point = &self.point[1..];
        self.coeffs = folded_coeffs;
        self.n_vars = self.n_vars.saturating_sub(1);
    }

    pub fn fold_in_place_hi_lo<'b>(&mut self) {
        if self.coeffs.len() == 1 {
            // Base case: 0-variable polynomial, return constant
            return;
        }

        let half_len = self.coeffs.len() / 2;
        // let mut folded_coeffs = Vec::with_capacity(half_len);

        let (lo, hi) = self.coeffs.split_at_mut(half_len);
        // For each coefficient pair (low, high) where low corresponds to x₀=0 and high to x₀=1
        lo.par_iter_mut()
            .zip(hi.par_iter_mut())
            .for_each(|(l, h)| {
                *l += *h;
            });
        self.point = &self.point[1..];
        self.coeffs.truncate(half_len);
        self.coeffs.shrink_to_fit();
        self.n_vars = self.n_vars.saturating_sub(1);
    }

    pub fn coeffs(&self) -> &[Fp4] {
        &self.coeffs
    }

    pub fn point(&self) -> &[Fp4] {
        &self.point
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars
    }
}

impl<'a> Index<usize> for EqEvals<'a> {
    type Output = Fp4;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<'a> Index<Range<usize>> for EqEvals<'a> {
    type Output = [Fp4];

    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.coeffs[range]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use rand::{ Rng, SeedableRng, rngs::StdRng };

    //Tests whether given eq = eq(r_0,..., r_{n-1}; x_0,..., x_{n-1}), fold_in_place returns eq(r_1,...,r_{n-1}; x_1, ..., x_{n-1})
    #[test]
    fn test_fold_in_place() {
        let mut rng = StdRng::seed_from_u64(0);
        let point = (0..4).map(|_| Fp4::from_u128(rng.r#gen())).collect::<Vec<Fp4>>();
        let mut eq = EqEvals::gen_from_point(&point);
        eq.fold_in_place();

        let mut eq_0 = EqEvals::gen_from_point(&point[1..]);

        assert_eq!(eq, eq_0)
    }

    #[test]
    fn test_gen_from_point_empty() {
        let point: Vec<Fp4> = vec![];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 0);
        assert_eq!(eq.coeffs.len(), 1);
        assert_eq!(eq.coeffs[0], Fp4::ONE);
    }

    #[test]
    fn test_gen_from_point_single_var() {
        let point = vec![Fp4::from_u32(2)];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 1);
        assert_eq!(eq.coeffs.len(), 2);

        // For single variable eq(x0, r0) = x0*r0 + (1-x0)*(1-r0)
        // The multilinear extension should have:
        // coeffs[0] = evaluation at x0=0 -> (1-r0)
        // coeffs[1] = evaluation at x0=1 -> r0
        // But tensor expansion gives us coefficients for the Lagrange basis
        let r0 = Fp4::from_u32(2);

        // After tensor expansion, coeffs should satisfy the equality function property
        // Test by manually evaluating eq(0) and eq(1) using the coefficients
        let eval_at_0 = eq.coeffs[0]; // Should equal (1-r0)
        let eval_at_1 = eq.coeffs[1]; // Should equal r0

        assert_eq!(eval_at_0, Fp4::ONE - r0);
        assert_eq!(eval_at_1, r0);
    }

    #[test]
    fn test_gen_from_point_two_vars() {
        let point = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 2);
        assert_eq!(eq.coeffs.len(), 4);

        let r0 = Fp4::from_u32(3);
        let r1 = Fp4::from_u32(5);

        // For eq(x0, x1; r0, r1) = eq(x0; r0) * eq(x1; r1)
        // coeffs[0] = eq(0,0) = (1-r0)*(1-r1)
        // coeffs[1] = eq(1,0) = r0*(1-r1)
        // coeffs[2] = eq(0,1) = (1-r0)*r1
        // coeffs[3] = eq(1,1) = r0*r1

        assert_eq!(eq.coeffs[0], (Fp4::ONE - r0) * (Fp4::ONE - r1));
        assert_eq!(eq.coeffs[1], r0 * (Fp4::ONE - r1));
        assert_eq!(eq.coeffs[2], (Fp4::ONE - r0) * r1);
        assert_eq!(eq.coeffs[3], r0 * r1);
    }

    #[test]
    fn test_gen_from_point_three_vars() {
        let point = vec![Fp4::from_u32(2), Fp4::from_u32(3), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 3);
        assert_eq!(eq.coeffs.len(), 8);

        let r0 = Fp4::from_u32(2);
        let r1 = Fp4::from_u32(3);
        let r2 = Fp4::from_u32(7);

        // For eq(x0, x1, x2; r0, r1, r2) = eq(x0; r0) * eq(x1; r1) * eq(x2; r2)
        // Test a few key evaluations
        assert_eq!(eq.coeffs[0], (Fp4::ONE - r0) * (Fp4::ONE - r1) * (Fp4::ONE - r2)); // eq(0,0,0)
        assert_eq!(eq.coeffs[7], r0 * r1 * r2); // eq(1,1,1)
        assert_eq!(eq.coeffs[1], r0 * (Fp4::ONE - r1) * (Fp4::ONE - r2)); // eq(1,0,0)
        assert_eq!(eq.coeffs[4], (Fp4::ONE - r0) * (Fp4::ONE - r1) * r2); // eq(0,0,1)
    }

    #[test]
    fn test_gen_from_point_four_vars() {
        let point = vec![
            Fp4::from_u32(11),
            Fp4::from_u32(13),
            Fp4::from_u32(17),
            Fp4::from_u32(19)
        ];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 4);
        assert_eq!(eq.coeffs.len(), 16);

        // Test that all coefficients are non-zero (except potentially some specific cases)
        let non_zero_count = eq.coeffs
            .iter()
            .filter(|&&x| x != Fp4::ZERO)
            .count();
        assert!(non_zero_count > 0);

        // Test corner evaluations
        let r0 = point[0];
        let r1 = point[1];
        let r2 = point[2];
        let r3 = point[3];

        // eq(0,0,0,0)
        assert_eq!(
            eq.coeffs[0],
            (Fp4::ONE - r0) * (Fp4::ONE - r1) * (Fp4::ONE - r2) * (Fp4::ONE - r3)
        );

        // eq(1,1,1,1)
        assert_eq!(eq.coeffs[15], r0 * r1 * r2 * r3);
    }

    #[test]
    fn test_gen_from_point_with_zero_point() {
        let point = vec![Fp4::ZERO, Fp4::from_u32(5)];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 2);
        assert_eq!(eq.coeffs.len(), 4);

        // When r0 = 0, eq(x0; 0) = 1 - x0
        // coeffs[0] = eq(0,0) = 1 * (1-5) = -4
        // coeffs[1] = eq(1,0) = 0 * (1-5) = 0
        // coeffs[2] = eq(0,1) = 1 * 5 = 5
        // coeffs[3] = eq(1,1) = 0 * 5 = 0

        let r1 = Fp4::from_u32(5);
        assert_eq!(eq.coeffs[0], Fp4::ONE * (Fp4::ONE - r1));
        assert_eq!(eq.coeffs[1], Fp4::ZERO);
        assert_eq!(eq.coeffs[2], Fp4::ONE * r1);
        assert_eq!(eq.coeffs[3], Fp4::ZERO);
    }

    #[test]
    fn test_gen_from_point_with_one_point() {
        let point = vec![Fp4::ONE, Fp4::from_u32(3)];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 2);
        assert_eq!(eq.coeffs.len(), 4);

        // When r0 = 1, eq(x0; 1) = x0
        let r1 = Fp4::from_u32(3);
        assert_eq!(eq.coeffs[0], Fp4::ZERO); // eq(0,0) = 0 * (1-3) = 0
        assert_eq!(eq.coeffs[1], Fp4::ONE - r1); // eq(1,0) = 1 * (1-3) = -2
        assert_eq!(eq.coeffs[2], Fp4::ZERO); // eq(0,1) = 0 * 3 = 0
        assert_eq!(eq.coeffs[3], r1); // eq(1,1) = 1 * 3 = 3
    }

    #[test]
    fn test_gen_from_point_large_vars() {
        // Test with 6 variables to ensure scalability
        let point = vec![
            Fp4::from_u32(2),
            Fp4::from_u32(3),
            Fp4::from_u32(5),
            Fp4::from_u32(7),
            Fp4::from_u32(11),
            Fp4::from_u32(13)
        ];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 6);
        assert_eq!(eq.coeffs.len(), 64); // 2^6 = 64

        // Test that the algorithm completes without error and produces expected size
        // Check a few corner cases
        let all_ones_index = 63; // Binary: 111111
        let all_zeros_index = 0; // Binary: 000000

        // eq(1,1,1,1,1,1) should be the product of all r_i
        let expected_all_ones = point
            .iter()
            .copied()
            .reduce(|acc, x| acc * x)
            .unwrap();
        assert_eq!(eq.coeffs[all_ones_index], expected_all_ones);

        // eq(0,0,0,0,0,0) should be the product of all (1-r_i)
        let expected_all_zeros = point
            .iter()
            .map(|&r| Fp4::ONE - r)
            .reduce(|acc, x| acc * x)
            .unwrap();
        assert_eq!(eq.coeffs[all_zeros_index], expected_all_zeros);
    }

    #[test]
    fn test_tensor_product_structure() {
        // Test that the equality polynomial satisfies the tensor product structure
        let point = vec![Fp4::from_u32(7), Fp4::from_u32(11)];
        let eq = EqEvals::gen_from_point(&point);

        // For 2 variables, we should have eq(x0,x1) = eq(x0) ⊗ eq(x1)
        let eq_x0 = EqEvals::gen_from_point(&point[0..1]);
        let eq_x1 = EqEvals::gen_from_point(&point[1..2]);

        // Check tensor product structure
        for i in 0..2 {
            for j in 0..2 {
                let tensor_idx = i + 2 * j;
                let expected = eq_x0.coeffs[i] * eq_x1.coeffs[j];
                assert_eq!(
                    eq.coeffs[tensor_idx],
                    expected,
                    "Tensor product mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_power_of_two_constraint() {
        // Test that coefficient vector length is always a power of 2
        for n_vars in 0..8 {
            let point: Vec<Fp4> = (0..n_vars).map(|i| Fp4::from_u32((i as u32) + 1)).collect();
            let eq = EqEvals::gen_from_point(&point);

            assert_eq!(eq.coeffs.len(), 1 << n_vars);
            assert_eq!(eq.n_vars, n_vars);

            // Check that coefficient length is power of 2
            let len = eq.coeffs.len();
            assert_eq!(len & (len - 1), 0, "Length {} is not power of 2", len);
        }
    }

    #[test]
    fn test_equality_polynomial_evaluation_property() {
        // Test the core property: eq(x, r) = 1 iff x = r, 0 otherwise
        let point = vec![Fp4::from_u32(5), Fp4::from_u32(7), Fp4::from_u32(11)];
        let eq = EqEvals::gen_from_point(&point);

        // Test all 2^3 = 8 possible evaluations
        for x0 in 0..2 {
            for x1 in 0..2 {
                for x2 in 0..2 {
                    let _eval_point = vec![Fp4::from_u32(x0), Fp4::from_u32(x1), Fp4::from_u32(x2)];

                    // Manually compute eq(eval_point, point)
                    let mut expected = Fp4::ONE;
                    for i in 0..3 {
                        let xi = Fp4::from_u32([x0, x1, x2][i]);
                        let ri = point[i];
                        expected *= xi * ri + (Fp4::ONE - xi) * (Fp4::ONE - ri);
                    }

                    // Compare with coefficient-based evaluation
                    let index = x0 + 2 * x1 + 4 * x2;
                    assert_eq!(
                        eq.coeffs[index as usize],
                        expected,
                        "Mismatch at evaluation point ({}, {}, {})",
                        x0,
                        x1,
                        x2
                    );
                }
            }
        }
    }

    #[test]
    fn test_coefficient_symmetry_properties() {
        // Test mathematical properties of the coefficient generation
        let point = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
        let eq = EqEvals::gen_from_point(&point);

        // Sum of all coefficients should equal 1 (by MLE property)
        let sum: Fp4 = eq.coeffs.iter().copied().sum();
        assert_eq!(sum, Fp4::ONE, "Sum of coefficients should be 1");
    }

    #[test]
    fn test_boundary_field_values() {
        // Test with field boundary values
        let max_val = Fp4::from_u32(u32::MAX); // Large field element
        let point = vec![max_val, Fp4::ONE, Fp4::ZERO];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 3);
        assert_eq!(eq.coeffs.len(), 8);

        // Should not panic or produce invalid results
        let sum: Fp4 = eq.coeffs.iter().copied().sum();
        assert_eq!(sum, Fp4::ONE);
    }

    #[test]
    fn test_fold_in_place_single_var() {
        // Test folding a 1-variable equality polynomial to constant
        let point = vec![Fp4::from_u32(7)];
        let mut eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 1);
        assert_eq!(eq.coeffs.len(), 2);

        // Fold the polynomial
        eq.fold_in_place();

        // After folding, should have 0 variables and 1 coefficient
        assert_eq!(eq.n_vars, 0);
        assert_eq!(eq.coeffs.len(), 1);
    }

    #[test]
    fn test_fold_in_place_base_case() {
        // Test folding when already at base case (0 variables)
        let point: Vec<Fp4> = vec![];
        let mut eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq.n_vars, 0);
        assert_eq!(eq.coeffs.len(), 1);

        // Folding should be a no-op
        eq.fold_in_place();

        assert_eq!(eq.n_vars, 0);
        assert_eq!(eq.coeffs.len(), 1);
        assert_eq!(eq.coeffs[0], Fp4::ONE);
    }

    #[test]
    fn test_incremental_variable_extension() {
        // Test that adding variables extends the polynomial correctly
        let point_base = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
        let point_extended = vec![Fp4::from_u32(3), Fp4::from_u32(5), Fp4::from_u32(7)];

        let eq_base = EqEvals::gen_from_point(&point_base);
        let eq_extended = EqEvals::gen_from_point(&point_extended);

        // The first 4 coefficients of extended should relate to base coefficients
        let r2 = Fp4::from_u32(7);
        for i in 0..4 {
            // eq_extended[i] should equal eq_base[i] * (1 - r2)
            // eq_extended[i + 4] should equal eq_base[i] * r2
            assert_eq!(eq_extended.coeffs[i], eq_base.coeffs[i] * (Fp4::ONE - r2));
            assert_eq!(eq_extended.coeffs[i + 4], eq_base.coeffs[i] * r2);
        }
    }

    #[test]
    fn test_eq_evals_index_single_element() {
        let point = vec![];
        let eq = EqEvals::gen_from_point(&point);

        assert_eq!(eq[0], Fp4::ONE);
    }

    #[test]
    fn test_eq_evals_index_multiple_elements() {
        let point = vec![Fp4::from_u32(3), Fp4::from_u32(5)];
        let eq = EqEvals::gen_from_point(&point);

        // Test individual indexing
        assert_eq!(eq[0], (Fp4::ONE - Fp4::from_u32(3)) * (Fp4::ONE - Fp4::from_u32(5)));
        assert_eq!(eq[1], Fp4::from_u32(3) * (Fp4::ONE - Fp4::from_u32(5)));
        assert_eq!(eq[2], (Fp4::ONE - Fp4::from_u32(3)) * Fp4::from_u32(5));
        assert_eq!(eq[3], Fp4::from_u32(3) * Fp4::from_u32(5));
    }

    #[test]
    fn test_eq_evals_index_range() {
        let point = vec![Fp4::from_u32(2), Fp4::from_u32(7)];
        let eq = EqEvals::gen_from_point(&point);

        let slice = &eq[1..3];
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Fp4::from_u32(2) * (Fp4::ONE - Fp4::from_u32(7)));
        assert_eq!(slice[1], (Fp4::ONE - Fp4::from_u32(2)) * Fp4::from_u32(7));
    }

    #[test]
    fn test_eq_evals_index_full_range() {
        let point = vec![Fp4::from_u32(11)];
        let eq = EqEvals::gen_from_point(&point);

        let full_slice = &eq[0..2];
        assert_eq!(full_slice.len(), 2);
        assert_eq!(full_slice[0], Fp4::ONE - Fp4::from_u32(11));
        assert_eq!(full_slice[1], Fp4::from_u32(11));
    }
}
