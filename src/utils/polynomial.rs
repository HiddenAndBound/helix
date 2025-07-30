use core::slice;

use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, PrimeCharacteristicRing, extension::BinomialExtensionField};

use crate::utils::Fp4;

pub struct MLE {
    coeffs: Vec<BabyBear>,
}

impl MLE {
    pub fn new(coeffs: Vec<BabyBear>) -> Self {
        assert!(coeffs.len().is_power_of_two());
        Self { coeffs }
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }
    pub fn n_vars(&self) -> usize {
        self.coeffs.len().trailing_zeros() as usize
    }
    // Evaluates the mle at the given point
    pub fn evaluate(
        &self,
        point: &[BinomialExtensionField<BabyBear, 4>],
    ) -> BinomialExtensionField<BabyBear, 4> {
        assert_eq!(
            point.len(),
            self.n_vars(),
            "Dimensions of point must match MLE variables"
        );
        let mut eval = Fp4::ZERO;

        eval
    }
}
