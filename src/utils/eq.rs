use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

use crate::utils::Fp4;
pub struct Eq<'a> {
    point: &'a [Fp4],
    coeffs: Vec<Fp4>,
    n_vars: usize,
}

impl<'a> Eq<'a> {
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
                coeffs[i] = coeffs[i] + coeffs[i | (1 << var)];
            }
        }

        Self {
            point,
            coeffs,
            n_vars,
        }
    }
}
