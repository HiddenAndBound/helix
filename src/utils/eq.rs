use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};

use crate::utils::Fp4;
pub struct Eq<'a> {
    point: &'a [Fp4],
    coeffs: Vec<Fp4>,
    n_vars: usize,
}

impl<'a> Eq<'a>{
    pub fn new(point: &'a [Fp4], coeffs: Vec<Fp4>, n_vars: usize) -> Self {
        Self {
            point,
            coeffs,
            n_vars,
        }
    }

    pub fn gen_from_point(point: &'a [Fp4])->Self{
        let coeffs = vec![Fp4::ZERO]
    }
}
