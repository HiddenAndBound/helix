use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

pub mod challenger;
pub mod eq;
pub mod merkle_tree;
pub mod polynomial;
pub mod sparse;
pub type Fp = BabyBear;
pub type Fp4 = BinomialExtensionField<BabyBear, 4>;
