use std::{ops::Index, vec};


use crate::Fp4;

//Offline Memory Check
pub struct OfflineMemoryCheck {
    fingerprints: Vec<Fingerprints>,
    product_trees: Vec<ProductTree>,
}

pub struct Fingerprints {
    w_init: Vec<Fp4>,
    w: Vec<Fp4>,
    r: Vec<Fp4>,
    s: Vec<Fp4>,
}

pub struct ProductTree {
    layer_left: Vec<Vec<Fp4>>,
    layer_right: Vec<Vec<Fp4>>,
}


