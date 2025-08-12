use crate::Fp;
use crate::Fp4;
use crate::challenger;
use crate::challenger::Challenger;
use crate::spartan::CubicSumCheckProof;
use crate::spartan::spark::gpa::ProductTree;

use p3_field::PrimeCharacteristicRing;
pub struct GKRProof {
    layer_proofs: Vec<LayerProof>,
    last_layer_products: Vec<Fp4>,
}

type LayerProof = CubicSumCheckProof;

impl GKRProof {
    pub fn new(layer_proofs: Vec<LayerProof>, last_layer_products: Vec<Fp4>) -> Self {
        Self {
            layer_proofs,
            last_layer_products,
        }
    }

    pub fn prove(circuits: &[ProductTree], challenger: &mut Challenger) {
        let depth = circuits[0].depth();
        circuits.iter().for_each(|c| assert_eq!(c.depth(), depth));

        let initial_claims = circuits
            .iter()
            .map(|c| c.get_final_layer_claims())
            .collect::<Vec<_>>();

        let mut layer_proofs = Vec::<CubicSumCheckProof>::new();
        let mut random_point = Vec::<Fp4>::new();
        let mut layer_claims = Vec::<Vec<(Fp4, Fp4)>>::new();

        let gamma = challenger.get_challenge();

        for d in 0..depth - 1 {
            //Random Challenge.
            let r = challenger.get_challenge();

            //Get the batched layer claim.
            let claim = layer_claims[d]
                .iter()
                .map(|&(left, right)| (Fp4::ONE - r) * left + r * right)
                .fold(Fp4::ZERO, |acc, g| g + acc * gamma);

        }
    }
}


