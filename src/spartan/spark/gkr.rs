use crate::Fp;
use crate::Fp4;
use crate::challenger;
use crate::challenger::Challenger;
use crate::spartan::{CubicSumCheckProof, BatchedCubicSumCheckProof};
use crate::spartan::spark::gpa::ProductTree;
use crate::utils::polynomial::MLE;
use crate::utils::eq::EqEvals;

use p3_field::PrimeCharacteristicRing;
pub struct GKRProof {
    /// Batched proofs for each layer (one proof per layer for all trees)
    layer_proofs: Vec<LayerProof>,
    /// Final product values for all trees
    final_products: Vec<Fp4>,
}

type LayerProof = BatchedCubicSumCheckProof;

impl GKRProof {
    pub fn new(layer_proofs: Vec<LayerProof>, final_products: Vec<Fp4>) -> Self {
        Self {
            layer_proofs,
            final_products,
        }
    }

    pub fn prove(circuits: &[ProductTree], challenger: &mut Challenger) -> Self {
        let depth = circuits[0].depth();
        let num_trees = circuits.len();
        
        // Validate all trees have same depth
        circuits.iter().for_each(|c| assert_eq!(c.depth(), depth));
        
        if num_trees == 0 {
            return Self::new(vec![], vec![]);
        }

        let mut layer_proofs = Vec::new();
        let mut random_point = Vec::new();
        
        // Initial claims from final layer of all trees (leaves)
        let r_final = challenger.get_challenge();
        random_point.push(r_final);
        
        let mut current_claims: Vec<Fp4> = circuits
            .iter()
            .map(|tree| {
                let (left, right) = tree.get_final_layer_claims();
                (Fp4::ONE - r_final) * left + r_final * right
            })
            .collect();
        
        // Process each layer from leaves towards root (depth-1 down to 0)
        for layer_depth in (0..depth).rev() {
            // Create MLEs from ProductTree layer data for all trees
            let left_mles: Vec<MLE<Fp4>> = circuits.iter()
                .map(|tree| MLE::new(tree.layer_left[layer_depth].clone()))
                .collect();
            let right_mles: Vec<MLE<Fp4>> = circuits.iter()
                .map(|tree| MLE::new(tree.layer_right[layer_depth].clone()))
                .collect();
                
            // Generate equality polynomials for current layer
            // All trees use the same accumulated random point
            let eq_evals: Vec<EqEvals> = (0..num_trees)
                .map(|_| EqEvals::gen_from_point(&random_point))
                .collect();
            
            // Generate batched proof for this layer
            let layer_proof = BatchedCubicSumCheckProof::prove(
                &left_mles,
                &right_mles, 
                &eq_evals,
                &current_claims,
                challenger,
            );
            
            // Get random challenge for next layer
            let r = challenger.get_challenge();
            random_point.push(r);
            
            // Extract final evaluations as next layer claims
            current_claims = layer_proof.final_evals.iter()
                .map(|&(left_eval, right_eval)| (Fp4::ONE - r) * left_eval + r * right_eval)
                .collect();
                
            layer_proofs.push(layer_proof);
        }
        
        // Final products are the root values
        let final_products: Vec<Fp4> = circuits.iter()
            .map(|tree| tree.root_value())
            .collect();
            
        Self::new(layer_proofs, final_products)
    }
}


