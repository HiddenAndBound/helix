use crate::Fp4;
use crate::challenger::Challenger;
use crate::spartan::spark::gpa::ProductTree;
use crate::spartan::sumcheck::BatchedCubicSumCheckProof;
use crate::utils::eq::EqEvals;
use crate::utils::polynomial::MLE;

use anyhow::bail;
use p3_field::PrimeCharacteristicRing;
pub struct GKRProof {
    /// Batched proofs for each layer (one proof per layer for all trees)
    layer_proofs: Vec<LayerProof>,
    /// Final product values for all trees
    final_products: Vec<Fp4>,

    layer_claims: Vec<Vec<(Fp4, Fp4)>>,
}

type LayerProof = BatchedCubicSumCheckProof;

impl GKRProof {
    pub fn new(
        layer_proofs: Vec<LayerProof>,
        final_products: Vec<Fp4>,
        layer_claims: Vec<Vec<(Fp4, Fp4)>>
    ) -> Self {
        Self {
            layer_proofs,
            final_products,
            layer_claims,
        }
    }

    pub fn prove(circuits: &[ProductTree], challenger: &mut Challenger) -> Self {
        let depth = circuits[0].depth();
        let num_trees = circuits.len();

        // Validate all trees have same depth
        circuits.iter().for_each(|c| assert_eq!(c.depth(), depth));

        let mut layer_proofs = Vec::new();
        let mut layer_claims = Vec::new();

        // Initial claims from final layer of all trees (leaves)
        let initial_claims: Vec<(Fp4, Fp4)> = circuits
            .iter()
            .map(|c| c.get_final_layer_claims())
            .collect();

        layer_claims.push(initial_claims);
        // Process each layer from leaves towards root (depth-1 down to 1)
        // Skip the root layer (depth 0) as it has only 1 element
        for layer_depth in (0..depth - 1).rev() {
            // Used to generate the sumcheck claim for the current layer, by the fact W(r_0,..., r_{n-2}, r) = (1-r).W(r_0,..., r_{n-2}, 0) + r.W(r_0, ..., r_{n-2}, 1).
            let r = challenger.get_challenge();
            let current_claims = layer_claims
                .last()
                .unwrap()
                .iter()
                .map(|&(left, right)| (Fp4::ONE - r) * left + r * right)
                .collect::<Vec<_>>();
            // Create MLEs from ProductTree layer data for all trees
            let left_mles: Vec<MLE<Fp4>> = circuits
                .iter()
                .map(|tree| MLE::new(tree.get_layer_left(layer_depth).clone()))
                .collect();
            let right_mles: Vec<MLE<Fp4>> = circuits
                .iter()
                .map(|tree| MLE::new(tree.get_layer_right(layer_depth).clone()))
                .collect();
            // Generate batched proof for this layer

            let layer_proof = BatchedCubicSumCheckProof::prove(
                &left_mles,
                &right_mles,
                &current_claims,
                challenger
            );

            // Get random challenge for next layer
            layer_claims.push(layer_proof.final_evals.clone());
            layer_proofs.push(layer_proof);
        }
        
        // Final products are the root values
        let final_products: Vec<Fp4> = circuits
            .iter()
            .map(|tree| tree.root_value())
            .collect();

        Self::new(layer_proofs, final_products, layer_claims)
    }

    /// Verifies the batched GKR proof
    pub fn verify(
        &self,
        expected_products: &[Fp4],
        challenger: &mut Challenger
    ) -> anyhow::Result<()> {
        if self.final_products.len() != expected_products.len() {
            bail!("Expected product length is not equal to final products length.");
        }

        let initial_claims = &self.layer_claims[0];

        for (&(left, right), &expected) in initial_claims.iter().zip(expected_products) {
            assert_eq!(left * right, expected, "Product did not match claim");
        }
        let mut random_point = Vec::new();

        // Initial claims from final layer (should match final products)
        let r = challenger.get_challenge();
        random_point.push(r);

        let mut current_claims = initial_claims
            .iter()
            .map(|&(left, right)| (Fp4::ONE - r) * left + r * right)
            .collect::<Vec<_>>();

        // Verify each layer proof
        for layer_proof in &self.layer_proofs {
            // Verify the batched cubic sumcheck proof for this layer
            layer_proof.verify(&current_claims, challenger)?;

            // Get random challenge for next layer
            let r = challenger.get_challenge();
            random_point.push(r);

            // Compute next layer claims from final evaluations
            current_claims = layer_proof.final_evals
                .iter()
                .map(|&(left_eval, right_eval)| (Fp4::ONE - r) * left_eval + r * right_eval)
                .collect();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use rand::{ rngs::StdRng, thread_rng, Rng, SeedableRng };

    use crate::{
        Fp4,
        challenger::{ self, Challenger },
        polynomial::MLE,
        spartan::spark::{ gkr::GKRProof, gpa::ProductTree },
    };

    #[test]
    // This tests the gkr protocol on a randomly generated product tree.
    pub fn gpa_test() {
        let k = 1 << 5;

        // Build both trees inline
        fn build_tree(mut leaves: Vec<Fp4>, actual_product: Fp4) -> ProductTree {
            let input_size = leaves.len();
            let depth = (input_size as f64).log2() as usize;
            let mut layer_left = Vec::with_capacity(depth);
            let mut layer_right = Vec::with_capacity(depth);

            // Build tree from leaves up to root using for loop
            for level in (0..depth).rev() {
                let level_size = 1 << level; // 2^level
                let mut left_half = vec![Fp4::ZERO; level_size];
                let mut right_half = vec![Fp4::ZERO; level_size];
                let mut next_level = vec![Fp4::ZERO; level_size];

                for i in 0..level_size {
                    let left = leaves[2 * i];
                    let right = leaves[2 * i + 1];

                    left_half[i] = left;
                    right_half[i] = right;
                    next_level[i] = left * right;
                }

                layer_left.push(left_half);
                layer_right.push(right_half);

                if level > 0 {
                    leaves = next_level;
                }
            }

            ProductTree {
                layer_left,
                layer_right,
                depth,
                input_size,
                root_value: actual_product,
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let leaves = MLE::new((0..k).map(|_| Fp4::from_u128(rng.r#gen())).collect());

        let actual_product = leaves.coeffs().iter().copied().product();
        let tree = build_tree(leaves.coeffs().to_owned(), actual_product);

        let mut challenger = Challenger::new();

        let proof = GKRProof::prove(&[tree], &mut challenger);

        let mut challenger = Challenger::new();

        proof.verify(&[actual_product], &mut challenger).unwrap();
    }
}
