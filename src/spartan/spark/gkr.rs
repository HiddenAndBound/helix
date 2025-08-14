use crate::Fp4;
use crate::challenger::Challenger;
use crate::spartan::spark::gpa::ProductTree;
use crate::spartan::sumcheck::BatchedCubicSumCheckProof;
use crate::utils::eq::EqEvals;
use crate::utils::polynomial::MLE;

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
        layer_claims: Vec<Vec<(Fp4, Fp4)>>,
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
        for layer_depth in (0..depth).rev() {
            // Used to generate the sumcheck claim for the current layer, by the fact W(r_0,..., r_{n-2}, r) = (1-r).W(r_0,..., r_{n-2}, 0) + r.W(r_0, ..., r_{n-2}, 1).
            let r = challenger.get_challenge();
            let current_claims = layer_claims[depth - layer_depth]
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
                challenger,
            );

            // Get random challenge for next layer
            layer_claims.push(layer_proof.final_evals.clone());
            layer_proofs.push(layer_proof);
        }

        // Final products are the root values
        let final_products: Vec<Fp4> = circuits.iter().map(|tree| tree.root_value()).collect();

        Self::new(layer_proofs, final_products, layer_claims)
    }

    /// Verifies the batched GKR proof
    pub fn verify(&self, expected_products: &[Fp4], challenger: &mut Challenger) -> bool {
        if self.final_products.len() != expected_products.len() {
            return false;
        }

        // Verify that claimed final products match expected products
        for (claimed, &expected) in self.final_products.iter().zip(expected_products.iter()) {
            if *claimed != expected {
                return false;
            }
        }

        let initial_claims = &self.layer_claims[0];

        for (&(left, right), &expected) in initial_claims.iter().zip(expected_products) {
            assert_eq!(left * right, expected, "Product did not match claim")
        }
        let mut random_point = Vec::new();

        // Initial claims from final layer (should match final products)
        let r_final = challenger.get_challenge();
        random_point.push(r_final);

        let mut current_claims = expected_products.to_vec();

        // Verify each layer proof
        for layer_proof in &self.layer_proofs {
            // Verify the batched cubic sumcheck proof for this layer
            if !layer_proof.verify(&current_claims, challenger) {
                return false;
            }

            // Get random challenge for next layer
            let r = challenger.get_challenge();
            random_point.push(r);

            // Compute next layer claims from final evaluations
            current_claims = layer_proof
                .final_evals
                .iter()
                .map(|&(left_eval, right_eval)| (Fp4::ONE - r) * left_eval + r * right_eval)
                .collect();
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spartan::spark::gpa::{Fingerprints, ProductTree};
    use crate::spartan::spark::sparse::TimeStamps;

    #[test]
    fn test_gkr_proof_single_tree() {
        // Test with a single ProductTree (simplest case)
        let mut challenger = Challenger::new();

        // Create a simple tree with 4 leaves: [1, 2, 3, 4]

        // Build tree layers manually
        // Layer 1 (leaves): left=[1,3], right=[2,4]
        // Layer 0 (root): left=[1*2=2], right=[3*4=12]
        let layer_left = vec![
            vec![Fp4::from_u32(2)],                   // root layer: 1*2 = 2
            vec![Fp4::from_u32(1), Fp4::from_u32(3)], // leaf layer: 1, 3
        ];
        let layer_right = vec![
            vec![Fp4::from_u32(12)],                  // root layer: 3*4 = 12
            vec![Fp4::from_u32(2), Fp4::from_u32(4)], // leaf layer: 2, 4
        ];

        let tree = ProductTree::new(layer_left, layer_right);
        let circuits = vec![tree];

        // Expected product: 1*2*3*4 = 24
        let expected_product = Fp4::from_u32(24);

        // Generate proof
        let proof = GKRProof::prove(&circuits, &mut challenger);

        // Verify proof structure
        assert_eq!(proof.final_products.len(), 1);
        assert_eq!(proof.final_products[0], expected_product);
        assert_eq!(proof.layer_proofs.len(), 1); // 1 layer (leaf only, root is skipped)

        // Verify the proof
        let mut verifier = Challenger::new();
        assert!(proof.verify(&[expected_product], &mut verifier));
    }

    #[test]
    fn test_gkr_proof_multiple_trees() {
        // Test with multiple ProductTrees (core batching scenario)
        let mut challenger = Challenger::new();

        // Create two trees with different values
        // Tree 1: [1, 2, 3, 4] -> product = 24
        let tree1_left = vec![
            vec![Fp4::from_u32(2)],                   // root: 1*2 = 2
            vec![Fp4::from_u32(1), Fp4::from_u32(3)], // leaves: 1, 3
        ];
        let tree1_right = vec![
            vec![Fp4::from_u32(12)],                  // root: 3*4 = 12
            vec![Fp4::from_u32(2), Fp4::from_u32(4)], // leaves: 2, 4
        ];

        // Tree 2: [5, 6, 7, 8] -> product = 1680
        let tree2_left = vec![
            vec![Fp4::from_u32(30)],                  // root: 5*6 = 30
            vec![Fp4::from_u32(5), Fp4::from_u32(7)], // leaves: 5, 7
        ];
        let tree2_right = vec![
            vec![Fp4::from_u32(56)],                  // root: 7*8 = 56
            vec![Fp4::from_u32(6), Fp4::from_u32(8)], // leaves: 6, 8
        ];

        let tree1 = ProductTree::new(tree1_left, tree1_right);
        let tree2 = ProductTree::new(tree2_left, tree2_right);
        let circuits = vec![tree1, tree2];

        let expected_products = vec![Fp4::from_u32(24), Fp4::from_u32(1680)];

        // Generate batched proof
        let proof = GKRProof::prove(&circuits, &mut challenger);

        // Verify proof structure
        assert_eq!(proof.final_products.len(), 2);
        assert_eq!(proof.final_products, expected_products);
        assert_eq!(proof.layer_proofs.len(), 2); // 2 layers

        // Each layer proof should handle 2 trees
        for layer_proof in &proof.layer_proofs {
            assert_eq!(layer_proof.num_claims, 2);
            assert_eq!(layer_proof.final_evals.len(), 2);
        }

        // Verify the proof
        let mut verifier = Challenger::new();
        assert!(proof.verify(&expected_products, &mut verifier));
    }

    #[test]
    fn test_gkr_proof_with_fingerprints() {
        // Test integration with actual memory checking scenario
        use crate::Fp;

        let mut challenger = Challenger::new();

        // Create a small memory table and access pattern
        let memory_table = vec![
            Fp4::from(Fp::from_usize(10)), // table[0] = 10
            Fp4::from(Fp::from_usize(20)), // table[1] = 20
            Fp4::from(Fp::from_usize(30)), // table[2] = 30
            Fp4::from(Fp::from_usize(40)), // table[3] = 40
        ];

        // Memory access pattern: read from addresses [0, 1, 0, 2]
        let read_addresses_fp = vec![
            Fp::from_usize(0), // Access address 0
            Fp::from_usize(1), // Access address 1
            Fp::from_usize(0), // Access address 0 again
            Fp::from_usize(2), // Access address 2
        ];
        let read_values = vec![
            Fp::from_usize(10), // Value at address 0
            Fp::from_usize(20), // Value at address 1
            Fp::from_usize(10), // Value at address 0 (unchanged)
            Fp::from_usize(30), // Value at address 2
        ];

        // Generate timestamps
        let max_address_space = 4;
        let timestamps = TimeStamps::compute(&read_addresses_fp, max_address_space).unwrap();
        let read_timestamps: Vec<Fp> = timestamps
            .read_ts()
            .iter()
            .take(read_addresses_fp.len())
            .cloned()
            .collect();
        let final_timestamps: Vec<Fp> = timestamps.final_ts().to_vec();

        let gamma = Fp4::from_u32(7);
        let tau = Fp4::from_u32(11);

        // Generate fingerprints
        let fingerprints = Fingerprints::generate(
            read_addresses_fp,
            read_values,
            memory_table,
            read_timestamps,
            final_timestamps,
            gamma,
            tau,
        );

        // Generate product trees
        let (left_tree, right_tree) = ProductTree::generate(&fingerprints);
        let circuits = vec![left_tree, right_tree];

        // The products should be equal for a valid memory trace
        let expected_products = vec![circuits[0].root_value(), circuits[1].root_value()];

        // Generate GKR proof
        let proof = GKRProof::prove(&circuits, &mut challenger);

        // Verify proof
        let mut verifier = Challenger::new();
        assert!(proof.verify(&expected_products, &mut verifier));

        // For memory consistency, the products should be equal
        assert_eq!(expected_products[0], expected_products[1]);
    }

    #[test]
    fn test_gkr_proof_empty_circuits() {
        // Test edge case with no circuits
        let mut challenger = Challenger::new();
        let circuits: Vec<ProductTree> = vec![];

        let proof = GKRProof::prove(&circuits, &mut challenger);

        assert_eq!(proof.final_products.len(), 0);
        assert_eq!(proof.layer_proofs.len(), 0);

        let mut verifier = Challenger::new();
        assert!(proof.verify(&[], &mut verifier));
    }

    #[test]
    fn test_gkr_proof_single_layer_tree() {
        // Test with minimal tree (single layer, 2 elements)
        let mut challenger = Challenger::new();

        // Single layer tree: just one multiplication
        let layer_left = vec![
            vec![Fp4::from_u32(3)], // left value
        ];
        let layer_right = vec![
            vec![Fp4::from_u32(5)], // right value
        ];

        let tree = ProductTree::new(layer_left, layer_right);
        let circuits = vec![tree];

        // Expected product: 3 * 5 = 15
        let expected_product = Fp4::from_u32(15);

        // Generate proof
        let proof = GKRProof::prove(&circuits, &mut challenger);

        // Should have exactly 1 layer proof
        assert_eq!(proof.layer_proofs.len(), 1);
        assert_eq!(proof.final_products[0], expected_product);

        // Verify proof
        let mut verifier = Challenger::new();
        assert!(proof.verify(&[expected_product], &mut verifier));
    }

    #[test]
    fn test_gkr_proof_verification_consistency() {
        // Test that prove/verify are consistent with same randomness
        let mut challenger1 = Challenger::new();
        let mut challenger2 = Challenger::new();

        // Create identical trees
        let layer_left = vec![
            vec![Fp4::from_u32(6)],                   // root
            vec![Fp4::from_u32(2), Fp4::from_u32(4)], // leaves
        ];
        let layer_right = vec![
            vec![Fp4::from_u32(35)],                  // root
            vec![Fp4::from_u32(5), Fp4::from_u32(7)], // leaves
        ];

        let tree = ProductTree::new(layer_left, layer_right);
        let circuits = vec![tree];
        let expected_products = vec![Fp4::from_u32(2 * 5 * 4 * 7)]; // = 280

        // Generate proof
        let proof = GKRProof::prove(&circuits, &mut challenger1);

        // Verify with same randomness source
        assert!(proof.verify(&expected_products, &mut challenger2));

        // Test with wrong expected product should fail
        let wrong_products = vec![Fp4::from_u32(123)];
        let mut challenger3 = Challenger::new();
        assert!(!proof.verify(&wrong_products, &mut challenger3));
    }

    #[test]
    fn test_layer_evaluation_propagation() {
        // Test that layer evaluations correctly propagate as claims
        let mut challenger = Challenger::new();

        // Create a two-layer tree to test propagation
        let layer_left = vec![
            vec![Fp4::from_u32(12)],                  // root layer
            vec![Fp4::from_u32(3), Fp4::from_u32(4)], // leaf layer
        ];
        let layer_right = vec![
            vec![Fp4::from_u32(35)],                  // root layer
            vec![Fp4::from_u32(5), Fp4::from_u32(7)], // leaf layer
        ];

        let tree = ProductTree::new(layer_left, layer_right);
        let circuits = vec![tree];

        let proof = GKRProof::prove(&circuits, &mut challenger);

        // Should have 2 layers of proofs
        assert_eq!(proof.layer_proofs.len(), 2);

        // Each layer should have 1 claim (single tree)
        for layer_proof in &proof.layer_proofs {
            assert_eq!(layer_proof.num_claims, 1);
            assert_eq!(layer_proof.final_evals.len(), 1);
        }

        // Final product should match tree root
        assert_eq!(proof.final_products[0], circuits[0].root_value());
    }
}
