use anyhow::{Error, Result};
use p3_baby_bear::BabyBear;

/// A Merkle tree structure for cryptographic proofs.
///
/// The tree is constructed from a vector of leaves, each of type `BabyBear`.
/// The tree uses the BLAKE3 hash function and stores all nodes in a flat vector.
/// The root of the tree is accessible via the `root` field.
pub struct MerkleTree {
    /// The root hash of the Merkle tree (top node).
    pub root: [u8; 32],
    /// All nodes of the Merkle tree, including leaves and internal nodes.
    /// The leaves are stored first, followed by each layer up to the root.
    pub nodes: Vec<[u8; 32]>,

    pub depth: u32,
}

impl MerkleTree {
    /// Constructs a new Merkle tree from the given leaves.
    ///
    /// # Arguments
    /// * `leaves` - A vector of `BabyBear` elements. The number of leaves must be a power of two.
    ///
    /// # Returns
    /// * `Result<MerkleTree>` - The constructed Merkle tree, or an error if serialization fails.
    ///
    /// # Panics
    /// Panics if the number of leaves is not a power of two.
    pub fn new(leaves: Vec<BabyBear>) -> Result<Self> {
        // Ensure the number of leaves is a power of two for a complete binary tree
        assert!(
            leaves.len().is_power_of_two(),
            "Expected leaves to be a power of 2"
        );
        let length = leaves.len();
        let depth = length.trailing_zeros();
        // Allocate space for all nodes (leaves + internal nodes)
        let mut nodes = vec![[0u8; 32]; 2 * leaves.len() - 1];

        // Hash each leaf and store in the first `length` positions
        for (leaf, node) in leaves.iter().zip(&mut nodes) {
            // Serialize the leaf and hash it
            let bytes = serde_json::to_vec(leaf)?;
            *node = blake3::hash(&bytes).into();
        }

        // Build the tree layer by layer, hashing pairs of nodes to form parents
        let mut current_layer_start = 0;
        let mut current_layer_size = length;

        for _d in 0..depth {
            let next_layer_start = current_layer_start + current_layer_size;
            let next_layer_size = current_layer_size / 2;

            for i in 0..next_layer_size {
                // Hash the left and right child to get the parent node
                let left = nodes[current_layer_start + (i << 1)];
                let right = nodes[current_layer_start + (i << 1) + 1];
                let parent_hash: [u8; 32] = blake3::hash(&[left, right].concat()).into();
                nodes[next_layer_start + i] = parent_hash;
            }

            current_layer_start = next_layer_start;
            current_layer_size = next_layer_size;
        }

        // The root is the last node in the vector
        Ok(MerkleTree {
            root: nodes[nodes.len() - 1],
            nodes,
            depth: depth as u32,
        })
    }

    /// Returns the root hash of the Merkle tree.
    pub fn root(&self) -> [u8; 32] {
        self.root
    }

    /// Returns a merkle path for the given index.
    pub fn get_path(&self, index: usize) -> Vec<[u8; 32]> {
        assert!(index < 1 << self.depth, "Index out of range.");
        
        let mut path = Vec::new();
        let mut current_index = index;
        let mut layer_start = 0;
        let mut layer_size = 1 << self.depth;
        
        for _ in 0..self.depth {
            // Find sibling index
            let sibling_index = current_index ^ 1;
            // Add sibling hash to path
            path.push(self.nodes[layer_start + sibling_index]);
            
            // Move to next layer
            current_index = current_index / 2;
            layer_start += layer_size;
            layer_size /= 2;
        }
        
        path
    }

    pub fn verify_path(&self, index: usize, path: Vec<[u8; 32]>) -> Result<()> {
        assert!(index < 1 << self.depth, "Index out of range.");
        let mut current_hash = self.nodes[index];
        let mut current_index = index;
        
        for (_level, sibling_hash) in path.into_iter().enumerate() {
            // Determine if current node is left or right child
            let is_left = current_index % 2 == 0;
            
            // Hash the current node with its sibling in the correct order
            let parent_hash: [u8; 32] = if is_left {
                blake3::hash(&[current_hash, sibling_hash].concat()).into()
            } else {
                blake3::hash(&[sibling_hash, current_hash].concat()).into()
            };
            
            current_hash = parent_hash;
            current_index = current_index / 2;
        }
        
        // Verify that the final computed hash matches the root
        if current_hash == self.root {
            Ok(())
        } else {
            Err(Error::msg("Merkle path verification failed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_leaves(count: usize) -> Vec<BabyBear> {
        (0..count).map(|i| BabyBear::new(i as u32)).collect()
    }

    #[test]
    fn test_new_single_leaf() {
        let leaves = create_test_leaves(1);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.depth, 0);
        assert_eq!(tree.nodes.len(), 1);
    }

    #[test]
    fn test_new_two_leaves() {
        let leaves = create_test_leaves(2);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.depth, 1);
        assert_eq!(tree.nodes.len(), 3);
    }

    #[test]
    fn test_new_four_leaves() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.depth, 2);
        assert_eq!(tree.nodes.len(), 7);
    }

    #[test]
    fn test_new_eight_leaves() {
        let leaves = create_test_leaves(8);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.depth, 3);
        assert_eq!(tree.nodes.len(), 15);
    }

    #[test]
    #[should_panic(expected = "Expected leaves to be a power of 2")]
    fn test_new_non_power_of_two_panics() {
        let leaves = create_test_leaves(3);
        MerkleTree::new(leaves).unwrap();
    }

    #[test]
    #[should_panic(expected = "Expected leaves to be a power of 2")]
    fn test_new_five_leaves_panics() {
        let leaves = create_test_leaves(5);
        MerkleTree::new(leaves).unwrap();
    }

    #[test]
    fn test_root_consistency() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.root(), tree.root);
        assert_eq!(tree.root, tree.nodes[tree.nodes.len() - 1]);
    }

    #[test]
    fn test_get_path_single_leaf() {
        let leaves = create_test_leaves(1);
        let tree = MerkleTree::new(leaves).unwrap();
        
        let path = tree.get_path(0);
        assert_eq!(path.len(), 0);
    }

    #[test]
    fn test_get_path_two_leaves() {
        let leaves = create_test_leaves(2);
        let tree = MerkleTree::new(leaves).unwrap();
        
        let path0 = tree.get_path(0);
        let path1 = tree.get_path(1);
        
        assert_eq!(path0.len(), 1);
        assert_eq!(path1.len(), 1);
        assert_eq!(path0[0], tree.nodes[1]);
        assert_eq!(path1[0], tree.nodes[0]);
    }

    #[test]
    fn test_get_path_four_leaves() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        for i in 0..4 {
            let path = tree.get_path(i);
            assert_eq!(path.len(), 2);
        }
    }

    #[test]
    #[should_panic(expected = "Index out of range")]
    fn test_get_path_index_out_of_range() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        tree.get_path(4);
    }

    #[test]
    fn test_verify_path_valid() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        for i in 0..4 {
            let path = tree.get_path(i);
            assert!(tree.verify_path(i, path).is_ok());
        }
    }

    #[test]
    fn test_verify_path_invalid() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        let path = tree.get_path(0);
        let mut invalid_path = path.clone();
        invalid_path[0][0] ^= 1;
        
        assert!(tree.verify_path(0, invalid_path).is_err());
    }

    #[test]
    #[should_panic(expected = "Index out of range")]
    fn test_verify_path_index_out_of_range() {
        let leaves = create_test_leaves(4);
        let tree = MerkleTree::new(leaves).unwrap();
        
        let path = tree.get_path(0);
        tree.verify_path(4, path).unwrap();
    }

    #[test]
    fn test_different_inputs_different_trees() {
        let leaves1 = create_test_leaves(4);
        let mut leaves2 = create_test_leaves(4);
        leaves2[0] = BabyBear::new(100);
        
        let tree1 = MerkleTree::new(leaves1).unwrap();
        let tree2 = MerkleTree::new(leaves2).unwrap();
        
        assert_ne!(tree1.root, tree2.root);
    }

    #[test]
    fn test_tree_structure_integrity() {
        let leaves = create_test_leaves(8);
        let tree = MerkleTree::new(leaves).unwrap();
        
        assert_eq!(tree.depth, 3);
        assert_eq!(tree.nodes.len(), 15);
        
        for i in 0..8 {
            let path = tree.get_path(i);
            assert_eq!(path.len(), 3);
            assert!(tree.verify_path(i, path).is_ok());
        }
    }
}