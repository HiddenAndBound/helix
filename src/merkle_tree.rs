use anyhow::{Error, Result};
use p3_baby_bear::BabyBear;
use std::hash::Hash;

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

        // Markers for the current and next layer in the flat node vector
        let mut previous_layer_marker = 0;
        let mut next_layer_marker = length;

        // Build the tree layer by layer, hashing pairs of nodes to form parents
        for d in 0..depth {
            // Split the current segment into previous and next layers
            let (previous_layer, next_layer) = nodes
                [previous_layer_marker..next_layer_marker + (length >> (d + 1))]
                .split_at_mut(length >> d);

            for i in 0..(length >> (d + 1)) {
                // Hash the left and right child to get the parent node
                let left = previous_layer[i << 1];
                let right = previous_layer[(i << 1) | 1];
                let parent_hash: [u8; 32] = blake3::hash(&[left, right].concat()).into();
                next_layer[i] = parent_hash;

                // Update the markers for the next layer using XOR to move up the tree
                previous_layer_marker ^= length >> d;
                next_layer_marker ^= length >> (d + 1);
            }
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
        (0..self.depth)
            .map(|j| {
                let node_index = (((1 << j) - 1) << (self.depth + 1 - j)) | (index >> j) ^ 1;
                self.nodes[node_index]
            })
            .collect()
    }

    pub fn verify_path(&self, index: usize, path: Vec<[u8; 32]>) -> Result<()> {
        assert!(index < 1 << self.depth, "Index out of range.");
        let mut current_hash = self.nodes[index];
        for (i, sibling_hash) in path.into_iter().enumerate() {
            // Hash the current node with its sibling to get the parent
            let parent_hash: [u8; 32] = blake3::hash(&[current_hash, sibling_hash].concat()).into();
            // Check if the computed parent hash matches the stored parent hash
            if parent_hash != self.nodes[(index >> (i + 1)) ^ 1] {
                return Err(Error::msg("Merkle path verification failed"));
            }
            current_hash = parent_hash;
        }
        Ok(())
    }
}
