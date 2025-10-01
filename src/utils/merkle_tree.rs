use anyhow::{ Error, Result };
use p3_field::RawDataSerializable;

/// A Merkle tree structure for cryptographic proofs.
///
/// The tree is constructed from a vector of leaves.
/// The tree uses the BLAKE3 hash function and stores all nodes in a flat vector.
/// The root of the tree is accessible via the `root` field.
pub type MerklePath = Vec<[u8; 32]>;
#[derive(Debug, Clone)]
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
    /// * `leaves` - A vector of serializable elements. The number of leaves must be a power of two.
    ///
    /// # Returns
    /// * `Result<MerkleTree>` - The constructed Merkle tree, or an error if serialization fails.
    ///
    /// # Panics
    /// Panics if the number of leaves is not a power of two.
    pub fn from_field<D: RawDataSerializable + Copy>(leaves: &[D]) -> Result<Self> {
        // Ensure the number of leaves is a power of two for a complete binary tree
        assert!(leaves.len().is_power_of_two(), "Expected leaves to be a power of 2");
        let length = leaves.len();
        let depth = length.trailing_zeros();
        // Allocate space for all nodes (leaves + internal nodes)
        let mut nodes = vec![[0u8; 32]; 2 * leaves.len() - 1];
        // Hash each leaf and store in the first `length` positions
        for (leaf, node) in leaves.iter().zip(&mut nodes) {
            // Serialize the leaf and hash it
            *node = blake3::hash(&leaf.into_bytes().into_iter().collect::<Vec<u8>>()).into();
        }

        // Build the tree layer by layer, hashing pairs of nodes to form parents
        let mut current_layer_start = 0;
        let mut current_layer_size = length;

        for _d in 0..depth {
            let next_layer_start = current_layer_start + current_layer_size;
            let next_layer_size = current_layer_size >> 1;

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

    pub fn from_hash(leaves: &[[u8; 32]]) -> Result<Self> {
        // Ensure the number of leaves is a power of two for a complete binary tree
        assert!(leaves.len().is_power_of_two(), "Expected leaves to be a power of 2");
        let length = leaves.len();
        let depth = length.trailing_zeros();
        // Allocate space for all nodes (leaves + internal nodes)
        let mut nodes = vec![[0u8; 32]; 2 * leaves.len() - 1];
        // Hash each leaf and store in the first `length` positions
        nodes[..leaves.len()].copy_from_slice(&leaves);

        // Build the tree layer by layer, hashing pairs of nodes to form parents
        let mut current_layer_start = 0;
        let mut current_layer_size = length;

        for _d in 0..depth {
            let next_layer_start = current_layer_start + current_layer_size;
            let next_layer_size = current_layer_size >> 1;

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
        (0..self.depth)
            .map(|j| {
                let node_index = (((1 << j) - 1) << (self.depth + 1 - j)) | ((index >> j) ^ 1);
                self.nodes[node_index]
            })
            .collect()
    }

    //TODO: Assert path length is equal to claimed tree depth
    pub fn verify_path(
        leaf: [u8; 32],
        index: usize,
        path: &[[u8; 32]],
        root: [u8; 32]
    ) -> Result<()> {
        let mut current_hash = leaf;
        let mut current_index = index;

        for (_level, &sibling_hash) in path.into_iter().enumerate() {
            // Determine if current node is left or right child
            let is_left = (current_index & 1) == 0;

            // Hash the current node with its sibling in the correct order
            let parent_hash: [u8; 32] = if is_left {
                blake3::hash(&[current_hash, sibling_hash].concat()).into()
            } else {
                blake3::hash(&[sibling_hash, current_hash].concat()).into()
            };

            current_hash = parent_hash;
            current_index >>= 1;
        }

        // Verify that the final computed hash matches the root
        if current_hash == root {
            Ok(())
        } else {
            Err(Error::msg(format!("Merkle path verification failed for index {index}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree() {
        let leaves = (0..1u8 << 5).map(|i| [i; 32]).collect::<Vec<_>>();
        let tree = MerkleTree::from_hash(&leaves).unwrap();
        let path = tree.get_path(15);
        MerkleTree::verify_path(leaves[15], 15, &path, tree.root).unwrap();
    }
}
