use std::{ io::Write, thread::{ available_parallelism, scope } };

use anyhow::{ Error, Result };
use hybrid_array::Array;
use p3_field::RawDataSerializable;
use rayon::{ iter::{ IndexedParallelIterator, ParallelIterator }, slice::ParallelSlice };
use serde::Serialize;
use sha3::{ digest::OutputSizeUser, Digest, Keccak256, Keccak256Full };

use crate::pcs::utils::fill_buf;
/// A Merkle tree structure for cryptographic proofs.
///
/// The tree is constructed from a vector of leaves.
/// The tree uses the BLAKE3 hash function and stores all nodes in a flat vector.
/// The root of the tree is accessible via the `root` field.
pub type MerklePath = Vec<HashOutput>;

pub type HashOutput = [u8; 32];
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

    #[tracing::instrument(level = "debug", skip_all)]
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
            let (prev_layer, next_layer) =
                nodes[current_layer_start..next_layer_start + next_layer_size].split_at_mut(
                    current_layer_size
                );

            parallel_layer(prev_layer, next_layer);
            current_layer_start = next_layer_start;
            current_layer_size = next_layer_size;
        }

        // The root is the last node in the vector
        Ok(MerkleTree {
            root: nodes[nodes.len() - 1].into(),
            nodes,
            depth: depth as u32,
        })
    }
    /// Returns the root hash of the Merkle tree.
    pub fn root(&self) -> HashOutput {
        self.root
    }

    /// Returns a merkle path for the given index.
    pub fn get_path(&self, index: usize) -> Vec<HashOutput> {
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
        leaf: HashOutput,
        index: usize,
        path: &MerklePath,
        root: HashOutput
    ) -> Result<()> {
        let mut current_hash = leaf;
        let mut current_index = index;

        for (_level, &sibling_hash) in path.into_iter().enumerate() {
            // Determine if current node is left or right child
            let is_left = (current_index & 1) == 0;

            // Hash the current node with its sibling in the correct order
            let parent_hash = if is_left {
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

#[inline(always)]
pub fn fill_buf_digests(left: [u8; 32], right: [u8; 32], buf: &mut [u8; 64]) {
    buf[0..32].copy_from_slice(&left);
    buf[32..64].copy_from_slice(&right);
}

fn parallel_layer(prev_layer: &mut [[u8; 32]], next_layer: &mut [[u8; 32]]) {
    let threads = available_parallelism().unwrap().get();
    scope(|s| {
        let chunk_size = next_layer.len().div_ceil(threads);
        let mut parent_chunks = next_layer.chunks_mut(chunk_size);
        let mut children_chunks = prev_layer.chunks_mut(2 * chunk_size);
        for _ in 0..parent_chunks.len() {
            let parent_chunk = parent_chunks.next().unwrap();
            let children_chunk = children_chunks.next().unwrap();
            assert_eq!(children_chunk.len(), parent_chunk.len() * 2);
            s.spawn(|| {
                let mut hasher = blake3::Hasher::new();
                let mut buffer = [0u8; 64];

                for (children, parent) in children_chunk.chunks_exact(2).zip(parent_chunk) {
                    fill_buf_digests(children[0], children[1], &mut buffer);
                    *parent = hasher.update(&buffer).finalize().into();
                    hasher.reset();
                }
            });
        }
    });
}

fn sequential_layer(prev_layer: &mut [[u8; 32]], next_layer: &mut [[u8; 32]]) {
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64];

    prev_layer
        .chunks_exact(2)
        .zip(next_layer.iter_mut())
        .for_each(|(children, parent)| {
            fill_buf_digests(children[0], children[1], &mut buffer);
            *parent = hasher.update(&buffer).finalize().into();
            hasher.reset();
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree() {
        let leaves = (0..1u8 << 5).map(|i| HashOutput::default()).collect::<Vec<_>>();
        let tree = MerkleTree::from_hash(&leaves).unwrap();
        let path = tree.get_path(15);
        MerkleTree::verify_path(leaves[15], 15, &path, tree.root).unwrap();
    }
}
