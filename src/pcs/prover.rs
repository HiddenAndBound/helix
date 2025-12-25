//! Prover-side helpers for the BaseFold-style commitment layer.

use anyhow::Result;

use crate::Fp4;
use crate::merkle::{HashOutput, MerkleTree};
use crate::pcs::utils::create_hash_leaves_std;

/// Commits to an already-encoded oracle by building a Merkle tree over leaf hashes.
#[tracing::instrument(level = "debug", skip_all)]
pub fn commit_oracle(current_encoding: &[Fp4]) -> Result<(HashOutput, MerkleTree)> {
    let leaves = create_hash_leaves_std(current_encoding);
    let merkle_tree = MerkleTree::from_hash(&leaves)?;
    Ok((merkle_tree.root(), merkle_tree))
}

/// Updates all query indices for the next folding round.
pub fn update_queries(queries: &mut Vec<usize>, halfsize: usize) {
    for query in queries {
        update_query(query, halfsize);
    }
}

/// Updates a single query index for the next folding round.
pub fn update_query(query: &mut usize, halfsize: usize) {
    debug_assert!(halfsize.is_power_of_two(), "halfsize must be a power of 2");
    *query &= halfsize - 1;
}
