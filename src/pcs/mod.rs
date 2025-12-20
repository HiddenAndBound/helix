//! BaseFold-style PCS helpers used by the Helix batch sum-check.
//!
//! This is a placeholder commitment layer: it exists to support the golden-path
//! `examples/helix.rs` flow and should not be treated as a production-sound PCS.

use crate::merkle::MerkleTree;
use crate::pcs::utils::{Commitment, Encoding};

pub mod prover;
pub mod utils;
pub mod verifier;

/// Configuration parameters for the BaseFold-style commitment helpers.
#[derive(Debug, Clone)]
pub struct BaseFoldConfig {
    /// Number of random queries for consistency verification.
    pub queries: usize,
    /// Reed-Solomon encoding rate (expansion factor), must be a power of two.
    pub rate: usize,
    /// Enable parallel processing for folding operations.
    pub enable_parallel: bool,
    /// Number of folding rounds to skip for performance.
    pub round_skip: usize,
    /// Early stopping threshold (used by the protocol layer).
    pub early_stopping_threshold: usize,
}

impl Default for BaseFoldConfig {
    fn default() -> Self {
        Self {
            queries: 144,
            rate: 2,
            enable_parallel: false,
            round_skip: 0,
            early_stopping_threshold: 0,
        }
    }
}

impl BaseFoldConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_queries(mut self, queries: usize) -> Self {
        self.queries = queries;
        self
    }

    pub fn with_rate(mut self, rate: usize) -> Self {
        self.rate = rate;
        self
    }

    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    pub fn with_early_stopping(mut self, early_stopping_threshold: usize) -> Self {
        self.early_stopping_threshold = early_stopping_threshold;
        self
    }

    pub fn with_round_skip(mut self, skip_rounds: usize) -> Self {
        self.round_skip = skip_rounds;
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.queries == 0 {
            anyhow::bail!("Query count must be greater than 0");
        }
        if self.rate == 0 || !self.rate.is_power_of_two() {
            anyhow::bail!("Rate must be a positive power of 2");
        }
        Ok(())
    }
}

/// Commitment to an encoded oracle, represented by a Merkle root.
#[derive(Debug, Clone)]
pub struct BasefoldCommitment {
    pub commitment: Commitment,
}

/// Prover-side data retained after committing.
#[derive(Debug)]
pub struct ProverData {
    pub merkle_tree: MerkleTree,
    pub encoding: Encoding,
}
