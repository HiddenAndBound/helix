use anyhow::Result;
use p3_baby_bear::BabyBear;
use utils::merkle_tree::MerkleTree;

use crate::utils;

// A prover struct for the FRI protocol.
pub struct WHIRCommitment {
    rate: usize, // The rate of the FRI prover
    // Fields for the prover
    merkle_tree: MerkleTree,
    code: Vec<BabyBear>, // Additional field to store codes
}

impl WHIRCommitment {
    /// Constructs a new FRI prover with the given evaluation points and Merkle tree.
    ///
    /// # Arguments
    /// * `evaluation_points` - A vector of evaluation points of type `BabyBear`.
    /// * `merkle_tree` - An instance of `MerkleTree` containing the necessary cryptographic structure.
    ///
    /// # Returns
    /// * `FRIProver` - The constructed prover instance.
    pub fn new(code: Vec<BabyBear>, merkle_tree: MerkleTree, rate: usize) -> Self {
        WHIRCommitment {
            merkle_tree,
            code,
            rate,
        }
    }

    // Additional methods for the prover can be added here
    pub fn commit(
        &mut self,
        domain_evals: &[BabyBear],
        root_table: Option<&[Vec<BabyBear>]>,
    ) -> Result<(), anyhow::Error> {
        let mut buffer = vec![BabyBear::new(0); domain_evals.len() * self.rate];
        buffer[..domain_evals.len()].copy_from_slice(domain_evals);
        // Commit a new code to the prover
        match root_table {
            Some(table) => {
                // If a root table is provided, use it to commit the codes
                BabyBear::forward_fft(&mut buffer, table);
            }
            None => {
                // If no root table is provided, use the evaluation points
                let table = BabyBear::roots_of_unity_table(buffer.len());
                BabyBear::forward_fft(&mut buffer, &table);
            }
        }

        let tree = MerkleTree::new(buffer)?;
        Ok(())
    }
}
