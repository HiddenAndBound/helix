use std::fs::Metadata;

use crate::challenger::Challenger;
use crate::eq::EqEvals;
use crate::pcs::utils::Commitment;
use crate::pcs::{ BaseFoldConfig, Basefold, BasefoldCommitment, EvalProof, ProverData };
use crate::spartan::spark::oracles::{ generate_oracle_pair, generate_spark_opening_oracles };
use crate::spartan::spark::sparse::SparkMetadata;
use crate::{ Fp, Fp4 };

pub struct SparkProof {}

impl SparkProof {
    pub fn prove(
        metadata: [SparkMetadata; 3],
        commitments: [SparkCommitment; 3],
        prover_data: [SparkProverData; 3],
        rx: &[Fp4],
        ry: &[Fp4],
        claimed_evaluations: [Fp4; 3]
    ) -> anyhow::Result<SparkProof> {
        let [metadata_a, metadata_b, metadata_c] = &metadata;
        let (eq_rx, eq_ry) = (EqEvals::gen_from_point(rx), EqEvals::gen_from_point(ry));
        let oracles = generate_oracle_pair(metadata_a, metadata_b, metadata_c, &eq_rx, &eq_ry)?;

        //Spark sum check

        //Offline memory check to prove correctness of oracles

        //Opening all the commitments.
        todo!()
    }
}
#[derive(Debug, Clone)]
pub struct SparkCommitment {
    /// Row indices as multilinear extension
    row: Commitment,
    /// Column indices as multilinear extension
    col: Commitment,
    /// Coefficient values as multilinear extension
    val: Commitment,
    /// Timestamp information for row accesses
    row_read_ts: Commitment,
    row_final_ts: Commitment,
    col_read_ts: Commitment,
    col_final_ts: Commitment,
}

impl SparkCommitment {
    /// Creates a new
    pub fn new(
        row: Commitment,
        col: Commitment,
        val: Commitment,
        row_read_ts: Commitment,
        row_final_ts: Commitment,
        col_read_ts: Commitment,
        col_final_ts: Commitment
    ) -> Self {
        Self {
            row,
            col,
            val,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
        }
    }
}

/// Prover-side data corresponding to each Spark commitment component.
#[derive(Debug)]
pub struct SparkProverData {
    row: ProverData,
    col: ProverData,
    val: ProverData,
    row_read_ts: ProverData,
    row_final_ts: ProverData,
    col_read_ts: ProverData,
    col_final_ts: ProverData,
}

impl SparkProverData {
    /// Creates a new `SparkProverData` wrapper around the component prover artifacts.
    pub fn new(
        row: ProverData,
        col: ProverData,
        val: ProverData,
        row_read_ts: ProverData,
        row_final_ts: ProverData,
        col_read_ts: ProverData,
        col_final_ts: ProverData
    ) -> Self {
        Self {
            row,
            col,
            val,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
        }
    }
}
