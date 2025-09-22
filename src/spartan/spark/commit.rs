use crate::challenger::Challenger;
use crate::pcs::{ BaseFoldConfig, Basefold, BasefoldCommitment, EvalProof, ProverData };
use crate::spartan::spark::sparse::SparkMetadata;
use crate::{ Fp, Fp4 };

type Commitment = [u8; 32];
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

    pub fn evaluate(&self, prover_data: SparkProverData, evaluations: [Fp4; 7]) {}
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
