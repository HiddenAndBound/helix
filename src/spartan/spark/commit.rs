use std::fs::Metadata;

use crate::challenger::Challenger;
use crate::eq::EqEvals;
use crate::pcs::utils::Commitment;
use crate::pcs::{BaseFoldConfig, Basefold, BasefoldCommitment, EvalProof, ProverData};
use crate::spartan::spark::gkr::GKRProof;
use crate::spartan::spark::gpa::{Fingerprints, ProductTree};
use crate::spartan::spark::oracles::{
    SparkOracles, generate_oracle_pair, generate_spark_opening_oracles,
};
use crate::spartan::spark::sparse::SparkMetadata;
use crate::spartan::sumcheck::SparkSumCheckProof;
use crate::{Fp, Fp4};

pub struct SparkProof {
    sumcheck_proof: SparkSumCheckProof,
    product_proofs: Vec<GKRProof>,
}

impl SparkProof {
    // TODO: Logup* and complete batching.
    pub fn prove(
        metadata: &[SparkMetadata; 3],
        commitments: &[SparkCommitment; 3],
        prover_data: &[SparkProverData; 3],
        rx: &[Fp4],
        ry: &[Fp4],
        claimed_evaluations: [Fp4; 3],
        challenger: &mut Challenger,
    ) -> anyhow::Result<SparkProof> {
        let [metadata_a, metadata_b, metadata_c] = &metadata;
        let (eq_rx, eq_ry) = (EqEvals::gen_from_point(rx), EqEvals::gen_from_point(ry));
        let oracles = generate_oracle_pair(metadata_a, metadata_b, metadata_c, &eq_rx, &eq_ry)?;

        let gamma = challenger.get_challenge();
        //Spark sum check
        SparkSumCheckProof::prove(metadata, &oracles, claimed_evaluations, gamma, challenger);

        // Offline memory check to prove correctness of oracles

        // First generate fingerprints for row and col lookups for 3 matrices.
        let gamma = challenger.get_challenge();
        let tau = challenger.get_challenge();

        let mut fingerprints = Vec::new();

        let SparkOracles {
            e_rx_a,
            e_ry_a,
            e_rx_b,
            e_ry_b,
            e_rx_c,
            e_ry_c,
        } = oracles;

        let oracle_refs = [(&e_rx_a, &e_ry_a), (&e_rx_b, &e_ry_b), (&e_rx_c, &e_ry_c)];

        for (
            SparkMetadata {
                row,
                col,
                row_read_ts,
                row_final_ts,
                col_read_ts,
                col_final_ts,
                ..
            },
            (e_rx, e_ry),
        ) in metadata.iter().zip(oracle_refs)
        {
            let row_fingerprint = Fingerprints::generate(
                &row,
                &e_rx,
                &eq_rx.coeffs(),
                row_read_ts,
                row_final_ts,
                gamma,
                tau,
            );

            let col_fingerprint = Fingerprints::generate(
                &col,
                &e_ry,
                &eq_ry.coeffs(),
                col_read_ts,
                col_final_ts,
                gamma,
                tau,
            );

            fingerprints.push(row_fingerprint);
            fingerprints.push(col_fingerprint);
        }

        // Generate product trees

        let product_trees = fingerprints
            .iter()
            .map(ProductTree::generate)
            .collect::<Vec<_>>();

        //Generate GKR Proofs
        let mut proofs = Vec::new();
        for (w_tree, r_tree) in product_trees {
            let w_proof = GKRProof::prove(&[w_tree], challenger);
            let r_proof = GKRProof::prove(&[r_tree], challenger);

            proofs.push(w_proof);
            proofs.push(r_proof);
        }

        //Opening all the commitments.
        todo!()
    }

    pub fn verify(
        &self,
        commitment: SparkCommitment,
        rx: &[Fp4],
        ry: &[Fp4],
    ) -> anyhow::Result<()> {
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
        col_final_ts: Commitment,
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
        col_final_ts: ProverData,
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
