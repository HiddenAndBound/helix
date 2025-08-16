mod error;
mod prover;
mod r1cs;
mod spark;
mod sumcheck;
pub mod univariate;

pub use r1cs::{R1CS, R1CSInstance, Witness};
pub use sumcheck::{
    batched_cubic_sumcheck::{BatchedCubicSumCheckProof, compute_batched_cubic_round},
    cubic_sumcheck::{CubicSumCheckProof, compute_cubic_first_round, compute_cubic_round},
    inner_sumcheck::{
        InnerSumCheckProof, compute_inner_first_round_batched, compute_inner_round_batched,
    },
    outer_sumcheck::{OuterSumCheckProof, compute_first_round, compute_round},
    spark_sumcheck::{
        SparkSumCheckProof, compute_spark_first_round_batched, compute_spark_round_batched,
    },
};
