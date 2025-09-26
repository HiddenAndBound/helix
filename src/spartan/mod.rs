mod error;
mod prover;
mod r1cs;
mod spark;
pub mod sumcheck;
pub mod univariate;

pub use prover::SpartanProof;
pub use r1cs::poseidon2::{
    Poseidon2ColumnSeed, Poseidon2Instance, Poseidon2Layout, Poseidon2Witness,
    Poseidon2WitnessMatrix, build_default_poseidon2_instance, build_poseidon2_instance,
    build_poseidon2_witness_matrix,
};
pub use r1cs::{R1CS, R1CSInstance, Witness};
