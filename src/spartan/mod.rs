mod error;
mod prover;
mod r1cs;
mod spark;
pub mod sumcheck;
pub mod univariate;

pub use prover::SpartanProof;
pub use r1cs::poseidon2::{
    Poseidon2Instance,
    Poseidon2Layout,
    Poseidon2Witness,
    build_default_poseidon2_instance,
    build_poseidon2_instance,
};
pub use r1cs::{ R1CS, R1CSInstance, Witness };
