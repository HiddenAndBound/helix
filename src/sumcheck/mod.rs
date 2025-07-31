use crate::utils::{Fp, Fp4};
use anyhow::Result;

pub mod prover;
pub mod verifier;
pub mod sparse;

pub use prover::*;
pub use verifier::*;
pub use sparse::*;

/// Sumcheck proof containing round polynomials and final evaluations
#[derive(Debug, Clone)]
pub struct SumcheckProof {
    /// Univariate polynomials for each round (degree 2, stored as [f(0), f(1), f(2)])
    pub round_polynomials: Vec<Vec<Fp4>>,
    /// Final evaluations at the random point
    pub final_evaluations: Vec<Fp4>,
}

impl SumcheckProof {
    pub fn new() -> Self {
        Self {
            round_polynomials: Vec::new(),
            final_evaluations: Vec::new(),
        }
    }
    
    pub fn num_rounds(&self) -> usize {
        self.round_polynomials.len()
    }
}