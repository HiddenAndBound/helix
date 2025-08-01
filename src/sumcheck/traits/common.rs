//! Common types and structures used across sumcheck traits.

use p3_field::Field;

/// Sumcheck proof structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckProof<F>
where
    F: Field,
{
    /// The intermediate polynomials for each round.
    pub polynomials: Vec<Vec<F>>,
    
    /// The challenges used in each round.
    pub challenges: Vec<F>,
    
    /// The final evaluation.
    pub final_evaluation: F,
}

/// Optimization hints for the prover.
#[derive(Debug, Clone, Default)]
pub struct ProverHints {
    /// Whether to use optimized multiplication strategies.
    pub use_optimized_mul: bool,
    
    /// Whether to use batch processing for multiple polynomials.
    pub use_batch_processing: bool,
    
    /// Whether to use parallel processing.
    pub use_parallel_processing: bool,
    
    /// The target memory usage in bytes.
    pub target_memory_usage: Option<usize>,
    
    /// The target computational complexity.
    pub target_complexity: Option<usize>,
}

/// Computational cost estimate for the prover.
#[derive(Debug, Clone, Default)]
pub struct ProverCost {
    /// The estimated number of field multiplications.
    pub field_multiplications: usize,
    
    /// The estimated number of field additions.
    pub field_additions: usize,
    
    /// The estimated memory usage in bytes.
    pub memory_usage: usize,
    
    /// The estimated time complexity.
    pub time_complexity: usize,
}

/// Verification hints for the verifier.
#[derive(Debug, Clone, Default)]
pub struct VerifierHints {
    /// Whether to use optimized verification strategies.
    pub use_optimized_verification: bool,
    
    /// Whether to use batch verification for multiple proofs.
    pub use_batch_verification: bool,
    
    /// Whether to use parallel processing.
    pub use_parallel_processing: bool,
    
    /// The target memory usage in bytes.
    pub target_memory_usage: Option<usize>,
    
    /// The target computational complexity.
    pub target_complexity: Option<usize>,
}

/// Computational cost estimate for the verifier.
#[derive(Debug, Clone, Default)]
pub struct VerifierCost {
    /// The estimated number of field multiplications.
    pub field_multiplications: usize,
    
    /// The estimated number of field additions.
    pub field_additions: usize,
    
    /// The estimated memory usage in bytes.
    pub memory_usage: usize,
    
    /// The estimated time complexity.
    pub time_complexity: usize,
}