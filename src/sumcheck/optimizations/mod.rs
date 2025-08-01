//! Optimization strategies for the enhanced sum-check module.
//! 
//! This module contains various optimization implementations that can be
//! plugged into the sumcheck protocol to improve performance for specific
//! use cases.

pub mod small_value;
pub mod karatsuba;
pub mod skip_rounds;

// Re-export the main optimization types for convenience
pub use small_value::{SmallValueProver, SmallValueComposer};
pub use karatsuba::{KaratsubaProver, KaratsubaComposer};
pub use skip_rounds::{SkipRoundsProver, SkipRoundsComposer};