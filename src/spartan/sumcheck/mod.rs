//! Sumcheck protocol implementations for Spartan zkSNARK.
//!
//! This module provides implementations for various sumcheck protocols:
//! - OuterSumCheck: Proves f(x) = A(x)·B(x) - C(x) = 0 over Boolean hypercube
//! - InnerSumCheck: Proves f(x) = ⟨A(x), B(x)⟩ for inner product constraints
//! - CubicSumCheck: Proves f(x) = left(x) * right(x) * eq(x) for cubic product constraints
//! - SparkSumCheck: Proves f(x) = A(x)·B(x)·C(x) for triple product constraints
//! - BatchedCubicSumCheck: Handles multiple cubic claims efficiently

pub mod outer_sumcheck;
pub mod inner_sumcheck;
pub mod cubic_sumcheck;
pub mod spark_sumcheck;
pub mod batched_cubic_sumcheck;