//! Configuration system for the enhanced sum-check module.
//! 
//! This module provides runtime configuration control over optimization selection,
//! allowing users to enable/disable specific optimizations based on their needs.
//! 
//! # Examples
//! 
//! ```rust
//! use deep_fri::sumcheck::config::SumcheckConfig;
//! 
//! // Create default configuration
//! let config = SumcheckConfig::default();
//! 
//! // Create custom configuration
//! let custom_config = SumcheckConfig {
//!     max_degree: 5,
//!     use_small_opt: false,
//!     use_karatsuba: true,
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for the enhanced sum-check protocol.
/// 
/// This struct provides runtime control over optimization selection,
/// allowing fine-tuning of performance characteristics based on use case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SumcheckConfig {
    /// Maximum degree supported for low-degree composition
    pub max_degree: usize,
    
    /// Enable small-value field splits optimization (ePrint 2025/1117)
    pub use_small_opt: bool,
    
    /// Enable Karatsuba-style reduction optimization (ePrint 2024/1046)
    pub use_karatsuba: bool,
    
    /// Threshold for unequal-degree skip optimization (ePrint 2024/108)
    pub skip_threshold: usize,
    
    /// Memory limit for large computations (in bytes)
    pub memory_limit: Option<usize>,
    
    /// Threshold for parallel computation activation
    pub parallel_threshold: usize,
    
    /// Enable debug mode with additional verification
    pub debug_mode: bool,
    
    /// Enable performance profiling
    pub profile_mode: bool,
}

impl Default for SumcheckConfig {
    fn default() -> Self {
        Self {
            max_degree: 3,
            use_small_opt: true,
            use_karatsuba: true,
            skip_threshold: 8,
            memory_limit: None,
            parallel_threshold: 1024,
            debug_mode: false,
            profile_mode: false,
        }
    }
}

impl SumcheckConfig {
    /// Creates a new configuration with all optimizations enabled.
    pub fn optimized() -> Self {
        Self {
            use_small_opt: true,
            use_karatsuba: true,
            ..Default::default()
        }
    }
    
    /// Creates a new configuration with all optimizations disabled.
    pub fn basic() -> Self {
        Self {
            use_small_opt: false,
            use_karatsuba: false,
            skip_threshold: usize::MAX, // Effectively disabled
            ..Default::default()
        }
    }
    
    /// Creates a configuration for testing purposes.
    pub fn test() -> Self {
        Self {
            debug_mode: true,
            profile_mode: true,
            ..Default::default()
        }
    }
    
    /// Returns true if small-value optimization is enabled.
    pub fn use_small_value_optimization(&self) -> bool {
        self.use_small_opt
    }
    
    /// Returns true if Karatsuba optimization is enabled.
    pub fn use_karatsuba_optimization(&self) -> bool {
        self.use_karatsuba
    }
    
    /// Returns true if skip-round optimization is enabled for the given round.
    pub fn should_skip_round(&self, round: usize) -> bool {
        round >= self.skip_threshold
    }
    
    /// Returns true if parallel computation should be used.
    pub fn use_parallel_computation(&self, input_size: usize) -> bool {
        input_size >= self.parallel_threshold
    }
    
    /// Validates the configuration for consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_degree == 0 {
            return Err("max_degree must be at least 1".to_string());
        }
        
        if self.skip_threshold == 0 {
            return Err("skip_threshold must be at least 1".to_string());
        }
        
        if let Some(limit) = self.memory_limit {
            if limit < 1024 {
                return Err("memory_limit must be at least 1024 bytes".to_string());
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SumcheckConfig::default();
        assert_eq!(config.max_degree, 3);
        assert!(config.use_small_opt);
        assert!(config.use_karatsuba);
        assert_eq!(config.skip_threshold, 8);
    }

    #[test]
    fn test_optimized_config() {
        let config = SumcheckConfig::optimized();
        assert!(config.use_small_opt);
        assert!(config.use_karatsuba);
    }

    #[test]
    fn test_basic_config() {
        let config = SumcheckConfig::basic();
        assert!(!config.use_small_opt);
        assert!(!config.use_karatsuba);
        assert_eq!(config.skip_threshold, usize::MAX);
    }

    #[test]
    fn test_config_validation() {
        assert!(SumcheckConfig::default().validate().is_ok());
        
        let mut invalid = SumcheckConfig::default();
        invalid.max_degree = 0;
        assert!(invalid.validate().is_err());
        
        let mut invalid = SumcheckConfig::default();
        invalid.skip_threshold = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_optimization_checks() {
        let config = SumcheckConfig::default();
        assert!(config.use_small_value_optimization());
        assert!(config.use_karatsuba_optimization());
        assert!(!config.should_skip_round(5));
        assert!(config.should_skip_round(10));
        assert!(!config.use_parallel_computation(100));
        assert!(config.use_parallel_computation(2000));
    }
}