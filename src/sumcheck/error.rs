use std::fmt;

/// Error types for the sumcheck protocol
#[derive(Debug, Clone, PartialEq)]
pub enum SumCheckError {
    /// Invalid number of variables
    InvalidNumVariables {
        expected: usize,
        actual: usize,
    },
    
    /// Invalid number of MLEs
    InvalidNumMles {
        expected: usize,
        actual: usize,
    },
    
    /// Mismatch in number of variables across MLEs
    VariableCountMismatch {
        expected: usize,
        actual: usize,
    },
    
    /// Invalid degree for polynomial
    InvalidDegree {
        max_supported: usize,
        actual: usize,
    },
    
    /// Interpolation error
    InterpolationError(String),
    
    /// Evaluation error
    EvaluationError(String),
    
    /// Proof verification failed
    VerificationFailed {
        round: usize,
        expected: String,
        actual: String,
    },
    
    /// Invalid proof format
    InvalidProofFormat(String),
    
    /// Field element conversion error
    FieldConversionError(String),
    
    /// Constraint evaluation error
    ConstraintEvaluationError(String),
}

impl fmt::Display for SumCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumCheckError::InvalidNumVariables { expected, actual } => {
                write!(f, "Invalid number of variables: expected {}, got {}", expected, actual)
            }
            SumCheckError::InvalidNumMles { expected, actual } => {
                write!(f, "Invalid number of MLEs: expected {}, got {}", expected, actual)
            }
            SumCheckError::VariableCountMismatch { expected, actual } => {
                write!(f, "Variable count mismatch: expected {}, got {}", expected, actual)
            }
            SumCheckError::InvalidDegree { max_supported, actual } => {
                write!(f, "Invalid degree: max supported {}, got {}", max_supported, actual)
            }
            SumCheckError::InterpolationError(msg) => {
                write!(f, "Interpolation error: {}", msg)
            }
            SumCheckError::EvaluationError(msg) => {
                write!(f, "Evaluation error: {}", msg)
            }
            SumCheckError::VerificationFailed { round, expected, actual } => {
                write!(f, "Verification failed at round {}: expected {}, got {}", round, expected, actual)
            }
            SumCheckError::InvalidProofFormat(msg) => {
                write!(f, "Invalid proof format: {}", msg)
            }
            SumCheckError::FieldConversionError(msg) => {
                write!(f, "Field conversion error: {}", msg)
            }
            SumCheckError::ConstraintEvaluationError(msg) => {
                write!(f, "Constraint evaluation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for SumCheckError {}

impl From<&'static str> for SumCheckError {
    fn from(msg: &'static str) -> Self {
        SumCheckError::EvaluationError(msg.to_string())
    }
}

impl From<String> for SumCheckError {
    fn from(msg: String) -> Self {
        SumCheckError::EvaluationError(msg)
    }
}