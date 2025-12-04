use std::fmt;

/// Errors that can occur during sumcheck operations
#[derive(Debug, Clone, PartialEq)]
pub enum SumCheckError {
    /// Input validation failed
    ValidationError(String),
    /// Mathematical constraint violation
    ConstraintFail(String),
}

impl fmt::Display for SumCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumCheckError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SumCheckError::ConstraintFail(msg) => write!(f, "Constraint violation: {}", msg),
        }
    }
}

impl std::error::Error for SumCheckError {}

/// Result type for sumcheck operations
pub type SumCheckResult<T> = Result<T, SumCheckError>;

/// Errors that can occur during sparse matrix operations
#[derive(Debug, Clone, PartialEq)]
pub enum SparseError {
    /// Input validation failed
    ValidationError(String),
    /// Matrix dimensions are incompatible
    DimensionMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// Index out of bounds
    IndexOutOfBounds {
        index: (usize, usize),
        bounds: (usize, usize),
    },
    /// Empty matrix operation attempted
    EmptyMatrix,
    /// Mathematical constraint violation
    ConstraintViolation(String),
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SparseError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            SparseError::IndexOutOfBounds { index, bounds } => {
                write!(f, "Index {:?} out of bounds {:?}", index, bounds)
            }
            SparseError::EmptyMatrix => write!(f, "Operation on empty matrix"),
            SparseError::ConstraintViolation(msg) => write!(f, "Constraint violation: {}", msg),
        }
    }
}

impl std::error::Error for SparseError {}

/// Result type for sparse matrix operations
pub type SparseResult<T> = Result<T, SparseError>;
