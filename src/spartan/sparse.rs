//! Sparse matrix operations for the Spartan zkSNARK protocol.
//!
//! Provides efficient sparse multilinear extension (MLE) polynomials, metadata preprocessing
//! for sum-check protocols, and timestamp tracking for memory consistency checking.
//!
//! Key features:
//! - Sparse MLE representation: O(nnz) storage vs O(n²) dense
//! - Metadata preprocessing for sum-check protocols  
//! - Twist & Shout memory checking timestamps

use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use std::collections::HashMap;
use std::fmt;

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

/// Sparse multilinear extension (MLE) polynomial for Spartan.
///
/// Stores only non-zero coefficients with their (row, col) coordinates,
/// reducing space complexity from O(2^m) to O(nnz).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMLE {
    /// Sparse coefficient storage: (row, col) -> field_element
    coeffs: HashMap<(usize, usize), BabyBear>,
    /// Cached matrix dimensions for validation
    dimensions: (usize, usize),
}

impl SparseMLE {
    /// Creates a new sparse MLE from a coefficient map.
    /// Returns `EmptyMatrix` error if coeffs is empty.
    pub fn new(coeffs: HashMap<(usize, usize), BabyBear>) -> SparseResult<Self> {
        if coeffs.is_empty() {
            return Err(SparseError::EmptyMatrix);
        }

        let dimensions = Self::compute_dimensions(&coeffs);

        Ok(SparseMLE {
            coeffs,
            dimensions: dimensions,
        })
    }

    /// Creates an empty sparse MLE.
    pub fn empty() -> Self {
        SparseMLE {
            coeffs: HashMap::new(),
            dimensions: (0, 0),
        }
    }

    /// Computes matrix dimensions as (max_row + 1, max_col + 1).
    fn compute_dimensions(coeffs: &HashMap<(usize, usize), BabyBear>) -> (usize, usize) {
        if coeffs.is_empty() {
            return (0, 0);
        }

        let max_row = coeffs
            .keys()
            .map(|(row, _)| *row)
            .max()
            .expect("Should be non-zero");

        let max_col = coeffs
            .keys()
            .map(|(_, col)| *col)
            .max()
            .expect("Should be non-zero");

        (max_row.next_power_of_two(), max_col.next_power_of_two())
    }

    /// Returns the number of non-zero entries.
    pub fn num_nonzeros(&self) -> usize {
        self.coeffs.len()
    }

    /// Returns the matrix dimensions as (rows, cols).
    pub fn dimensions(&self) -> (usize, usize) {
        self.dimensions
    }

    /// Gets the coefficient at (row, col), returns zero if not present.
    pub fn get(&self, row: usize, col: usize) -> BabyBear {
        self.coeffs
            .get(&(row, col))
            .copied()
            .unwrap_or(BabyBear::ZERO)
    }

    /// Returns an iterator over non-zero entries as ((row, col), value) tuples.
    pub fn iter(&self) -> impl Iterator<Item = (&(usize, usize), &BabyBear)> {
        self.coeffs.iter()
    }

    /// Multiplies this sparse matrix by a dense vector.
    /// Vector length must match matrix column count.
    pub fn multiply_by_vector(&self, vector: &[BabyBear]) -> SparseResult<Vec<BabyBear>> {
        let (rows, cols) = self.dimensions;

        // Validate that vector length matches matrix column count
        if vector.len() != cols {
            return Err(SparseError::DimensionMismatch {
                expected: (cols, vector.len()),
                actual: (cols, vector.len()),
            });
        }

        // Initialize result vector with zeros
        let mut result = vec![BabyBear::ZERO; rows];

        // Perform sparse matrix-vector multiplication
        for ((row, col), &value) in self.iter() {
            // Ensure we don't go out of bounds (defensive programming)
            if *row < rows && *col < cols {
                result[*row] += value * vector[*col];
            }
        }

        Ok(result)
    }
}

/// Metadata for Spartan sum-check protocols.
///
/// Converts sparse MLE into dense vectors (row, col, val) with timestamps
/// for efficient evaluation and memory consistency checking.
#[derive(Debug, Clone, PartialEq)]
pub struct SpartanMetadata {
    /// Row indices converted to field elements
    row: Vec<BabyBear>,
    /// Column indices converted to field elements  
    col: Vec<BabyBear>,
    /// Coefficient values as field elements
    val: Vec<BabyBear>,
    /// Timestamp information for row accesses
    row_ts: TimeStamps,
    /// Timestamp information for column accesses
    col_ts: TimeStamps,
}

impl SpartanMetadata {
    /// Creates metadata from preprocessed components.
    /// All vectors must have the same length.
    pub fn new(
        row: Vec<BabyBear>,
        col: Vec<BabyBear>,
        val: Vec<BabyBear>,
        row_ts: TimeStamps,
        col_ts: TimeStamps,
    ) -> SparseResult<Self> {
        // Validate that all vectors have the same length
        if row.len() != col.len() || col.len() != val.len() {
            return Err(SparseError::ValidationError(format!(
                "Vector length mismatch: row={}, col={}, val={}",
                row.len(),
                col.len(),
                val.len()
            )));
        }

        Ok(SpartanMetadata {
            row,
            col,
            val,
            row_ts,
            col_ts,
        })
    }

    /// Preprocesses sparse MLE into metadata format for sum-check protocols.
    /// Time: O(nnz + log(max_dim)), Space: O(nnz + max_dim).
    pub fn preprocess(sparse_mle: &SparseMLE) -> SparseResult<Self> {
        if sparse_mle.num_nonzeros() == 0 {
            return Err(SparseError::EmptyMatrix);
        }

        let (max_rows, max_cols) = sparse_mle.dimensions;

        // Validate dimensions are reasonable
        if max_rows == 0 || max_cols == 0 {
            return Err(SparseError::ValidationError(
                "Matrix dimensions cannot be zero".to_string(),
            ));
        }

        // Convert sparse representation to dense vectors
        let (row_vec, col_vec, val_vec) = Self::extract_dense_vectors(sparse_mle)?;

        // Compute timestamp information for memory consistency checking
        let row_ts = TimeStamps::compute(&row_vec, max_rows)?;
        let col_ts = TimeStamps::compute(&col_vec, max_cols)?;

        Self::new(row_vec, col_vec, val_vec, row_ts, col_ts)
    }

    /// Extracts dense vectors from sparse representation in deterministic order.
    fn extract_dense_vectors(
        sparse_mle: &SparseMLE,
    ) -> SparseResult<(Vec<BabyBear>, Vec<BabyBear>, Vec<BabyBear>)> {
        let num_entries = sparse_mle.num_nonzeros();

        let mut row_vec = Vec::with_capacity(num_entries);
        let mut col_vec = Vec::with_capacity(num_entries);
        let mut val_vec = Vec::with_capacity(num_entries);

        // Process entries in a deterministic order for reproducibility
        let mut entries: Vec<_> = sparse_mle.iter().collect();
        entries.sort_by_key(|(coord, _)| *coord);

        for ((row_idx, col_idx), entry_val) in entries {
            row_vec.push(BabyBear::from_usize(*row_idx));
            col_vec.push(BabyBear::from_usize(*col_idx));
            val_vec.push(*entry_val);
        }

        Ok((row_vec, col_vec, val_vec))
    }

    /// Returns the number of non-zero entries.
    pub fn len(&self) -> usize {
        self.row.len()
    }
}

/// Timestamp tracking for Twist & Shout memory consistency protocols.
///
/// Maintains read_ts[i] ≤ final_ts[i] invariant for all addresses.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeStamps {
    /// Read timestamps for each address
    read_ts: Vec<BabyBear>,
    /// Final write timestamps for each address
    final_ts: Vec<BabyBear>,
}

impl TimeStamps {
    /// Creates timestamp structure, validating read_ts ≤ final_ts invariant.
    pub fn new(read_ts: Vec<BabyBear>, final_ts: Vec<BabyBear>) -> SparseResult<Self> {
        if read_ts.len() != final_ts.len() {
            return Err(SparseError::DimensionMismatch {
                expected: (read_ts.len(), read_ts.len()),
                actual: (read_ts.len(), final_ts.len()),
            });
        }

        // Validate timestamp consistency: read_ts ≤ final_ts
        for (i, (&read_time, &final_time)) in read_ts.iter().zip(&final_ts).enumerate() {
            if read_time.as_canonical_u32() > final_time.as_canonical_u32() {
                return Err(SparseError::ConstraintViolation(format!(
                    "Read timestamp {} > final timestamp {} at address {}",
                    read_time.as_canonical_u32(),
                    final_time.as_canonical_u32(),
                    i
                )));
            }
        }

        Ok(TimeStamps { read_ts, final_ts })
    }

    /// Computes timestamps from memory access sequence.
    /// Address space must be power of 2. Time: O(n+m), Space: O(m).
    pub fn compute(indices: &[BabyBear], max_address_space: usize) -> SparseResult<Self> {
        if indices.is_empty() {
            return Err(SparseError::ValidationError(
                "Cannot compute timestamps for empty index sequence".to_string(),
            ));
        }

        if !max_address_space.is_power_of_two() {
            return Err(SparseError::ValidationError(
                "Address space size must be a power of 2".to_string(),
            ));
        }

        let padded_size = indices.len().next_power_of_two();
        let timestamp_size = max_address_space.next_power_of_two();

        // Initialize timestamp vectors
        let mut read_ts = vec![BabyBear::ZERO; padded_size];
        let mut final_ts = vec![BabyBear::ZERO; timestamp_size];

        // Process each memory access in sequence
        for (logical_time, &addr_field) in indices.iter().enumerate() {
            let address = addr_field.as_canonical_u32() as usize;

            // Record read timestamp (current state before this access)
            read_ts[logical_time] = final_ts[address];

            // Update write timestamp (increment for this address)
            final_ts[address] = final_ts[address] + BabyBear::ONE;
        }

        TimeStamps::new(read_ts, final_ts)
    }
}
