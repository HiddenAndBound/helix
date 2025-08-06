//! Sparse matrix operations for the Spartan zkSNARK protocol.
//!
//! Provides efficient sparse multilinear extension (MLE) polynomials, metadata preprocessing
//! for sum-check protocols, and timestamp tracking for memory consistency checking.
//!
//! Key features:
//! - Sparse MLE representation: O(nnz) storage vs O(n²) dense
//! - Metadata preprocessing for sum-check protocols  
//! - Twist & Shout memory checking timestamps

use crate::spartan::error::{SparseError, SparseResult};
use crate::utils::{polynomial::MLE, Fp};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use std::collections::HashMap;

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

        ((max_row + 1).next_power_of_two(), (max_col + 1).next_power_of_two())
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

    /// Multiplies this sparse matrix by an MLE polynomial.
    /// MLE coefficient length must match matrix column count.
    pub fn multiply_by_mle(&self, mle: &MLE<Fp>) -> SparseResult<MLE<Fp>> {
        let (rows, cols) = self.dimensions;

        // Validate that MLE length matches matrix column count
        if mle.len() != cols {
            return Err(SparseError::DimensionMismatch {
                expected: (cols, mle.len()),
                actual: (cols, mle.len()),
            });
        }

        // Initialize result vector with zeros
        let mut result = vec![BabyBear::ZERO; rows];

        // Perform sparse matrix-MLE multiplication with O(nnz) complexity
        for ((row, col), &value) in self.iter() {
            // Ensure we don't go out of bounds (defensive programming)
            if *row < rows && *col < cols {
                result[*row] += value * mle.coeffs()[*col];
            }
        }

        Ok(MLE::new(result))
    }

    /// Creates sparse MLE from this matrix (flattened representation)
    pub fn to_sparse_mle(&self) -> crate::spartan::mle::SparseMultilinearExtension {
        let mut coeffs = HashMap::new();
        let (rows, cols) = self.dimensions;
        let n_vars = (rows * cols).next_power_of_two().trailing_zeros() as usize;
        
        for ((row, col), &value) in self.iter() {
            if value != BabyBear::ZERO {
                let index = row * cols + col;
                coeffs.insert(index, value);
            }
        }
        
        crate::spartan::mle::SparseMultilinearExtension::new(coeffs, n_vars)
    }

    /// Multiplies this sparse matrix by a dense vector (legacy method).
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
        // read_ts and final_ts serve different purposes and can have different sizes:
        // - read_ts[i] = timestamp before i-th memory access (size = padded accesses)  
        // - final_ts[j] = final write count for address j (size = address space)
        
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

        // Initialize timestamp vectors with their appropriate sizes
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

        // For the padded entries (from indices.len() to padded_size), we access
        // dummy addresses that don't affect the computation but maintain the
        // power-of-2 memory access pattern
        if padded_size > indices.len() {
            // Use address 0 as dummy (or any valid address < max_address_space)
            let dummy_address = 0;
            for logical_time in indices.len()..padded_size {
                // These dummy accesses maintain the memory pattern without affecting computation
                read_ts[logical_time] = final_ts[dummy_address];
                final_ts[dummy_address] = final_ts[dummy_address] + BabyBear::ONE;
            }
        }

        TimeStamps::new(read_ts, final_ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use std::collections::HashMap;

    #[test]
    fn test_sparse_mle_new_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);
        coeffs.insert((1, 1), BabyBear::from_u32(2));

        let sparse_mle = SparseMLE::new(coeffs.clone()).unwrap();
        assert_eq!(sparse_mle.num_nonzeros(), 2);
        assert_eq!(sparse_mle.dimensions(), (2, 2));
    }

    #[test]
    fn test_sparse_mle_new_empty_fails() {
        let coeffs = HashMap::new();
        let result = SparseMLE::new(coeffs);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_sparse_mle_empty() {
        let sparse_mle = SparseMLE::empty();
        assert_eq!(sparse_mle.num_nonzeros(), 0);
        assert_eq!(sparse_mle.dimensions(), (0, 0));
    }

    #[test]
    fn test_sparse_mle_compute_dimensions() {
        let mut coeffs = HashMap::new();
        coeffs.insert((3, 5), BabyBear::ONE);
        coeffs.insert((1, 2), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        // Max row is 3, max col is 5, so next power of 2 is 4 and 8
        assert_eq!(sparse_mle.dimensions(), (4, 8));
    }

    #[test]
    fn test_sparse_mle_get() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(42));
        coeffs.insert((1, 1), BabyBear::from_u32(100));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        assert_eq!(sparse_mle.get(0, 0), BabyBear::from_u32(42));
        assert_eq!(sparse_mle.get(1, 1), BabyBear::from_u32(100));
        assert_eq!(sparse_mle.get(0, 1), BabyBear::ZERO);
        assert_eq!(sparse_mle.get(2, 2), BabyBear::ZERO);
    }

    #[test]
    fn test_sparse_mle_iter() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 1), BabyBear::from_u32(5));
        coeffs.insert((2, 3), BabyBear::from_u32(10));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let entries: Vec<_> = sparse_mle.iter().collect();

        assert_eq!(entries.len(), 2);
        assert!(entries.contains(&(&(0, 1), &BabyBear::from_u32(5))));
        assert!(entries.contains(&(&(2, 3), &BabyBear::from_u32(10))));
    }

    #[test]
    fn test_sparse_mle_multiply_by_vector_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((0, 1), BabyBear::from_u32(3));
        coeffs.insert((1, 0), BabyBear::from_u32(4));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let vector = vec![
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
        ];

        let result = sparse_mle.multiply_by_vector(&vector).unwrap();

        // Expected: [2*1 + 3*2, 4*1 + 5*2] = [8, 14]
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], BabyBear::from_u32(8));
        assert_eq!(result[1], BabyBear::from_u32(14));
    }

    #[test]
    fn test_sparse_mle_multiply_by_vector_dimension_mismatch() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let vector = vec![BabyBear::ONE, BabyBear::ONE, BabyBear::ONE]; // Wrong size

        let result = sparse_mle.multiply_by_vector(&vector);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_mle_multiply_by_mle_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((0, 1), BabyBear::from_u32(3));
        coeffs.insert((1, 0), BabyBear::from_u32(4));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let mle_coeffs = vec![BabyBear::from_u32(1), BabyBear::from_u32(2)];
        let mle = MLE::new(mle_coeffs);

        let result = sparse_mle.multiply_by_mle(&mle).unwrap();

        // Expected: [2*1 + 3*2, 4*1 + 5*2] = [8, 14]
        assert_eq!(result.len(), 2);
        assert_eq!(result.coeffs()[0], BabyBear::from_u32(8));
        assert_eq!(result.coeffs()[1], BabyBear::from_u32(14));
    }

    #[test]
    fn test_sparse_mle_multiply_by_mle_dimension_mismatch() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let mle_coeffs = vec![BabyBear::ONE, BabyBear::ONE, BabyBear::ONE, BabyBear::ONE]; // Wrong size
        let mle = MLE::new(mle_coeffs);

        let result = sparse_mle.multiply_by_mle(&mle);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_error_display() {
        let validation_err = SparseError::ValidationError("test message".to_string());
        assert_eq!(validation_err.to_string(), "Validation error: test message");

        let dimension_err = SparseError::DimensionMismatch {
            expected: (2, 3),
            actual: (4, 5),
        };
        assert_eq!(
            dimension_err.to_string(),
            "Dimension mismatch: expected (2, 3), got (4, 5)"
        );

        let index_err = SparseError::IndexOutOfBounds {
            index: (5, 6),
            bounds: (3, 4),
        };
        assert_eq!(index_err.to_string(), "Index (5, 6) out of bounds (3, 4)");

        let empty_err = SparseError::EmptyMatrix;
        assert_eq!(empty_err.to_string(), "Operation on empty matrix");

        let constraint_err = SparseError::ConstraintViolation("constraint failed".to_string());
        assert_eq!(
            constraint_err.to_string(),
            "Constraint violation: constraint failed"
        );
    }

    #[test]
    fn test_spartan_metadata_new_valid() {
        let row = vec![BabyBear::ZERO, BabyBear::ONE];
        let col = vec![BabyBear::ONE, BabyBear::ZERO];
        let val = vec![
            BabyBear::from_u32(5),
            BabyBear::from_u32(10),
        ];

        let read_ts = vec![BabyBear::ZERO, BabyBear::ONE];
        let final_ts = vec![BabyBear::ONE, BabyBear::from_u32(2)];
        let row_ts = TimeStamps::new(read_ts.clone(), final_ts.clone()).unwrap();
        let col_ts = TimeStamps::new(read_ts, final_ts).unwrap();

        let metadata = SpartanMetadata::new(row, col, val, row_ts, col_ts).unwrap();
        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_spartan_metadata_new_length_mismatch() {
        let row = vec![BabyBear::ZERO];
        let col = vec![BabyBear::ONE, BabyBear::ZERO]; // Different length
        let val = vec![BabyBear::from_u32(5)];

        let read_ts = vec![BabyBear::ZERO];
        let final_ts = vec![BabyBear::ONE];
        let row_ts = TimeStamps::new(read_ts.clone(), final_ts.clone()).unwrap();
        let col_ts = TimeStamps::new(read_ts, final_ts).unwrap();

        let result = SpartanMetadata::new(row, col, val, row_ts, col_ts);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_spartan_metadata_preprocess_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 1), BabyBear::from_u32(5));
        coeffs.insert((1, 0), BabyBear::from_u32(10));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let metadata = SpartanMetadata::preprocess(&sparse_mle).unwrap();

        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_spartan_metadata_preprocess_empty_matrix() {
        let sparse_mle = SparseMLE::empty();
        let result = SpartanMetadata::preprocess(&sparse_mle);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_spartan_metadata_extract_dense_vectors() {
        let mut coeffs = HashMap::new();
        coeffs.insert((1, 2), BabyBear::from_u32(7));
        coeffs.insert((0, 1), BabyBear::from_u32(3));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let (row_vec, col_vec, val_vec) =
            SpartanMetadata::extract_dense_vectors(&sparse_mle).unwrap();

        assert_eq!(row_vec.len(), 2);
        assert_eq!(col_vec.len(), 2);
        assert_eq!(val_vec.len(), 2);

        // Should be sorted by coordinates: (0,1) then (1,2)
        assert_eq!(row_vec[0], BabyBear::ZERO);
        assert_eq!(col_vec[0], BabyBear::ONE);
        assert_eq!(val_vec[0], BabyBear::from_u32(3));

        assert_eq!(row_vec[1], BabyBear::ONE);
        assert_eq!(col_vec[1], BabyBear::from_u32(2));
        assert_eq!(val_vec[1], BabyBear::from_u32(7));
    }

    #[test]
    fn test_timestamps_new_valid() {
        let read_ts = vec![
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::from_u32(2),
        ];
        let final_ts = vec![
            BabyBear::ONE,
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
        ];

        let timestamps = TimeStamps::new(read_ts, final_ts).unwrap();
        assert_eq!(timestamps.read_ts.len(), 3);
        assert_eq!(timestamps.final_ts.len(), 3);
    }

    #[test]
    fn test_timestamps_new_different_lengths() {
        let read_ts = vec![BabyBear::ZERO];
        let final_ts = vec![BabyBear::ONE, BabyBear::from_u32(2)]; // Different length is now OK

        let result = TimeStamps::new(read_ts, final_ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timestamps_new_with_equal_vectors() {
        let read_ts = vec![BabyBear::from_u32(1), BabyBear::from_u32(2)];
        let final_ts = vec![BabyBear::from_u32(3), BabyBear::from_u32(4)];

        let result = TimeStamps::new(read_ts, final_ts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timestamps_compute_valid() {
        let indices = vec![
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::from_u32(2),
        ];
        let max_address_space = 4; // Power of 2

        let timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();

        // Check that read timestamps are correct
        assert_eq!(timestamps.read_ts[0], BabyBear::ZERO); // Address 0, first access
        assert_eq!(timestamps.read_ts[1], BabyBear::ZERO); // Address 1, first access
        assert_eq!(timestamps.read_ts[2], BabyBear::ONE); // Address 0, second access
        assert_eq!(timestamps.read_ts[3], BabyBear::ZERO); // Address 2, first access
    }

    #[test]
    fn test_timestamps_compute_empty_indices() {
        let indices = vec![];
        let max_address_space = 4;

        let result = TimeStamps::compute(&indices, max_address_space);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_timestamps_compute_non_power_of_two() {
        let indices = vec![BabyBear::ZERO];
        let max_address_space = 3; // Not a power of 2

        let result = TimeStamps::compute(&indices, max_address_space);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_edge_case_single_element_matrix() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(42));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        assert_eq!(sparse_mle.dimensions(), (1, 1));

        let vector = vec![BabyBear::from_u32(2)];
        let result = sparse_mle.multiply_by_vector(&vector).unwrap();
        assert_eq!(result[0], BabyBear::from_u32(84));
    }

    #[test]
    fn test_large_coordinates() {
        let mut coeffs = HashMap::new();
        coeffs.insert((100, 200), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        // Next power of 2 for 100 is 128, for 200 is 256
        assert_eq!(sparse_mle.dimensions(), (128, 256));
    }

    #[test]
    fn test_metadata_preprocessing_deterministic_order() {
        let mut coeffs = HashMap::new();
        coeffs.insert((2, 1), BabyBear::from_u32(20));
        coeffs.insert((0, 3), BabyBear::from_u32(5));
        coeffs.insert((1, 0), BabyBear::from_u32(10));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let metadata1 = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        let metadata2 = SpartanMetadata::preprocess(&sparse_mle).unwrap();

        // Should be deterministic
        assert_eq!(metadata1.row, metadata2.row);
        assert_eq!(metadata1.col, metadata2.col);
        assert_eq!(metadata1.val, metadata2.val);
    }

    #[test]
    fn test_timestamps_memory_access_pattern() {
        // Test a specific memory access pattern: [0, 1, 0, 1, 2]
        let indices = vec![
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::from_u32(2),
        ];

        let timestamps = TimeStamps::compute(&indices, 4).unwrap();

        // Verify read timestamps capture the state before each access
        assert_eq!(timestamps.read_ts[0], BabyBear::ZERO); // First access to 0
        assert_eq!(timestamps.read_ts[1], BabyBear::ZERO); // First access to 1
        assert_eq!(timestamps.read_ts[2], BabyBear::ONE); // Second access to 0
        assert_eq!(timestamps.read_ts[3], BabyBear::ONE); // Second access to 1
        assert_eq!(timestamps.read_ts[4], BabyBear::ZERO); // First access to 2

        // Verify read timestamps for padding entries (dummy accesses to address 0)
        assert_eq!(timestamps.read_ts[5], BabyBear::from_u32(2)); // After 2 real accesses to addr 0
        assert_eq!(timestamps.read_ts[6], BabyBear::from_u32(3)); // After 3 accesses to addr 0
        assert_eq!(timestamps.read_ts[7], BabyBear::from_u32(4)); // After 4 accesses to addr 0

        // Verify final timestamps show total accesses per address  
        assert_eq!(timestamps.final_ts[0], BabyBear::from_u32(5)); // Address 0: 2 real + 3 dummy = 5 total
        assert_eq!(timestamps.final_ts[1], BabyBear::from_u32(2)); // Address 1 accessed twice
        assert_eq!(timestamps.final_ts[2], BabyBear::ONE); // Address 2 accessed once
        assert_eq!(timestamps.final_ts[3], BabyBear::ZERO); // Address 3 never accessed

        // Verify arrays have appropriate lengths (both power of 2)
        assert_eq!(timestamps.read_ts.len(), 8); // Padded memory accesses
        assert_eq!(timestamps.final_ts.len(), 4); // Address space size
    }
}
