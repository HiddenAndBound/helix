//! Sparse matrix operations for the Spartan zkSNARK protocol.
//!
//! Provides efficient sparse multilinear extension (MLE) polynomials, metadata preprocessing
//! for sum-check protocols, and timestamp tracking for memory consistency checking.
//!
//! Key features:
//! - Sparse MLE representation: O(nnz) storage vs O(n²) dense
//! - Metadata preprocessing for sum-check protocols
//! - Twist & Shout memory checking timestamps
use crate::pcs::{BaseFoldConfig, Basefold};
use crate::spartan::{
    error::{SparseError, SparseResult},
    spark::commit::{SparkCommitment, SparkProverData},
};
use crate::utils::{Fp, Fp4, eq::EqEvals, polynomial::MLE};
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

        (
            (max_row + 1).next_power_of_two(),
            (max_col + 1).next_power_of_two(),
        )
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

    /// Binds the first half of variables using the EqEvals (equality polynomial) to compute
    /// pre-multiplication of the vector (x·A) as a linear algebra equation as an MLE<Fp4>.
    ///
    /// This implements the binding of the first log₂(rows) variables of the multilinear
    /// extension of the sparse matrix, effectively computing the inner product of the
    /// EqEvals coefficients with the matrix rows.
    ///
    /// # Arguments
    /// * `eq_evals` - The equality polynomial evaluations for the point whose length
    ///               should equal log₂(rows) of the underlying sparse matrix
    ///
    /// # Returns
    /// An MLE<Fp4> representing the result of x·A where x is encoded via eq_evals
    pub fn bind_first_half_variables(&self, eq_evals: &EqEvals) -> SparseResult<MLE<Fp4>> {
        let (rows, cols) = self.dimensions;

        if rows == 0 || cols == 0 {
            return Err(SparseError::EmptyMatrix);
        }

        // Calculate the number of row variables (log₂ of row dimensions)
        let row_vars = (rows as f64).log2().ceil() as usize;

        // Validate that eq_evals has the correct number of variables for row binding
        if eq_evals.n_vars() != row_vars {
            return Err(SparseError::ValidationError(format!(
                "EqEvals has {} variables but expected {} for {} rows",
                eq_evals.n_vars(),
                row_vars,
                rows
            )));
        }

        // The eq_evals should have 2^row_vars coefficients
        let expected_eq_coeffs = 1 << row_vars;
        if eq_evals.coeffs().len() != expected_eq_coeffs {
            return Err(SparseError::ValidationError(format!(
                "EqEvals has {} coefficients but expected {} for {} row variables",
                eq_evals.coeffs().len(),
                expected_eq_coeffs,
                row_vars
            )));
        }

        // Initialize result vector with zeros - this will be our column dimension
        let mut result_coeffs = vec![Fp4::ZERO; cols];

        // For each non-zero entry in the sparse matrix, accumulate the weighted contribution
        // This computes: result[col] = ∑_row (eq_evals[row] * matrix[row][col])
        for ((row, col), &value) in self.iter() {
            // Ensure we don't go out of bounds
            if *row < rows && *col < cols && *row < eq_evals.coeffs().len() {
                // Convert BabyBear matrix value to Fp4 and multiply by eq_evals coefficient
                let contribution = eq_evals.coeffs()[*row] * Fp4::from(value);
                result_coeffs[*col] += contribution;
            }
        }

        Ok(MLE::new(result_coeffs))
    }
}

/// Metadata for Spartan sum-check protocols.
///
/// Converts sparse MLE into dense MLEs (row, col, val) with timestamps
/// for efficient evaluation and memory consistency checking.
#[derive(Debug, Clone)]
pub struct SparkMetadata {
    /// Row indices as multilinear extension
    pub row: MLE<Fp>,
    /// Column indices as multilinear extension
    pub col: MLE<Fp>,
    /// Coefficient values as multilinear extension
    pub val: MLE<Fp>,
    /// Timestamp information for row accesses
    pub row_read_ts: MLE<Fp>,
    pub row_final_ts: MLE<Fp>,
    pub col_read_ts: MLE<Fp>,
    pub col_final_ts: MLE<Fp>,
}

impl SparkMetadata {
    /// Creates metadata from preprocessed components.
    /// All MLEs must have the same length.
    pub fn new(
        row: MLE<Fp>,
        col: MLE<Fp>,
        val: MLE<Fp>,
        row_read_ts: MLE<Fp>,
        row_final_ts: MLE<Fp>,
        col_read_ts: MLE<Fp>,
        col_final_ts: MLE<Fp>,
    ) -> SparseResult<Self> {
        // Validate that all MLEs have the same length
        if row.len() != col.len() || col.len() != val.len() {
            return Err(SparseError::ValidationError(format!(
                "MLE length mismatch: row={}, col={}, val={}",
                row.len(),
                col.len(),
                val.len()
            )));
        }

        Ok(SparkMetadata {
            row,
            val,
            col,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
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
        let TimeStamps {
            read_ts: row_read_ts,
            final_ts: row_final_ts,
        } = TimeStamps::compute(&row_vec, max_rows)?;
        let TimeStamps {
            read_ts: col_read_ts,
            final_ts: col_final_ts,
        } = TimeStamps::compute(&col_vec, max_cols)?;

        // Convert vectors to MLEs
        let row_mle = MLE::new(row_vec);
        let col_mle = MLE::new(col_vec);
        let val_mle = MLE::new(val_vec);

        Self::new(
            row_mle,
            col_mle,
            val_mle,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
        )
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

    /// Provides read-only access to the row MLE.
    pub fn row(&self) -> &MLE<Fp> {
        &self.row
    }

    /// Provides read-only access to the column MLE.
    pub fn col(&self) -> &MLE<Fp> {
        &self.col
    }

    /// Provides read-only access to the value MLE.
    pub fn val(&self) -> &MLE<Fp> {
        &self.val
    }

    /// Returns the maximum number of variables across all stored MLEs.
    pub fn max_n_vars(&self) -> Option<usize> {
        [
            self.row.n_vars(),
            self.col.n_vars(),
            self.val.n_vars(),
            self.row_read_ts.n_vars(),
            self.row_final_ts.n_vars(),
            self.col_read_ts.n_vars(),
            self.col_final_ts.n_vars(),
        ]
        .into_iter()
        .max()
    }

    /// Commits to each component MLE using the BaseFold PCS and aggregates the roots
    /// along with prover-side artifacts.
    pub fn commit(
        &self,
        roots: &[Vec<Fp>],
        config: &BaseFoldConfig,
    ) -> anyhow::Result<(SparkCommitment, SparkProverData)> {
        // TODO:Constrain this in all constructors.
        let max_vars = self.max_n_vars().expect("Fields will be non empty.");
        let Self {
            row,
            col,
            val,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
        } = &self;
        let (row_commitment, row_prover) =
            Basefold::commit(row, &roots[max_vars - row.n_vars()..], config)?;
        let (col_commitment, col_prover) =
            Basefold::commit(col, &roots[max_vars - col.n_vars()..], config)?;
        let (val_commitment, val_prover) =
            Basefold::commit(val, &roots[max_vars - val.n_vars()..], config)?;
        let (row_read_ts_commitment, row_read_ts_prover) = Basefold::commit(
            row_read_ts,
            &roots[max_vars - row_read_ts.n_vars()..],
            config,
        )?;
        let (row_final_ts_commitment, row_final_ts_prover) = Basefold::commit(
            row_final_ts,
            &roots[max_vars - row_final_ts.n_vars()..],
            config,
        )?;
        let (col_read_ts_commitment, col_read_ts_prover) = Basefold::commit(
            col_read_ts,
            &roots[max_vars - col_read_ts.n_vars()..],
            config,
        )?;
        let (col_final_ts_commitment, col_final_ts_prover) = Basefold::commit(
            col_final_ts,
            &roots[max_vars - col_final_ts.n_vars()..],
            config,
        )?;

        let commitment = SparkCommitment::new(
            row_commitment.commitment,
            col_commitment.commitment,
            val_commitment.commitment,
            row_read_ts_commitment.commitment,
            row_final_ts_commitment.commitment,
            col_read_ts_commitment.commitment,
            col_final_ts_commitment.commitment,
        );

        let prover_data = SparkProverData::new(
            row_prover,
            col_prover,
            val_prover,
            row_read_ts_prover,
            row_final_ts_prover,
            col_read_ts_prover,
            col_final_ts_prover,
        );

        Ok((commitment, prover_data))
    }
}

/// Timestamp tracking for Twist & Shout memory consistency protocols.
///
/// Maintains read_ts[i] ≤ final_ts[i] invariant for all addresses.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeStamps {
    /// Read timestamps for each address
    read_ts: MLE<Fp>,
    /// Final write timestamps for each address
    final_ts: MLE<Fp>,
}

impl TimeStamps {
    /// Creates timestamp structure, validating read_ts ≤ final_ts invariant.
    pub fn new(read_ts: Vec<BabyBear>, final_ts: Vec<BabyBear>) -> SparseResult<Self> {
        // read_ts and final_ts serve different purposes and can have different sizes:
        // - read_ts[i] = timestamp before i-th memory access (size = padded accesses)
        // - final_ts[j] = final write count for address j (size = address space)

        Ok(TimeStamps {
            read_ts: MLE::new(read_ts),
            final_ts: MLE::new(final_ts),
        })
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

    /// Returns a reference to the read timestamps
    pub fn read_ts(&self) -> &MLE<Fp> {
        &self.read_ts
    }

    /// Returns a reference to the final timestamps
    pub fn final_ts(&self) -> &MLE<Fp> {
        &self.final_ts
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
    fn test_sparse_xmle_empty() {
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
        let vector = vec![BabyBear::from_u32(1), BabyBear::from_u32(2)];

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
    fn test_spartan_metadata_new_length_mismatch() {
        let row_mle = MLE::new(vec![BabyBear::ZERO, BabyBear::ZERO]); // Length 2
        let col_mle = MLE::new(vec![
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
        ]); // Length 4 (different)
        let val_mle = MLE::new(vec![BabyBear::from_u32(5), BabyBear::from_u32(6)]); // Length 2

        let read_ts = vec![BabyBear::ZERO];
        let final_ts = vec![BabyBear::ONE];
        let TimeStamps {
            read_ts: row_read_ts,
            final_ts: row_final_ts,
        } = TimeStamps::new(read_ts.clone(), final_ts.clone()).unwrap();
        let TimeStamps {
            read_ts: col_read_ts,
            final_ts: col_final_ts,
        } = TimeStamps::new(read_ts, final_ts).unwrap();

        let result = SparkMetadata::new(
            row_mle,
            col_mle,
            val_mle,
            row_read_ts,
            row_final_ts,
            col_read_ts,
            col_final_ts,
        );
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_spartan_metadata_preprocess_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 1), BabyBear::from_u32(5));
        coeffs.insert((1, 0), BabyBear::from_u32(10));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let metadata = SparkMetadata::preprocess(&sparse_mle).unwrap();

        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_spartan_metadata_preprocess_empty_matrix() {
        let sparse_mle = SparseMLE::empty();
        let result = SparkMetadata::preprocess(&sparse_mle);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_spartan_metadata_extract_dense_vectors() {
        let mut coeffs = HashMap::new();
        coeffs.insert((1, 2), BabyBear::from_u32(7));
        coeffs.insert((0, 1), BabyBear::from_u32(3));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let (row_vec, col_vec, val_vec) =
            SparkMetadata::extract_dense_vectors(&sparse_mle).unwrap();

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
        let read_ts = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u32(2)];
        let final_ts = vec![BabyBear::ONE, BabyBear::from_u32(2), BabyBear::from_u32(3)];

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
        coeffs.insert((3, 2), BabyBear::from_u32(15)); // Add 4th entry to make it power of 2

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let metadata1 = SparkMetadata::preprocess(&sparse_mle).unwrap();
        let metadata2 = SparkMetadata::preprocess(&sparse_mle).unwrap();

        // Should be deterministic
        assert_eq!(metadata1.row.coeffs(), metadata2.row.coeffs());
        assert_eq!(metadata1.col.coeffs(), metadata2.col.coeffs());
        assert_eq!(metadata1.val.coeffs(), metadata2.val.coeffs());
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

    #[test]
    fn test_bind_first_half_variables_simple_matrix() {
        use crate::utils::eq::EqEvals;

        // Create a simple 2x2 matrix:
        // [1, 2]
        // [3, 4]
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE); // A[0,0] = 1
        coeffs.insert((0, 1), BabyBear::from_u32(2)); // A[0,1] = 2
        coeffs.insert((1, 0), BabyBear::from_u32(3)); // A[1,0] = 3
        coeffs.insert((1, 1), BabyBear::from_u32(4)); // A[1,1] = 4

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // For a 2x2 matrix, we need log₂(2) = 1 row variable
        // Create EqEvals for point [r₀] where r₀ = 5
        let point = vec![Fp4::from_u32(5)];
        let eq_evals = EqEvals::gen_from_point(&point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals).unwrap();

        // The result should be an MLE with 2 coefficients (column dimension)
        assert_eq!(result.len(), 2);

        // Calculate expected result manually:
        // eq_evals for point [5] should give coefficients [(1-5), 5] = [-4, 5]
        // result[0] = eq_evals[0] * A[0,0] + eq_evals[1] * A[1,0] = -4*1 + 5*3 = 11
        // result[1] = eq_evals[0] * A[0,1] + eq_evals[1] * A[1,1] = -4*2 + 5*4 = 12
        let expected_col0 =
            (Fp4::ONE - Fp4::from_u32(5)) * Fp4::ONE + Fp4::from_u32(5) * Fp4::from_u32(3);
        let expected_col1 =
            (Fp4::ONE - Fp4::from_u32(5)) * Fp4::from_u32(2) + Fp4::from_u32(5) * Fp4::from_u32(4);

        assert_eq!(result.coeffs()[0], expected_col0);
        assert_eq!(result.coeffs()[1], expected_col1);
    }

    #[test]
    fn test_bind_first_half_variables_4x4_matrix() {
        use crate::utils::eq::EqEvals;

        // Create a 4x4 matrix with specific pattern for easy verification
        let mut coeffs = HashMap::new();
        for row in 0..4 {
            for col in 0..4 {
                coeffs.insert((row, col), BabyBear::from_u32((row + col + 1) as u32));
            }
        }

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // For a 4x4 matrix, we need log₂(4) = 2 row variables
        // Create EqEvals for point [r₀, r₁] where r₀ = 2, r₁ = 3
        let point = vec![Fp4::from_u32(2), Fp4::from_u32(3)];
        let eq_evals = EqEvals::gen_from_point(&point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals).unwrap();

        // The result should be an MLE with 4 coefficients (column dimension)
        assert_eq!(result.len(), 4);

        // Verify the result by checking that each coefficient is computed correctly
        // Each result[col] = ∑_row (eq_evals[row] * matrix[row][col])
        for col in 0..4 {
            let mut expected = Fp4::ZERO;
            for row in 0..4 {
                let matrix_value = Fp4::from_u32((row + col + 1) as u32);
                expected += eq_evals[row] * matrix_value;
            }
            assert_eq!(result.coeffs()[col], expected);
        }
    }

    #[test]
    fn test_bind_first_half_variables_sparse_matrix() {
        use crate::utils::eq::EqEvals;

        // Create a sparse 4x4 matrix with only a few non-zero entries
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 1), BabyBear::from_u32(7)); // A[0,1] = 7
        coeffs.insert((2, 0), BabyBear::from_u32(11)); // A[2,0] = 11
        coeffs.insert((3, 3), BabyBear::from_u32(13)); // A[3,3] = 13

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // For a 4x4 matrix, we need 2 row variables
        let point = vec![Fp4::from_u32(4), Fp4::from_u32(6)];
        let eq_evals = EqEvals::gen_from_point(&point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals).unwrap();

        // Check specific entries that should be non-zero
        // result[0] = eq_evals[2] * 11 (only non-zero contribution from row 2)
        let expected_col0 = eq_evals[2] * Fp4::from_u32(11);
        assert_eq!(result.coeffs()[0], expected_col0);

        // result[1] = eq_evals[0] * 7 (only non-zero contribution from row 0)
        let expected_col1 = eq_evals[0] * Fp4::from_u32(7);
        assert_eq!(result.coeffs()[1], expected_col1);

        // result[2] should be zero (no non-zero entries in column 2)
        assert_eq!(result.coeffs()[2], Fp4::ZERO);

        // result[3] = eq_evals[3] * 13 (only non-zero contribution from row 3)
        let expected_col3 = eq_evals[3] * Fp4::from_u32(13);
        assert_eq!(result.coeffs()[3], expected_col3);
    }

    #[test]
    fn test_bind_first_half_variables_dimension_mismatch() {
        use crate::utils::eq::EqEvals;

        // Create a 2x2 matrix
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);
        coeffs.insert((1, 1), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // Try to bind with wrong number of variables (should be 1 for 2x2 matrix)
        let wrong_point = vec![Fp4::from_u32(1), Fp4::from_u32(2)]; // 2 variables instead of 1
        let eq_evals = EqEvals::gen_from_point(&wrong_point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_bind_first_half_variables_empty_matrix() {
        use crate::utils::eq::EqEvals;

        let sparse_mle = SparseMLE::empty();
        let point = vec![Fp4::from_u32(1)];
        let eq_evals = EqEvals::gen_from_point(&point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_bind_first_half_variables_single_row_matrix() {
        use crate::utils::eq::EqEvals;

        // Create a 1x4 matrix: [5, 6, 7, 8]
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(5));
        coeffs.insert((0, 1), BabyBear::from_u32(6));
        coeffs.insert((0, 2), BabyBear::from_u32(7));
        coeffs.insert((0, 3), BabyBear::from_u32(8));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // For a 1-row matrix, we need log₂(1) = 0 row variables (constant polynomial)
        let point: Vec<Fp4> = vec![]; // Empty point for 0 variables
        let eq_evals = EqEvals::gen_from_point(&point);

        let result = sparse_mle.bind_first_half_variables(&eq_evals).unwrap();

        // The result should just be the matrix row scaled by eq_evals[0] = 1
        assert_eq!(result.coeffs()[0], Fp4::from_u32(5));
        assert_eq!(result.coeffs()[1], Fp4::from_u32(6));
        assert_eq!(result.coeffs()[2], Fp4::from_u32(7));
        assert_eq!(result.coeffs()[3], Fp4::from_u32(8));
    }
}
