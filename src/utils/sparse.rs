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
use crate::utils::{Fp, Fp4, eq::EqEvals, polynomial::MLE};
use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
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
        let result = self.multiply_by_vector(mle.coeffs())?;
        Ok(MLE::new(result))
    }

    /// Multiplies this sparse matrix by a dense vector (legacy method).
    /// Vector length must match matrix column count.
    pub fn multiply_by_vector(&self, vector: &[BabyBear]) -> SparseResult<Vec<BabyBear>> {
        let matrix_result = self.multiply_by_matrix(vector)?;
        Ok(matrix_result.coeffs().to_vec())
    }

    /// Multiplies this sparse matrix (n × k) by a collection of dense column vectors (k × m).
    /// Returns the dense result as an MLE whose coefficients are stored column-major,
    /// mirroring the input ordering. The input `matrix` slice should contain the columns
    /// concatenated in column-major order.
    ///
    /// The number of columns must be a power of two so the flattened output respects
    /// Spartan's hypercube layout requirements.
    pub fn multiply_by_matrix<F: ExtensionField<Fp> + Field>(
        &self,
        matrix: &[F],
    ) -> SparseResult<MLE<F>> {
        let (rows, cols) = self.dimensions;

        if rows == 0 || cols == 0 {
            return Err(SparseError::EmptyMatrix);
        }

        if matrix.is_empty() {
            return Ok(MLE::new(vec![F::ZERO; rows]));
        }

        if matrix.len() % cols != 0 {
            return Err(SparseError::DimensionMismatch {
                expected: (cols, matrix.len()),
                actual: (cols, matrix.len()),
            });
        }

        let num_columns = matrix.len() / cols;

        if !num_columns.is_power_of_two() {
            return Err(SparseError::ValidationError(
                "Number of column vectors must be a power of two".to_string(),
            ));
        }

        let mut flattened = vec![F::ZERO; rows * num_columns];

        for ((row, col), &value) in self.iter() {
            if *row < rows && *col < cols {
                for idx in 0..num_columns {
                    let input_offset = idx * cols + *col;
                    let output_offset = idx * rows + *row;
                    flattened[output_offset] += matrix[input_offset] * value;
                }
            }
        }

        Ok(MLE::new(flattened))
    }

    /// Computes the product `matrix · A^T` where `A` is the sparse matrix represented by
    /// `self` and `matrix` is a dense `(m × k)` matrix provided in column-major order.
    /// The result is returned column-major with dimensions `(m × n)` (where `n` is the
    /// number of rows in `A`).
    ///
    /// The input slice is assumed to already store the dense matrix transpose (so each
    /// column corresponds to a column of `A`).
    pub fn transpose_multiply_by_matrix<F: Field + ExtensionField<Fp>>(
        &self,
        matrix: &[F],
    ) -> SparseResult<MLE<F>> {
        let (rows, cols) = self.dimensions;

        if rows == 0 || cols == 0 {
            return Err(SparseError::EmptyMatrix);
        }

        if matrix.is_empty() {
            return Ok(MLE::new(vec![F::ZERO; rows]));
        }

        if matrix.len() % cols != 0 {
            return Err(SparseError::DimensionMismatch {
                expected: (cols, matrix.len()),
                actual: (cols, matrix.len()),
            });
        }

        let matrix_rows = matrix.len() / cols;

        if !matrix_rows.is_power_of_two() {
            return Err(SparseError::ValidationError(
                "Number of row vectors must be a power of two".to_string(),
            ));
        }

        let mut flattened = vec![F::ZERO; matrix_rows * rows];

        for ((row, col), &value) in self.iter() {
            if *row < rows && *col < cols {
                let column_start = *col * matrix_rows;
                for idx in 0..matrix_rows {
                    let matrix_entry = matrix[column_start + idx];
                    let output_offset = row * matrix_rows + idx;
                    flattened[output_offset] += matrix_entry * value;
                }
            }
        }

        Ok(MLE::new(flattened))
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
    fn test_sparse_mle_multiply_by_matrix_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((0, 1), BabyBear::from_u32(3));
        coeffs.insert((1, 0), BabyBear::from_u32(4));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
        ];

        let result = sparse_mle.multiply_by_matrix(&columns).unwrap();
        assert_eq!(result.len(), 4);
        let coeffs = result.coeffs();
        assert_eq!(coeffs[0], BabyBear::from_u32(8));
        assert_eq!(coeffs[1], BabyBear::from_u32(14));
        assert_eq!(coeffs[2], BabyBear::from_u32(18));
        assert_eq!(coeffs[3], BabyBear::from_u32(32));
    }

    #[test]
    fn test_sparse_mle_multiply_by_matrix_dimension_mismatch() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);
        coeffs.insert((0, 1), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![BabyBear::ONE, BabyBear::ONE, BabyBear::ONE];

        let result = sparse_mle.multiply_by_matrix(&columns);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_mle_multiply_by_matrix_empty_columns() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let empty_columns: [BabyBear; 0] = [];

        let result = sparse_mle.multiply_by_matrix(&empty_columns).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.coeffs().iter().all(|entry| *entry == BabyBear::ZERO));
    }

    #[test]
    fn test_sparse_mle_multiply_by_matrix_non_power_of_two_columns() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
            BabyBear::from_u32(5),
        ];

        let result = sparse_mle.multiply_by_matrix(&columns);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_sparse_mle_transpose_multiply_by_matrix_valid() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((0, 1), BabyBear::from_u32(3));
        coeffs.insert((1, 0), BabyBear::from_u32(4));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
        ];

        let result = sparse_mle.transpose_multiply_by_matrix(&columns).unwrap();
        assert_eq!(result.len(), 4);
        let coeffs = result.coeffs();
        assert_eq!(coeffs[0], BabyBear::from_u32(11));
        assert_eq!(coeffs[1], BabyBear::from_u32(16));
        assert_eq!(coeffs[2], BabyBear::from_u32(19));
        assert_eq!(coeffs[3], BabyBear::from_u32(28));
    }

    #[test]
    fn test_sparse_mle_transpose_multiply_by_matrix_dimension_mismatch() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::ONE);
        coeffs.insert((0, 1), BabyBear::ONE);

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![BabyBear::ONE, BabyBear::ONE, BabyBear::ONE];

        let result = sparse_mle.transpose_multiply_by_matrix(&columns);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_sparse_mle_transpose_multiply_by_matrix_non_power_of_two_columns() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::from_u32(2));
        coeffs.insert((1, 1), BabyBear::from_u32(5));

        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let columns = vec![
            BabyBear::ONE,
            BabyBear::ONE,
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
            BabyBear::from_u32(5),
        ];

        let result = sparse_mle.transpose_multiply_by_matrix(&columns);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
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
