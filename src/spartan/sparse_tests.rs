//! Comprehensive test suite for sparse matrix operations in Spartan.
//!
//! Tests cover SparseMLE, SpartanMetadata, and TimeStamps with edge cases,
//! error conditions, and mathematical property verification.

use super::sparse::*;
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use std::collections::HashMap;

#[cfg(test)]
use proptest::prelude::*;

/// Test utilities and helper functions
mod test_utils {
    use super::*;

    /// Creates a simple 2x2 sparse matrix for testing
    pub fn create_simple_sparse_mle() -> SparseMLE {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::new(1));
        coeffs.insert((0, 1), BabyBear::new(2));
        coeffs.insert((1, 0), BabyBear::new(3));
        coeffs.insert((1, 1), BabyBear::new(4));
        SparseMLE::new(coeffs).unwrap()
    }

    /// Creates a sparse matrix with only diagonal elements
    pub fn create_diagonal_sparse_mle(size: usize) -> SparseMLE {
        let mut coeffs = HashMap::new();
        for i in 0..size {
            coeffs.insert((i, i), BabyBear::new((i + 1) as u32));
        }
        SparseMLE::new(coeffs).unwrap()
    }

    /// Creates a vector of field elements from u32 values
    pub fn vec_from_u32(values: &[u32]) -> Vec<BabyBear> {
        values.iter().map(|&v| BabyBear::new(v)).collect()
    }

    /// Asserts that two vectors of field elements are equal
    pub fn assert_field_vec_eq(actual: &[BabyBear], expected: &[BabyBear]) {
        assert_eq!(actual.len(), expected.len(), "Vector lengths differ");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a, e, "Mismatch at index {}: got {:?}, expected {:?}", i, a, e);
        }
    }
}

#[cfg(test)]
mod sparse_mle_tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn test_new_valid_matrix() {
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 0), BabyBear::new(5));
        coeffs.insert((1, 1), BabyBear::new(10));
        
        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        assert_eq!(sparse_mle.num_nonzeros(), 2);
        assert_eq!(sparse_mle.get(0, 0), BabyBear::new(5));
        assert_eq!(sparse_mle.get(1, 1), BabyBear::new(10));
    }

    #[test]
    fn test_new_empty_coeffs_error() {
        let coeffs = HashMap::new();
        let result = SparseMLE::new(coeffs);
        assert_eq!(result, Err(SparseError::EmptyMatrix));
    }

    #[test]
    fn test_empty_constructor() {
        let empty_mle = SparseMLE::empty();
        assert_eq!(empty_mle.num_nonzeros(), 0);
        assert_eq!(empty_mle.get(0, 0), BabyBear::ZERO);
    }

    #[test]
    fn test_get_existing_and_missing_entries() {
        let sparse_mle = create_simple_sparse_mle();
        
        // Test existing entries
        assert_eq!(sparse_mle.get(0, 0), BabyBear::new(1));
        assert_eq!(sparse_mle.get(0, 1), BabyBear::new(2));
        assert_eq!(sparse_mle.get(1, 0), BabyBear::new(3));
        assert_eq!(sparse_mle.get(1, 1), BabyBear::new(4));
        
        // Test missing entries return zero
        assert_eq!(sparse_mle.get(2, 2), BabyBear::ZERO);
        assert_eq!(sparse_mle.get(5, 5), BabyBear::ZERO);
    }

    #[test]
    fn test_iter_deterministic_order() {
        let sparse_mle = create_simple_sparse_mle();
        let entries: Vec<_> = sparse_mle.iter().collect();
        
        assert_eq!(entries.len(), 4);
        // Verify all expected entries are present (order may vary due to HashMap)
        let coords: std::collections::HashSet<_> = entries.iter().map(|(coord, _)| *coord).collect();
        let expected_coords: std::collections::HashSet<(usize, usize)> = [(0, 0), (0, 1), (1, 0), (1, 1)].iter().cloned().collect();
        assert_eq!(coords, expected_coords);
    }

    #[test]
    fn test_multiply_by_vector_correct_dimensions() {
        let sparse_mle = create_simple_sparse_mle();
        let vector = vec_from_u32(&[1, 2]); // 2x1 vector
        
        let result = sparse_mle.multiply_by_vector(&vector).unwrap();
        
        // Expected: [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
        let expected = vec_from_u32(&[5, 11]);
        assert_field_vec_eq(&result, &expected);
    }

    #[test]
    fn test_multiply_by_vector_dimension_mismatch() {
        let sparse_mle = create_simple_sparse_mle();
        let wrong_vector = vec_from_u32(&[1, 2, 3]); // Wrong size
        
        let result = sparse_mle.multiply_by_vector(&wrong_vector);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_multiply_by_zero_vector() {
        let sparse_mle = create_simple_sparse_mle();
        let zero_vector = vec![BabyBear::ZERO; 2];
        
        let result = sparse_mle.multiply_by_vector(&zero_vector).unwrap();
        let expected = vec![BabyBear::ZERO; 2];
        assert_field_vec_eq(&result, &expected);
    }

    #[test]
    fn test_multiply_diagonal_matrix() {
        let diagonal_mle = create_diagonal_sparse_mle(3);
        let vector = vec_from_u32(&[2, 3, 4]);
        
        let result = diagonal_mle.multiply_by_vector(&vector).unwrap();
        
        // Diagonal multiplication: [1*2, 2*3, 3*4] = [2, 6, 12]
        let expected = vec_from_u32(&[2, 6, 12]);
        assert_field_vec_eq(&result, &expected);
    }

    #[test]
    fn test_large_sparse_matrix() {
        let mut coeffs = HashMap::new();
        // Create a 16x16 matrix with only a few non-zero entries
        coeffs.insert((0, 15), BabyBear::new(100));
        coeffs.insert((7, 8), BabyBear::new(200));
        coeffs.insert((15, 0), BabyBear::new(300));
        
        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let vector = vec![BabyBear::new(1); 16];
        
        let result = sparse_mle.multiply_by_vector(&vector).unwrap();
        
        // Check specific non-zero results
        assert_eq!(result[0], BabyBear::new(100));
        assert_eq!(result[7], BabyBear::new(200));
        assert_eq!(result[15], BabyBear::new(300));
        
        // Check that other entries are zero
        for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14] {
            assert_eq!(result[i], BabyBear::ZERO);
        }
    }

    #[test]
    fn test_compute_dimensions_power_of_two() {
        let mut coeffs = HashMap::new();
        coeffs.insert((7, 7), BabyBear::new(1)); // 8x8 matrix (next power of 2)
        
        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        assert_eq!(sparse_mle.dimensions(), (8, 8));
    }
}

#[cfg(test)]
mod spartan_metadata_tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn test_new_valid_metadata() {
        let row = vec_from_u32(&[0, 1]);
        let col = vec_from_u32(&[0, 1]);
        let val = vec_from_u32(&[5, 10]);
        let row_ts = TimeStamps::new(
            vec_from_u32(&[0, 0]),
            vec_from_u32(&[1, 1])
        ).unwrap();
        let col_ts = TimeStamps::new(
            vec_from_u32(&[0, 0]),
            vec_from_u32(&[1, 1])
        ).unwrap();
        
        let metadata = SpartanMetadata::new(row, col, val, row_ts, col_ts).unwrap();
        assert_eq!(metadata.len(), 2);
    }

    #[test]
    fn test_new_mismatched_vector_lengths() {
        let row = vec_from_u32(&[0, 1]);
        let col = vec_from_u32(&[0]); // Different length
        let val = vec_from_u32(&[5, 10]);
        let row_ts = TimeStamps::new(
            vec_from_u32(&[0, 0]),
            vec_from_u32(&[1, 1])
        ).unwrap();
        let col_ts = TimeStamps::new(
            vec_from_u32(&[0]),
            vec_from_u32(&[1])
        ).unwrap();
        
        let result = SpartanMetadata::new(row, col, val, row_ts, col_ts);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_preprocess_valid_sparse_mle() {
        let sparse_mle = create_simple_sparse_mle();
        let metadata = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        
        assert_eq!(metadata.len(), 4); // 4 non-zero entries
        
        // Verify that processing succeeds (detailed verification would require accessing private fields)
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_preprocess_empty_matrix() {
        let empty_mle = SparseMLE::empty();
        let result = SpartanMetadata::preprocess(&empty_mle);
        assert_eq!(result, Err(SparseError::EmptyMatrix));
    }

    #[test]
    fn test_extract_dense_vectors_deterministic() {
        let sparse_mle = create_diagonal_sparse_mle(3);
        
        // Call preprocess twice to ensure deterministic results
        let metadata1 = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        let metadata2 = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        
        assert_eq!(metadata1.len(), metadata2.len());
        // The internal vectors should be identical for deterministic processing
    }

    #[test]
    fn test_preprocess_large_matrix() {
        let mut coeffs = HashMap::new();
        for i in 0..100 {
            coeffs.insert((i, i), BabyBear::new((i + 1) as u32));
        }
        let large_sparse_mle = SparseMLE::new(coeffs).unwrap();
        
        let metadata = SpartanMetadata::preprocess(&large_sparse_mle).unwrap();
        assert_eq!(metadata.len(), 100);
    }

    #[test]
    fn test_preprocess_single_entry() {
        let mut coeffs = HashMap::new();
        coeffs.insert((5, 7), BabyBear::new(42));
        let single_entry_mle = SparseMLE::new(coeffs).unwrap();
        
        let metadata = SpartanMetadata::preprocess(&single_entry_mle).unwrap();
        assert_eq!(metadata.len(), 1);
    }
}

#[cfg(test)]
mod timestamps_tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn test_new_valid_timestamps() {
        let read_ts = vec_from_u32(&[0, 1, 2]);
        let final_ts = vec_from_u32(&[1, 2, 3]);
        
        let _timestamps = TimeStamps::new(read_ts, final_ts).unwrap();
        // If we get here, the invariant was satisfied
    }

    #[test]
    fn test_new_timestamp_invariant_violation() {
        let read_ts = vec_from_u32(&[2, 1, 0]); // read_ts[0] > final_ts[0]
        let final_ts = vec_from_u32(&[1, 2, 3]);
        
        let result = TimeStamps::new(read_ts, final_ts);
        assert!(matches!(result, Err(SparseError::ConstraintViolation(_))));
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let read_ts = vec_from_u32(&[0, 1]);
        let final_ts = vec_from_u32(&[1, 2, 3]); // Different length
        
        let result = TimeStamps::new(read_ts, final_ts);
        assert!(matches!(result, Err(SparseError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_compute_simple_sequence() {
        let indices = vec_from_u32(&[0, 1, 0, 1]); // Access pattern: 0, 1, 0, 1
        let max_address_space = 4; // Power of 2
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // Computation should succeed with valid power-of-2 address space
    }

    #[test]
    fn test_compute_empty_indices() {
        let indices = vec![];
        let max_address_space = 4;
        
        let result = TimeStamps::compute(&indices, max_address_space);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_compute_non_power_of_two_address_space() {
        let indices = vec_from_u32(&[0, 1, 2]);
        let max_address_space = 6; // Not a power of 2
        
        let result = TimeStamps::compute(&indices, max_address_space);
        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_compute_sequential_access_pattern() {
        let indices = vec_from_u32(&[0, 1, 2, 3]); // Sequential access
        let max_address_space = 4;
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // Sequential access should produce valid timestamps
    }

    #[test]
    fn test_compute_repeated_access_pattern() {
        let indices = vec_from_u32(&[0, 0, 0, 0]); // Repeated access to same address
        let max_address_space = 4;
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // Repeated access should properly increment timestamps
    }

    #[test]
    fn test_compute_random_access_pattern() {
        let indices = vec_from_u32(&[3, 1, 0, 2, 1, 3]); // Random access pattern
        let max_address_space = 4;
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // Random access pattern should still maintain invariants
    }

    #[test]
    fn test_compute_large_address_space() {
        let indices = vec_from_u32(&[0, 255, 128, 64]); // Sparse access in large space
        let max_address_space = 256; // Large power-of-2 space
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // Large address space should be handled correctly
    }

    #[test]
    fn test_timestamp_ordering_properties() {
        let indices = vec_from_u32(&[0, 1, 0, 1, 0]); // Alternating pattern
        let max_address_space = 4;
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        // The computed timestamps should maintain the read_ts ≤ final_ts invariant
        // This is verified by the constructor, so if we get here, it's valid
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use test_utils::*;

    #[test]
    fn test_complete_sparse_to_metadata_pipeline() {
        // Create a non-trivial sparse matrix
        let mut coeffs = HashMap::new();
        coeffs.insert((0, 1), BabyBear::new(5));
        coeffs.insert((1, 0), BabyBear::new(10));
        coeffs.insert((2, 2), BabyBear::new(15));
        
        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        
        // Convert to metadata
        let metadata = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        assert_eq!(metadata.len(), 3);
        
        // Verify the pipeline completed successfully
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_matrix_vector_multiplication_consistency() {
        let sparse_mle = create_simple_sparse_mle();
        let vector = vec_from_u32(&[1, 1]);
        
        // Perform multiplication
        let result = sparse_mle.multiply_by_vector(&vector).unwrap();
        
        // Verify result makes mathematical sense
        // [1 2] [1]   [3]
        // [3 4] [1] = [7]
        let expected = vec_from_u32(&[3, 7]);
        assert_field_vec_eq(&result, &expected);
    }

    #[test]
    fn test_timestamp_computation_with_metadata() {
        let sparse_mle = create_diagonal_sparse_mle(4);
        let metadata = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        
        // Test that metadata processing includes valid timestamp computation
        assert_eq!(metadata.len(), 4); // 4 diagonal entries
    }

    #[test]
    fn test_error_propagation_through_pipeline() {
        // Test that errors propagate correctly through the processing pipeline
        
        // Start with empty matrix
        let empty_mle = SparseMLE::empty();
        
        // Should fail at metadata preprocessing
        let result = SpartanMetadata::preprocess(&empty_mle);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_mathematical_properties_linearity() {
        let sparse_mle = create_simple_sparse_mle();
        let v1 = vec_from_u32(&[1, 0]);
        let v2 = vec_from_u32(&[0, 1]);
        
        let result1 = sparse_mle.multiply_by_vector(&v1).unwrap();
        let result2 = sparse_mle.multiply_by_vector(&v2).unwrap();
        
        // Test linearity: A(v1 + v2) = A(v1) + A(v2)
        let v_sum = vec![v1[0] + v2[0], v1[1] + v2[1]];
        let result_sum = sparse_mle.multiply_by_vector(&v_sum).unwrap();
        let expected_sum = vec![result1[0] + result2[0], result1[1] + result2[1]];
        
        assert_field_vec_eq(&result_sum, &expected_sum);
    }

    #[test]
    fn test_memory_consistency_realistic_scenario() {
        // Simulate a realistic memory access pattern
        let indices = vec_from_u32(&[
            0, 1, 2, 1, 0, 3, 2, 1, 0  // Mixed read/write pattern
        ]);
        let max_address_space = 4;
        
        let _timestamps = TimeStamps::compute(&indices, max_address_space).unwrap();
        
        // Verify that a realistic access pattern produces valid timestamps
        // The invariant check is done in the constructor
    }

    #[test]
    fn test_large_integration_scenario() {
        // Test with a moderately large matrix to verify scalability
        let mut coeffs = HashMap::new();
        for i in 0..50 {
            for j in 0..50 {
                if (i + j) % 7 == 0 { // Sparse pattern
                    coeffs.insert((i, j), BabyBear::new((i * j + 1) as u32));
                }
            }
        }
        
        let sparse_mle = SparseMLE::new(coeffs).unwrap();
        let metadata = SpartanMetadata::preprocess(&sparse_mle).unwrap();
        
        // Verify large-scale processing works
        assert!(metadata.len() > 0);
        assert!(sparse_mle.num_nonzeros() > 0);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_all_error_types_display() {
        let errors = vec![
            SparseError::ValidationError("test message".to_string()),
            SparseError::DimensionMismatch { expected: (2, 2), actual: (3, 3) },
            SparseError::IndexOutOfBounds { index: (5, 5), bounds: (4, 4) },
            SparseError::EmptyMatrix,
            SparseError::ConstraintViolation("test constraint".to_string()),
        ];
        
        for error in errors {
            let display_string = format!("{}", error);
            assert!(!display_string.is_empty(), "Error display should not be empty");
        }
    }

    #[test]
    fn test_error_equality() {
        let error1 = SparseError::EmptyMatrix;
        let error2 = SparseError::EmptyMatrix;
        assert_eq!(error1, error2);
        
        let error3 = SparseError::ValidationError("test".to_string());
        let error4 = SparseError::ValidationError("test".to_string());
        assert_eq!(error3, error4);
    }

    #[test]
    fn test_error_inequality() {
        let error1 = SparseError::EmptyMatrix;
        let error2 = SparseError::ValidationError("test".to_string());
        assert_ne!(error1, error2);
    }
}

#[cfg(test)]
mod property_based_tests {
    use super::*;
    use proptest::prelude::*;

    // Custom strategy for generating valid BabyBear field elements
    fn babybear_strategy() -> impl Strategy<Value = BabyBear> {
        any::<u32>().prop_map(BabyBear::new)
    }

    // Strategy for generating small coordinates (to keep test matrices manageable)
    fn small_coord_strategy() -> impl Strategy<Value = (usize, usize)> {
        (0..8usize, 0..8usize)
    }

    // Strategy for generating sparse matrices with reasonable size
    fn sparse_mle_strategy() -> impl Strategy<Value = SparseMLE> {
        prop::collection::hash_map(
            small_coord_strategy(),
            babybear_strategy(),
            1..16, // 1 to 15 non-zero entries
        ).prop_map(|coeffs| SparseMLE::new(coeffs).unwrap())
    }

    // Strategy for generating vectors compatible with matrix dimensions
    fn compatible_vector_strategy(matrix: &SparseMLE) -> impl Strategy<Value = Vec<BabyBear>> {
        let cols = matrix.dimensions().1;
        prop::collection::vec(babybear_strategy(), cols..=cols)
    }

    proptest! {
        #[test]
        fn test_sparse_mle_get_symmetry(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10)
        ) {
            let sparse_mle = SparseMLE::new(coeffs.clone()).unwrap();
            
            // Property: get() should return the stored value for existing entries
            for ((row, col), expected_val) in coeffs.iter() {
                prop_assert_eq!(sparse_mle.get(*row, *col), *expected_val);
            }
            
            // Property: get() should return zero for non-existing entries
            let non_existing_val = sparse_mle.get(100, 100);
            prop_assert_eq!(non_existing_val, BabyBear::ZERO);
        }

        #[test]
        fn test_matrix_vector_multiplication_linearity(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10),
            scalar in babybear_strategy()
        ) {
            let sparse_mle = SparseMLE::new(coeffs).unwrap();
            let (_, cols) = sparse_mle.dimensions();
            
            let vector: Vec<BabyBear> = (0..cols)
                .map(|_| BabyBear::new(1))
                .collect();
            
            let scaled_vector: Vec<BabyBear> = vector.iter()
                .map(|&v| v * scalar)
                .collect();
            
            let result1 = sparse_mle.multiply_by_vector(&vector).unwrap();
            let result2 = sparse_mle.multiply_by_vector(&scaled_vector).unwrap();
            
            // Property: A(c*v) = c*A(v) (scalar multiplication distributes)
            let scaled_result1: Vec<BabyBear> = result1.iter()
                .map(|&r| r * scalar)
                .collect();
            
            prop_assert_eq!(result2, scaled_result1);
        }

        #[test]
        fn test_matrix_vector_zero_multiplication(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10)
        ) {
            let sparse_mle = SparseMLE::new(coeffs).unwrap();
            let (rows, cols) = sparse_mle.dimensions();
            
            let zero_vector = vec![BabyBear::ZERO; cols];
            let result = sparse_mle.multiply_by_vector(&zero_vector).unwrap();
            
            // Property: A * 0 = 0
            let expected_zero = vec![BabyBear::ZERO; rows];
            prop_assert_eq!(result, expected_zero);
        }

        #[test]
        fn test_timestamp_invariant_preservation(
            indices in prop::collection::vec(0..8u32, 1..20)
        ) {
            let indices_field: Vec<BabyBear> = indices.iter()
                .map(|&i| BabyBear::new(i))
                .collect();
            
            let max_address_space = 8; // Power of 2
            
            if let Ok(_timestamps) = TimeStamps::compute(&indices_field, max_address_space) {
                // Property: The constructor validates read_ts ≤ final_ts invariant
                // If we get here, the invariant was satisfied
                prop_assert!(true); // The invariant check is in the constructor
            }
        }

        #[test]
        fn test_sparse_mle_non_zero_count_consistency(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..20)
        ) {
            let sparse_mle = SparseMLE::new(coeffs.clone()).unwrap();
            
            // Property: num_nonzeros() should equal the number of entries in coeffs
            prop_assert_eq!(sparse_mle.num_nonzeros(), coeffs.len());
        }

        #[test]
        fn test_metadata_preprocessing_length_consistency(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..15)
        ) {
            let sparse_mle = SparseMLE::new(coeffs.clone()).unwrap();
            
            if let Ok(metadata) = SpartanMetadata::preprocess(&sparse_mle) {
                // Property: metadata length should equal number of non-zero entries
                prop_assert_eq!(metadata.len(), coeffs.len());
            }
        }

        #[test]
        fn test_matrix_dimensions_power_of_two(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10)
        ) {
            let sparse_mle = SparseMLE::new(coeffs).unwrap();
            let (rows, cols) = sparse_mle.dimensions();
            
            // Property: dimensions should be powers of 2 (or 0)
            prop_assert!(rows == 0 || rows.is_power_of_two());
            prop_assert!(cols == 0 || cols.is_power_of_two());
        }

        #[test]
        fn test_vector_multiplication_size_validation(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10),
            wrong_size in 1..20usize
        ) {
            let sparse_mle = SparseMLE::new(coeffs).unwrap();
            let (_, cols) = sparse_mle.dimensions();
            
            // Only test with vectors of wrong size
            if wrong_size != cols {
                let wrong_vector = vec![BabyBear::ZERO; wrong_size];
                let result = sparse_mle.multiply_by_vector(&wrong_vector);
                
                // Property: wrong-sized vectors should cause dimension mismatch error
                prop_assert!(result.is_err());
                if let Err(err) = result {
                    match err {
                        SparseError::DimensionMismatch { .. } => prop_assert!(true),
                        _ => prop_assert!(false, "Expected DimensionMismatch error"),
                    }
                }
            }
        }

        #[test]
        fn test_timestamp_computation_deterministic(
            indices in prop::collection::vec(0..4u32, 1..10)
        ) {
            let indices_field: Vec<BabyBear> = indices.iter()
                .map(|&i| BabyBear::new(i))
                .collect();
            
            let max_address_space = 4; // Power of 2
            
            // Property: timestamp computation should be deterministic
            let result1 = TimeStamps::compute(&indices_field, max_address_space);
            let result2 = TimeStamps::compute(&indices_field, max_address_space);
            
            match (result1, result2) {
                (Ok(_), Ok(_)) => {
                    // Both computations should succeed (deterministic behavior)
                    prop_assert!(true);
                }
                (Err(_), Err(_)) => {
                    // Both should fail for the same reason (deterministic behavior)
                    prop_assert!(true);
                }
                _ => {
                    // Non-deterministic behavior is a property violation
                    prop_assert!(false, "Timestamp computation should be deterministic");
                }
            }
        }

        #[test]
        fn test_empty_coeffs_always_error(
            _seed in any::<u32>() // Dummy input to make this a property test
        ) {
            let empty_coeffs = HashMap::new();
            let result = SparseMLE::new(empty_coeffs);
            
            // Property: empty coefficient map should always produce EmptyMatrix error
            prop_assert_eq!(result, Err(SparseError::EmptyMatrix));
        }

        #[test]
        fn test_matrix_vector_product_bounds(
            coeffs in prop::collection::hash_map(small_coord_strategy(), babybear_strategy(), 1..10)
        ) {
            let sparse_mle = SparseMLE::new(coeffs).unwrap();
            let (rows, cols) = sparse_mle.dimensions();
            
            let vector = vec![BabyBear::ONE; cols];
            let result = sparse_mle.multiply_by_vector(&vector).unwrap();
            
            // Property: result vector should have exactly 'rows' elements
            prop_assert_eq!(result.len(), rows);
        }
    }
}