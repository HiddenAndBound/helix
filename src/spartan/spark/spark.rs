//! Spark opening oracles for efficient sparse polynomial commitment opening.
//!
//! This module provides functionality to generate e_rx and e_ry oracles for the Spark
//! protocol, enabling efficient opening of polynomial commitments to sparse multilinear
//! extension polynomials.
//!
//! ## Key Components
//! 
//! - [`SparkOracles`]: Struct containing separate e_rx and e_ry oracles for matrices A, B, and C
//! - [`generate_spark_opening_oracles`]: Main function to generate all oracles from evaluation point
//! - [`generate_single_oracle_pair`]: Utility function to generate oracles for a single matrix

use crate::spartan::error::{SparseError, SparseResult};
use crate::spartan::spark::sparse::SpartanMetadata;
use crate::utils::{Fp4, eq::EqEvals, polynomial::MLE};
use p3_field::PrimeField32;

/// Spark oracles for polynomial commitment opening across all three constraint matrices.
///
/// Contains separate e_rx and e_ry oracles for matrices A, B, and C, where:
/// - e_rx oracles map row indices to rx equality polynomial evaluations
/// - e_ry oracles map column indices to ry equality polynomial evaluations
#[derive(Debug, Clone)]
pub struct SparkOracles {
    /// e_rx oracle for constraint matrix A
    pub e_rx_a: MLE<Fp4>,
    /// e_ry oracle for constraint matrix A
    pub e_ry_a: MLE<Fp4>,
    /// e_rx oracle for constraint matrix B
    pub e_rx_b: MLE<Fp4>,
    /// e_ry oracle for constraint matrix B
    pub e_ry_b: MLE<Fp4>,
    /// e_rx oracle for constraint matrix C
    pub e_rx_c: MLE<Fp4>,
    /// e_ry oracle for constraint matrix C
    pub e_ry_c: MLE<Fp4>,
}

/// Generates e_rx and e_ry oracles for Spark opening of polynomial commitments.
///
/// This function takes the preprocessed metadata for the three constraint matrices
/// (A, B, C) and an evaluation point that is the concatenation of the outer sum-check
/// challenge (rx) and inner sum-check challenge (ry). It generates separate equality
/// polynomial evaluations for rx and ry, then creates oracle mappings for efficient
/// polynomial commitment opening.
///
/// # Arguments
/// * `metadata_a` - Preprocessed metadata for constraint matrix A
/// * `metadata_b` - Preprocessed metadata for constraint matrix B
/// * `metadata_c` - Preprocessed metadata for constraint matrix C
/// * `evaluation_point` - Concatenated evaluation point [rx || ry] from sum-check protocol
///
/// # Returns
/// SparkOracles struct containing separate e_rx and e_ry oracles for matrices A, B, C
///
/// # Errors
/// - `ValidationError` if evaluation_point has odd length (cannot split into rx/ry)
/// - `ValidationError` if metadata vectors are empty
/// - `IndexOutOfBounds` if row/col indices exceed EqEvals bounds
///
/// # Protocol Context
/// In the Spark protocol:
/// 1. The outer sum-check generates challenges rx
/// 2. The inner sum-check generates challenges ry  
/// 3. The evaluation point is rx ⊕ ry (concatenation)
/// 4. For each matrix metadata, we create oracles where:
///    - e_rx[i] = eq_rx[row[i]] (row indices map to rx equality polynomial)
///    - e_ry[i] = eq_ry[col[i]] (column indices map to ry equality polynomial)
pub fn generate_spark_opening_oracles(
    metadata_a: &SpartanMetadata,
    metadata_b: &SpartanMetadata,
    metadata_c: &SpartanMetadata,
    evaluation_point: &[Fp4],
) -> SparseResult<SparkOracles> {
    // Validate evaluation point can be split into rx and ry
    if evaluation_point.len() % 2 != 0 {
        return Err(SparseError::ValidationError(format!(
            "Evaluation point length {} must be even to split into rx and ry",
            evaluation_point.len()
        )));
    }

    let half_len = evaluation_point.len() / 2;

    // Split evaluation point into rx (first half) and ry (second half)
    let rx_point = &evaluation_point[..half_len];
    let ry_point = &evaluation_point[half_len..];

    // Generate equality polynomial evaluations for rx and ry
    let eq_rx = EqEvals::gen_from_point(rx_point);
    let eq_ry = EqEvals::gen_from_point(ry_point);

    // Generate oracles for all three matrices
    generate_oracle_pair(metadata_a, metadata_b, metadata_c, &eq_rx, &eq_ry)
}

/// Generates e_rx and e_ry oracle pairs for all three constraint matrices.
///
/// For each position i in each metadata:
/// - e_rx[i] = eq_rx[row[i]] (equality polynomial evaluated at row index)
/// - e_ry[i] = eq_ry[col[i]] (equality polynomial evaluated at column index)
fn generate_oracle_pair(
    metadata_a: &SpartanMetadata,
    metadata_b: &SpartanMetadata,
    metadata_c: &SpartanMetadata,
    eq_rx: &EqEvals,
    eq_ry: &EqEvals,
) -> SparseResult<SparkOracles> {
    // Generate oracle pairs for each metadata
    let (e_rx_a, e_ry_a) = generate_single_oracle_pair(metadata_a, eq_rx, eq_ry)?;
    let (e_rx_b, e_ry_b) = generate_single_oracle_pair(metadata_b, eq_rx, eq_ry)?;
    let (e_rx_c, e_ry_c) = generate_single_oracle_pair(metadata_c, eq_rx, eq_ry)?;

    Ok(SparkOracles {
        e_rx_a,
        e_ry_a,
        e_rx_b,
        e_ry_b,
        e_rx_c,
        e_ry_c,
    })
}

/// Generates e_rx and e_ry oracle pair for a single metadata.
///
/// For each position i in the metadata:
/// - e_rx[i] = eq_rx[row[i]] (equality polynomial evaluated at row index)
/// - e_ry[i] = eq_ry[col[i]] (equality polynomial evaluated at column index)
pub fn generate_single_oracle_pair(
    metadata: &SpartanMetadata,
    eq_rx: &EqEvals,
    eq_ry: &EqEvals,
) -> SparseResult<(MLE<Fp4>, MLE<Fp4>)> {
    let len = metadata.len();

    if len == 0 {
        return Err(SparseError::ValidationError(
            "Cannot generate oracles for empty metadata".to_string(),
        ));
    }

    let mut e_rx_vec = Vec::with_capacity(len);
    let mut e_ry_vec = Vec::with_capacity(len);

    // Access the underlying coefficients of the MLEs
    let row_coeffs = metadata.row().coeffs();
    let col_coeffs = metadata.col().coeffs();

    for i in 0..len {
        // Convert field elements to indices
        let row_index = row_coeffs[i].as_canonical_u32() as usize;
        let col_index = col_coeffs[i].as_canonical_u32() as usize;

        // Validate indices are within bounds
        if row_index >= eq_rx.coeffs.len() {
            return Err(SparseError::IndexOutOfBounds {
                index: (row_index, 0),
                bounds: (eq_rx.coeffs.len(), 0),
            });
        }
        if col_index >= eq_ry.coeffs.len() {
            return Err(SparseError::IndexOutOfBounds {
                index: (col_index, 0),
                bounds: (eq_ry.coeffs.len(), 0),
            });
        }

        // Populate oracle vectors
        e_rx_vec.push(eq_rx.coeffs[row_index]);
        e_ry_vec.push(eq_ry.coeffs[col_index]);
    }

    // Convert to MLEs
    let e_rx_mle = MLE::new(e_rx_vec);
    let e_ry_mle = MLE::new(e_ry_vec);

    Ok((e_rx_mle, e_ry_mle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spartan::spark::sparse::SparseMLE;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use std::collections::HashMap;

    fn create_test_metadata(entries: Vec<(usize, usize, u32)>) -> SpartanMetadata {
        // Create sparse MLE from test entries
        let mut coeffs = HashMap::new();
        for (row, col, val) in entries {
            coeffs.insert((row, col), BabyBear::from_u32(val));
        }
        let sparse_mle = SparseMLE::new(coeffs).unwrap();

        // Preprocess to metadata
        SpartanMetadata::preprocess(&sparse_mle).unwrap()
    }

    #[test]
    fn test_generate_spark_opening_oracles_simple() {
        // Create simple 2x2 test matrices
        let metadata_a = create_test_metadata(vec![(0, 0, 1), (1, 1, 2)]);
        let metadata_b = create_test_metadata(vec![(0, 1, 3), (1, 0, 4)]);
        let metadata_c = create_test_metadata(vec![(0, 0, 5), (0, 1, 6)]);

        // Evaluation point with 2 variables (1 for rx, 1 for ry)
        let evaluation_point = vec![
            Fp4::from_u32(2), // rx
            Fp4::from_u32(3), // ry
        ];

        let result = generate_spark_opening_oracles(
            &metadata_a,
            &metadata_b,
            &metadata_c,
            &evaluation_point,
        )
        .unwrap();

        // Verify we get SparkOracles with all 6 oracle fields
        // Each oracle should have MLEs with length matching metadata
        assert_eq!(result.e_rx_a.len(), metadata_a.len()); // e_rx for A
        assert_eq!(result.e_ry_a.len(), metadata_a.len()); // e_ry for A
        assert_eq!(result.e_rx_b.len(), metadata_b.len()); // e_rx for B
        assert_eq!(result.e_ry_b.len(), metadata_b.len()); // e_ry for B
        assert_eq!(result.e_rx_c.len(), metadata_c.len()); // e_rx for C
        assert_eq!(result.e_ry_c.len(), metadata_c.len()); // e_ry for C
    }

    #[test]
    fn test_generate_spark_opening_oracles_manual_verification() {
        // Create simple metadata with known indices
        let metadata = create_test_metadata(vec![(1, 0, 10)]); // row=1, col=0, val=10

        // Evaluation point: rx=[2], ry=[3]
        let evaluation_point = vec![Fp4::from_u32(2), Fp4::from_u32(3)];

        let result =
            generate_spark_opening_oracles(&metadata, &metadata, &metadata, &evaluation_point)
                .unwrap();

        // For this test case:
        // - eq_rx for point [2] gives coefficients [(1-2), 2] = [-1, 2]
        // - eq_ry for point [3] gives coefficients [(1-3), 3] = [-2, 3]
        // - metadata has row[0] = 1, col[0] = 0
        // - So e_rx[0] = eq_rx[1] = 2, e_ry[0] = eq_ry[0] = -2

        let expected_e_rx_0 = Fp4::from_u32(2);
        let expected_e_ry_0 = Fp4::ONE - Fp4::from_u32(3); // (1-3) = -2

        assert_eq!(result.e_rx_a.coeffs()[0], expected_e_rx_0);
        assert_eq!(result.e_ry_a.coeffs()[0], expected_e_ry_0);
    }

    #[test]
    fn test_generate_spark_opening_oracles_odd_evaluation_point() {
        let metadata = create_test_metadata(vec![(0, 0, 1)]);

        // Odd length evaluation point should fail
        let odd_evaluation_point = vec![Fp4::from_u32(1), Fp4::from_u32(2), Fp4::from_u32(3)];

        let result =
            generate_spark_opening_oracles(&metadata, &metadata, &metadata, &odd_evaluation_point);

        assert!(matches!(result, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_generate_oracle_pair_bounds_checking() {
        let metadata = create_test_metadata(vec![(2, 3, 1)]); // row=2, col=3 (out of bounds)

        // Small evaluation points that will create small EqEvals
        let rx_point = vec![Fp4::from_u32(1)]; // Creates 2 coefficients (indices 0,1)
        let ry_point = vec![Fp4::from_u32(2)]; // Creates 2 coefficients (indices 0,1)

        let eq_rx = EqEvals::gen_from_point(&rx_point);
        let eq_ry = EqEvals::gen_from_point(&ry_point);

        // Should fail because row=2 and col=3 are out of bounds for 2-element EqEvals
        let result = generate_single_oracle_pair(&metadata, &eq_rx, &eq_ry);
        assert!(matches!(result, Err(SparseError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_generate_oracle_pair_empty_metadata() {
        // Create empty metadata (this should actually fail at metadata creation)
        // But let's test the oracle generation boundary
        let empty_coeffs = HashMap::new();
        let result = SparseMLE::new(empty_coeffs);
        assert!(matches!(result, Err(SparseError::EmptyMatrix)));
    }

    #[test]
    fn test_generate_spark_opening_oracles_four_variables() {
        // Test with larger evaluation points (4 variables: 2 for rx, 2 for ry)
        let metadata = create_test_metadata(vec![(0, 1, 7), (2, 0, 8)]);

        let evaluation_point = vec![
            Fp4::from_u32(1),
            Fp4::from_u32(2), // rx = [1, 2]
            Fp4::from_u32(3),
            Fp4::from_u32(4), // ry = [3, 4]
        ];

        let result =
            generate_spark_opening_oracles(&metadata, &metadata, &metadata, &evaluation_point)
                .unwrap();

        // Verify structure is correct
        assert_eq!(result.e_rx_a.len(), 2); // Two non-zero entries in metadata
        assert_eq!(result.e_ry_a.len(), 2);
        assert_eq!(result.e_rx_b.len(), 2); // Same metadata used for B
        assert_eq!(result.e_ry_b.len(), 2);
        assert_eq!(result.e_rx_c.len(), 2); // Same metadata used for C
        assert_eq!(result.e_ry_c.len(), 2);
    }

    #[test]
    fn test_oracle_values_consistency() {
        // Test that oracle values are consistent with EqEvals lookups
        // Use 4 entries to get a power of 2
        let metadata = create_test_metadata(vec![(0, 0, 1), (1, 1, 2), (0, 1, 3), (1, 0, 4)]);

        let rx_point = vec![Fp4::from_u32(5)];
        let ry_point = vec![Fp4::from_u32(7)];
        let evaluation_point = [rx_point.clone(), ry_point.clone()].concat();

        let result =
            generate_spark_opening_oracles(&metadata, &metadata, &metadata, &evaluation_point)
                .unwrap();

        // Manual verification: create EqEvals and check lookups
        let eq_rx = EqEvals::gen_from_point(&rx_point);
        let eq_ry = EqEvals::gen_from_point(&ry_point);

        // metadata has entries at (0,0), (0,1), (1,0), (1,1) (sorted order)
        // So row indices are [0, 0, 1, 1] and col indices are [0, 1, 0, 1]
        let expected_e_rx = vec![eq_rx[0], eq_rx[0], eq_rx[1], eq_rx[1]];
        let expected_e_ry = vec![eq_ry[0], eq_ry[1], eq_ry[0], eq_ry[1]];

        for i in 0..4 {
            assert_eq!(result.e_rx_a.coeffs()[i], expected_e_rx[i]);
            assert_eq!(result.e_ry_a.coeffs()[i], expected_e_ry[i]);
            // All three matrices use the same metadata, so they should have the same values
            assert_eq!(result.e_rx_b.coeffs()[i], expected_e_rx[i]);
            assert_eq!(result.e_ry_b.coeffs()[i], expected_e_ry[i]);
            assert_eq!(result.e_rx_c.coeffs()[i], expected_e_rx[i]);
            assert_eq!(result.e_ry_c.coeffs()[i], expected_e_ry[i]);
        }
    }
}
