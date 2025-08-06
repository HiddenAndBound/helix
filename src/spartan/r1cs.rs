//! R1CS (Rank-1 Constraint System) implementation for the Spartan zkSNARK protocol.
//!
//! This module provides the core data structures and operations for working with
//! Rank-1 Constraint Systems, which are central to the Spartan protocol's constraint
//! satisfaction verification.
//!
//! Mathematical specification:
//! Given constraint matrices A, B, C ∈ F^(m×n) and witness vector z ∈ F^n,
//! the constraint system is satisfied if: (A·z) ∘ (B·z) = C·z
//! where ∘ denotes the Hadamard (element-wise) product.

use crate::spartan::error::{SparseError, SparseResult};
use crate::spartan::sparse::SparseMLE;
use crate::utils::polynomial::MLE;
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use std::collections::HashMap;

/// A Rank-1 Constraint System (R1CS) instance.
///
/// Represents a set of constraints defined by three sparse matrices A, B, C
/// where each constraint i is satisfied if: (A_i · z) * (B_i · z) = C_i · z
#[derive(Debug, Clone, PartialEq)]
pub struct R1CS {
    /// Constraint matrix A: coefficients for left input of multiplication gates
    pub a: SparseMLE,
    /// Constraint matrix B: coefficients for right input of multiplication gates
    pub b: SparseMLE,
    /// Constraint matrix C: coefficients for output of multiplication gates
    pub c: SparseMLE,
    /// Number of constraints (rows in matrices)
    pub num_constraints: usize,
    /// Number of variables (columns in matrices)
    pub num_variables: usize,
    /// Number of public inputs (first variables in witness)
    pub num_public_inputs: usize,
}

impl R1CS {
    /// Creates a new R1CS instance from constraint matrices.
    ///
    /// # Arguments
    /// * `a` - Constraint matrix A
    /// * `b` - Constraint matrix B  
    /// * `c` - Constraint matrix C
    /// * `num_public_inputs` - Number of public input variables
    ///
    /// # Returns
    /// A new R1CS instance or an error if matrices are incompatible
    pub fn new(
        a: SparseMLE,
        b: SparseMLE,
        c: SparseMLE,
        num_public_inputs: usize,
    ) -> SparseResult<Self> {
        // Validate matrix dimensions match
        let (a_rows, a_cols) = a.dimensions();
        let (b_rows, b_cols) = b.dimensions();
        let (c_rows, c_cols) = c.dimensions();

        if a_rows != b_rows || b_rows != c_rows {
            return Err(SparseError::ValidationError(format!(
                "Matrix row count mismatch: A={}, B={}, C={}",
                a_rows, b_rows, c_rows
            )));
        }

        if a_cols != b_cols || b_cols != c_cols {
            return Err(SparseError::ValidationError(format!(
                "Matrix column count mismatch: A={}, B={}, C={}",
                a_cols, b_cols, c_cols
            )));
        }

        if num_public_inputs > a_cols {
            return Err(SparseError::ValidationError(format!(
                "Number of public inputs {} exceeds number of variables {}",
                num_public_inputs, a_cols
            )));
        }

        Ok(R1CS {
            a,
            b,
            c,
            num_constraints: a_rows,
            num_variables: a_cols,
            num_public_inputs,
        })
    }

    /// Creates an empty R1CS instance with no constraints.
    pub fn empty() -> Self {
        R1CS {
            a: SparseMLE::empty(),
            b: SparseMLE::empty(),
            c: SparseMLE::empty(),
            num_constraints: 0,
            num_variables: 0,
            num_public_inputs: 0,
        }
    }

    /// Validates that the constraint matrices have consistent dimensions.
    pub fn validate_dimensions(&self) -> SparseResult<()> {
        let (a_rows, a_cols) = self.a.dimensions();
        let (b_rows, b_cols) = self.b.dimensions();
        let (c_rows, c_cols) = self.c.dimensions();

        if a_rows != b_rows || b_rows != c_rows || a_rows != self.num_constraints {
            return Err(SparseError::ValidationError(
                "Inconsistent constraint matrix dimensions".to_string(),
            ));
        }

        if a_cols != b_cols || b_cols != c_cols || a_cols != self.num_variables {
            return Err(SparseError::ValidationError(
                "Inconsistent variable count across matrices".to_string(),
            ));
        }

        Ok(())
    }

    /// Computes the constraint satisfaction check: (A·z) ∘ (B·z) = C·z
    ///
    /// # Arguments
    /// * `z` - The full witness vector including public inputs and private variables
    ///
    /// # Returns
    /// A vector indicating which constraints are satisfied
    pub fn check_constraints(&self, z: &MLE<BabyBear>) -> SparseResult<Vec<bool>> {
        if z.len() != self.num_variables {
            return Err(SparseError::DimensionMismatch {
                expected: (self.num_variables, z.len()),
                actual: (self.num_variables, z.len()),
            });
        }

        // Compute A·z, B·z, and C·z
        let az = self.a.multiply_by_mle(z)?;
        let bz = self.b.multiply_by_mle(z)?;
        let cz = self.c.multiply_by_mle(z)?;

        // Check (A·z) ∘ (B·z) = C·z element-wise
        let mut satisfied = vec![false; self.num_constraints];

        for i in 0..self.num_constraints {
            let left = az.coeffs()[i] * bz.coeffs()[i];
            let right = cz.coeffs()[i];
            satisfied[i] = left == right;
        }

        Ok(satisfied)
    }

    /// Validates that all constraints are satisfied by a given witness.
    pub fn verify(&self, z: &MLE<BabyBear>) -> SparseResult<bool> {
        let satisfied = self.check_constraints(z)?;
        Ok(satisfied.iter().all(|&s| s))
    }

    /// Creates a simple test R1CS instance for development and testing.
    ///
    /// Creates a constraint system representing: x * y = z
    /// with public input x = 2 and solution y = 3, z = 6
    pub fn simple_test_instance() -> SparseResult<(Self, Witness)> {
        // Constraint: x * y = z
        // Variables: [x, y, z] where x is public input
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        let mut c_coeffs = HashMap::new();

        // Use 8 columns to ensure consistent dimensions (power of 2)

        // Constraint: x * y = z
        // A matrix: selects x (column 0)
        a_coeffs.insert((0, 0), BabyBear::ONE);
        // B matrix: selects y (column 1)
        b_coeffs.insert((0, 1), BabyBear::ONE);
        // C matrix: selects z (column 2)
        c_coeffs.insert((0, 2), BabyBear::ONE);

        // Add dummy entries to ensure all matrices have 8 columns
        for i in 3..8 {
            a_coeffs.insert((0, i), BabyBear::new(0));
            b_coeffs.insert((0, i), BabyBear::new(0));
            c_coeffs.insert((0, i), BabyBear::new(0));
        }

        let a = SparseMLE::new(a_coeffs)?;
        let b = SparseMLE::new(b_coeffs)?;
        let c = SparseMLE::new(c_coeffs)?;

        let r1cs = R1CS::new(a, b, c, 1)?;

        // Create witness: x = 2 (public), y = 3, z = 6, rest = 0
        let mut witness_vars = vec![BabyBear::new(0); 8];
        witness_vars[0] = BabyBear::new(2); // x = 2
        witness_vars[1] = BabyBear::new(3); // y = 3
        witness_vars[2] = BabyBear::new(6); // z = 6
        // Remaining positions [3..7] stay 0

        let witness = Witness::new(
            vec![BabyBear::new(2)], // public input x = 2
            vec![
                BabyBear::new(3), // y = 3
                BabyBear::new(6), // z = 6
                BabyBear::new(0), // padding
                BabyBear::new(0), // padding
                BabyBear::new(0), // padding
                BabyBear::new(0), // padding
                BabyBear::new(0), // padding
            ], // private vars + padding to make 7 total
        )?;

        Ok((r1cs, witness))
    }

    /// Creates a test R1CS instance with multiple constraints.
    ///
    /// Creates constraints:
    /// 1. x1 * x2 = y1
    /// 2. y1 * x3 = y2
    /// 3. y2 * x4 = out
    pub fn multi_constraint_test_instance() -> SparseResult<(Self, Witness)> {
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        let mut c_coeffs = HashMap::new();

        // Use 8 columns to ensure consistent dimensions

        // Constraint 1: x1 * x2 = y1
        a_coeffs.insert((0, 0), BabyBear::ONE); // x1
        b_coeffs.insert((0, 1), BabyBear::ONE); // x2
        c_coeffs.insert((0, 4), BabyBear::ONE); // y1

        // Constraint 2: y1 * x3 = y2
        a_coeffs.insert((1, 4), BabyBear::ONE); // y1
        b_coeffs.insert((1, 2), BabyBear::ONE); // x3
        c_coeffs.insert((1, 5), BabyBear::ONE); // y2

        // Constraint 3: y2 * x4 = out
        a_coeffs.insert((2, 5), BabyBear::ONE); // y2
        b_coeffs.insert((2, 3), BabyBear::ONE); // x4
        c_coeffs.insert((2, 6), BabyBear::ONE); // out

        // Ensure consistent dimensions by padding with zeros
        for matrix in [&mut a_coeffs, &mut b_coeffs, &mut c_coeffs] {
            for row in 0..3 {
                for col in 0..8 {
                    if !matrix.contains_key(&(row, col)) {
                        matrix.insert((row, col), BabyBear::new(0));
                    }
                }
            }
        }

        let a = SparseMLE::new(a_coeffs)?;
        let b = SparseMLE::new(b_coeffs)?;
        let c = SparseMLE::new(c_coeffs)?;

        let r1cs = R1CS::new(a, b, c, 4)?;

        // Create witness: x1=2, x2=3, x3=4, x4=5 (public inputs)
        // Then: y1=6, y2=24, out=120, rest = 0
        let mut witness_vars = vec![BabyBear::new(0); 8];
        witness_vars[0] = BabyBear::new(2); // x1 = 2
        witness_vars[1] = BabyBear::new(3); // x2 = 3
        witness_vars[2] = BabyBear::new(4); // x3 = 4
        witness_vars[3] = BabyBear::new(5); // x4 = 5
        witness_vars[4] = BabyBear::new(6); // y1 = 2*3
        witness_vars[5] = BabyBear::new(24); // y2 = 6*4
        witness_vars[6] = BabyBear::new(120); // out = 24*5
        witness_vars[7] = BabyBear::new(0); // padding

        let witness = Witness::new(
            vec![
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
            ],
            vec![
                BabyBear::new(6),   // y1 = 2*3
                BabyBear::new(24),  // y2 = 6*4
                BabyBear::new(120), // out = 24*5
                BabyBear::new(0),   // padding
            ],
        )?;

        Ok((r1cs, witness))
    }
}

/// Witness vector containing public inputs and private variables.
///
/// The witness is structured as z = [public_inputs || private_variables]
/// where public_inputs are known to the verifier and private_variables are secret.
#[derive(Debug, Clone, PartialEq)]
pub struct Witness {
    /// Public input values (known to both prover and verifier)
    pub public_inputs: Vec<BabyBear>,
    /// Private variable values (known only to prover)
    pub private_variables: Vec<BabyBear>,
}

impl Witness {
    /// Creates a new witness from public inputs and private variables.
    pub fn new(
        public_inputs: Vec<BabyBear>,
        private_variables: Vec<BabyBear>,
    ) -> SparseResult<Self> {
        Ok(Witness {
            public_inputs,
            private_variables,
        })
    }

    /// Creates a new witness from a single vector.
    ///
    /// # Arguments
    /// * `all_variables` - Vector containing all variables [public || private]
    /// * `num_public_inputs` - Number of public inputs at start of vector
    pub fn from_vec(all_variables: Vec<BabyBear>, num_public_inputs: usize) -> Self {
        let (public_inputs, private_variables) = all_variables.split_at(num_public_inputs);
        Witness {
            public_inputs: public_inputs.to_vec(),
            private_variables: private_variables.to_vec(),
        }
    }

    /// Returns the full witness vector as MLE.
    pub fn to_mle(&self) -> MLE<BabyBear> {
        let mut full_witness = Vec::with_capacity(self.len());
        full_witness.extend_from_slice(&self.public_inputs);
        full_witness.extend_from_slice(&self.private_variables);
        MLE::new(full_witness)
    }

    /// Returns the length of the full witness vector.
    pub fn len(&self) -> usize {
        self.public_inputs.len() + self.private_variables.len()
    }

    /// Returns true if the witness is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the value at the specified index in the full witness.
    pub fn get(&self, index: usize) -> Option<&BabyBear> {
        if index < self.public_inputs.len() {
            self.public_inputs.get(index)
        } else {
            self.private_variables.get(index - self.public_inputs.len())
        }
    }

    /// Creates a simple test witness for development.
    pub fn test_witness() -> Self {
        Witness {
            public_inputs: vec![BabyBear::new(2)], // x = 2
            private_variables: vec![
                BabyBear::new(3), // y = 3
                BabyBear::new(6), // z = 6
            ],
        }
    }
}

/// A complete R1CS instance with a specific constraint system and witness.
///
/// Combines an R1CS constraint system with a specific witness to form a complete
/// instance that can be verified or used in proof generation.
#[derive(Debug, Clone, PartialEq)]
pub struct R1CSInstance {
    /// The constraint system
    pub r1cs: R1CS,
    /// The witness satisfying the constraints
    pub witness: Witness,
}

impl R1CSInstance {
    /// Creates a new R1CS instance.
    pub fn new(r1cs: R1CS, witness: Witness) -> SparseResult<Self> {
        // Validate witness length matches constraint system
        if witness.len() != r1cs.num_variables {
            return Err(SparseError::DimensionMismatch {
                expected: (r1cs.num_variables, witness.len()),
                actual: (r1cs.num_variables, witness.len()),
            });
        }

        Ok(R1CSInstance { r1cs, witness })
    }

    /// Creates a simple test instance.
    pub fn simple_test() -> SparseResult<Self> {
        let (r1cs, witness) = R1CS::simple_test_instance()?;
        Ok(R1CSInstance { r1cs, witness })
    }

    /// Creates a multi-constraint test instance.
    pub fn multi_constraint_test() -> SparseResult<Self> {
        let (r1cs, witness) = R1CS::multi_constraint_test_instance()?;
        Ok(R1CSInstance { r1cs, witness })
    }

    /// Verifies that the witness satisfies all constraints.
    pub fn verify(&self) -> SparseResult<bool> {
        let z = self.witness.to_mle();
        self.r1cs.verify(&z)
    }

    /// Returns the full witness vector as MLE.
    pub fn witness_mle(&self) -> MLE<BabyBear> {
        self.witness.to_mle()
    }

    /// Returns the number of constraints in this instance.
    pub fn num_constraints(&self) -> usize {
        self.r1cs.num_constraints
    }

    /// Returns the number of variables in this instance.
    pub fn num_variables(&self) -> usize {
        self.r1cs.num_variables
    }

    /// Returns the number of public inputs.
    pub fn num_public_inputs(&self) -> usize {
        self.r1cs.num_public_inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use std::collections::HashMap;

    #[test]
    fn test_r1cs_new_valid() {
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        let mut c_coeffs = HashMap::new();

        // Use consistent dimensions - 8 columns (next power of 2 after 3)
        a_coeffs.insert((0, 0), BabyBear::ONE);
        b_coeffs.insert((0, 1), BabyBear::ONE);
        c_coeffs.insert((0, 2), BabyBear::ONE);

        // Add more entries to ensure consistent dimensions
        a_coeffs.insert((1, 3), BabyBear::ONE);
        b_coeffs.insert((1, 4), BabyBear::ONE);
        c_coeffs.insert((1, 5), BabyBear::ONE);

        a_coeffs.insert((2, 6), BabyBear::ONE);
        b_coeffs.insert((2, 7), BabyBear::ONE);
        c_coeffs.insert((2, 0), BabyBear::ONE);

        let a = SparseMLE::new(a_coeffs).unwrap();
        let b = SparseMLE::new(b_coeffs).unwrap();
        let c = SparseMLE::new(c_coeffs).unwrap();

        let r1cs = R1CS::new(a, b, c, 1);
        assert!(r1cs.is_ok());
    }

    #[test]
    fn test_r1cs_new_dimension_mismatch() {
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        let mut c_coeffs = HashMap::new();

        a_coeffs.insert((0, 0), BabyBear::ONE);
        b_coeffs.insert((0, 1), BabyBear::ONE);
        c_coeffs.insert((1, 2), BabyBear::ONE); // Different row count

        let a = SparseMLE::new(a_coeffs).unwrap();
        let b = SparseMLE::new(b_coeffs).unwrap();
        let c = SparseMLE::new(c_coeffs).unwrap();

        let r1cs = R1CS::new(a, b, c, 1);
        assert!(matches!(r1cs, Err(SparseError::ValidationError(_))));
    }

    #[test]
    fn test_witness_new() {
        let witness = Witness::new(
            vec![BabyBear::new(1), BabyBear::new(2)],
            vec![BabyBear::new(3), BabyBear::new(4)],
        );
        assert!(witness.is_ok());
        assert_eq!(witness.unwrap().len(), 4);
    }

    #[test]
    fn test_witness_to_mle() {
        let witness = Witness {
            public_inputs: vec![BabyBear::new(1), BabyBear::new(2)],
            private_variables: vec![BabyBear::new(3), BabyBear::new(4)],
        };

        let mle = witness.to_mle();
        assert_eq!(mle.len(), 4);
        assert_eq!(mle.coeffs()[0], BabyBear::new(1));
        assert_eq!(mle.coeffs()[1], BabyBear::new(2));
        assert_eq!(mle.coeffs()[2], BabyBear::new(3));
        assert_eq!(mle.coeffs()[3], BabyBear::new(4));
    }

    #[test]
    fn test_r1cs_simple_test_instance() {
        let (r1cs, witness) = R1CS::simple_test_instance().unwrap();

        assert_eq!(r1cs.num_constraints, 1);
        assert_eq!(r1cs.num_variables, 8); // Power of 2
        assert_eq!(r1cs.num_public_inputs, 1);

        assert_eq!(witness.public_inputs.len(), 1);
        assert_eq!(witness.private_variables.len(), 7); // 8 total - 1 public

        // Verify the instance
        let instance = R1CSInstance::new(r1cs, witness).unwrap();
        assert!(instance.verify().unwrap());
    }

    #[test]
    fn test_r1cs_multi_constraint_test_instance() {
        let (r1cs, witness) = R1CS::multi_constraint_test_instance().unwrap();

        assert_eq!(r1cs.num_constraints, 4); // 3 constraints padded to next power of 2
        assert_eq!(r1cs.num_variables, 8); // Power of 2
        assert_eq!(r1cs.num_public_inputs, 4);

        assert_eq!(witness.public_inputs.len(), 4);
        assert_eq!(witness.private_variables.len(), 4); // 4 private variables (4 total - 4 public inputs)

        // Verify the instance
        let instance = R1CSInstance::new(r1cs, witness).unwrap();
        assert!(instance.verify().unwrap());
    }

    #[test]
    fn test_r1cs_verify_invalid_witness() {
        let mut a_coeffs = HashMap::new();
        let mut b_coeffs = HashMap::new();
        let mut c_coeffs = HashMap::new();

        // Create matrices with consistent dimensions (8 columns)
        for matrix in [&mut a_coeffs, &mut b_coeffs, &mut c_coeffs] {
            for col in 0..8 {
                matrix.insert((0, col), BabyBear::new(0));
            }
        }

        // Set up the constraint: x * y = z
        a_coeffs.insert((0, 0), BabyBear::ONE); // x
        b_coeffs.insert((0, 1), BabyBear::ONE); // y
        c_coeffs.insert((0, 2), BabyBear::ONE); // z

        let a = SparseMLE::new(a_coeffs).unwrap();
        let b = SparseMLE::new(b_coeffs).unwrap();
        let c = SparseMLE::new(c_coeffs).unwrap();

        let r1cs = R1CS::new(a, b, c, 1).unwrap();

        // Create invalid witness: x=2, y=3, z=7 (should be 6)
        let mut witness_vars = vec![BabyBear::new(0); 8];
        witness_vars[0] = BabyBear::new(2); // x = 2
        witness_vars[1] = BabyBear::new(3); // y = 3
        witness_vars[2] = BabyBear::new(7); // z = 7 (should be 6)

        let z = MLE::new(witness_vars);
        assert!(!r1cs.verify(&z).unwrap());
    }

    #[test]
    fn test_r1cs_instance_creation() {
        let (r1cs, witness) = R1CS::simple_test_instance().unwrap();
        let instance = R1CSInstance::new(r1cs.clone(), witness.clone());
        assert!(instance.is_ok());

        // Test dimension mismatch
        let bad_witness = Witness {
            public_inputs: vec![BabyBear::ONE],
            private_variables: vec![BabyBear::ONE], // Too short
        };
        let bad_instance = R1CSInstance::new(r1cs, bad_witness);
        assert!(matches!(
            bad_instance,
            Err(SparseError::DimensionMismatch { .. })
        ));
    }
}
