use crate::{Fp, Fp4, spartan::spark::gkr::GKRProof};
use p3_field::PrimeCharacteristicRing;

//Offline Memory Check
pub struct OfflineMemoryCheck {
    fingerprints: Vec<Fingerprints>,
    product_trees: Vec<ProductTree>,
    gpa_proofs: Vec<GKRProof>,
}

pub struct Fingerprints {
    w_init: Vec<Fp4>,
    w: Vec<Fp4>,
    r: Vec<Fp4>,
    s: Vec<Fp4>,
}

pub struct ProductTree {
    /// Left halves for each layer (root to leaves)
    /// layer_left[0] is root layer, layer_left[depth-1] is leaf layer
    pub layer_left: Vec<Vec<Fp4>>,
    /// Right halves for each layer (root to leaves)
    pub layer_right: Vec<Vec<Fp4>>,
    /// Tree depth (log₂ of input size)
    pub depth: usize,
    /// Original input size (power of 2)
    pub input_size: usize,
    /// Final product value (root of tree)
    pub root_value: Fp4,
}

impl ProductTree {
    pub fn new(layer_left: Vec<Vec<Fp4>>, layer_right: Vec<Vec<Fp4>>) -> Self {
        let depth = layer_left.len();
        let input_size = if depth > 0 {
            layer_left[depth - 1].len() * 2
        } else {
            0
        };
        let root_value = if depth > 0 && !layer_left[0].is_empty() {
            layer_left[0][0] * layer_right[0][0]
        } else {
            Fp4::ZERO
        };

        Self {
            layer_left,
            layer_right,
            depth,
            input_size,
            root_value,
        }
    }

    /// Generates two product trees from fingerprint data
    /// Returns (left_tree, right_tree) where:
    /// - left_tree has leaves as concatenation of w ∪ r
    /// - right_tree has leaves as concatenation of w_init ∪ s
    pub fn generate(fingerprints: &Fingerprints) -> (Self, Self) {
        // Build left and right leaf vectors using copy
        let mut left_leaves = Vec::with_capacity(fingerprints.w.len() + fingerprints.r.len());
        left_leaves.extend_from_slice(&fingerprints.w);
        left_leaves.extend_from_slice(&fingerprints.r);

        let mut right_leaves = Vec::with_capacity(fingerprints.w_init.len() + fingerprints.s.len());
        right_leaves.extend_from_slice(&fingerprints.w_init);
        right_leaves.extend_from_slice(&fingerprints.s);

        // Compute actual products before padding
        let left_actual_product = left_leaves.iter().fold(Fp4::ONE, |acc, &x| acc * x);
        let right_actual_product = right_leaves.iter().fold(Fp4::ONE, |acc, &x| acc * x);

        // Ensure both have the same power-of-2 size
        let max_size = left_leaves.len().max(right_leaves.len());
        let padded_size = max_size.next_power_of_two();

        left_leaves.resize(padded_size, Fp4::ONE);
        right_leaves.resize(padded_size, Fp4::ONE);

        // Build both trees inline
        fn build_tree(mut leaves: Vec<Fp4>, actual_product: Fp4) -> ProductTree {
            let input_size = leaves.len();
            let depth = (input_size as f64).log2() as usize;
            let mut layer_left = Vec::with_capacity(depth);
            let mut layer_right = Vec::with_capacity(depth);

            // Build tree from leaves up to root using for loop
            for level in (0..depth).rev() {
                let level_size = 1 << level; // 2^level
                let mut left_half = vec![Fp4::ZERO; level_size];
                let mut right_half = vec![Fp4::ZERO; level_size];
                let mut next_level = vec![Fp4::ZERO; level_size];

                for i in 0..level_size {
                    let left = leaves[2 * i];
                    let right = leaves[2 * i + 1];

                    left_half[i] = left;
                    right_half[i] = right;
                    next_level[i] = left * right;
                }

                layer_left.push(left_half);
                layer_right.push(right_half);

                if level > 0 {
                    leaves = next_level;
                }
            }

            ProductTree {
                layer_left,
                layer_right,
                depth,
                input_size,
                root_value: actual_product,
            }
        }

        let left_tree = build_tree(left_leaves, left_actual_product);
        let right_tree = build_tree(right_leaves, right_actual_product);

        (left_tree, right_tree)
    }

    /// Returns the root value (final product)
    pub fn root_value(&self) -> Fp4 {
        self.root_value
    }

    /// Returns the tree depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn get_final_layer_claims(&self) -> (Fp4, Fp4) {
        (
            self.layer_left.last().expect("Will be non-empty")[0],
            self.layer_right.last().expect("Will be non-empty")[0],
        )
    }

    /// Returns the left layer data at the specified depth
    pub fn get_layer_left(&self, depth: usize) -> &Vec<Fp4> {
        &self.layer_left[depth]
    }

    /// Returns the right layer data at the specified depth
    pub fn get_layer_right(&self, depth: usize) -> &Vec<Fp4> {
        &self.layer_right[depth]
    }
}

impl Fingerprints {
    pub fn generate(
        indices: &[Fp],
        values: &[Fp],
        table: &[Fp4],
        read_ts: &[Fp],
        final_ts: &[Fp],
        gamma: Fp4,
        tau: Fp4,
    ) -> Self {
        // Basic validation - all read operation vectors should have same length
        assert_eq!(indices.len(), values.len());
        assert_eq!(indices.len(), read_ts.len());

        let n_reads = indices.len();
        let table_size = table.len();

        // w_init and s: Initial and final memory fingerprints (same size: table_size)
        let mut w_init = vec![Fp4::ZERO; table_size];
        let mut s = vec![Fp4::ZERO; final_ts.len()];
        let gamma_squared = gamma * gamma;
        for (addr, &value) in table.iter().enumerate() {
            let addr_fp = Fp::from_usize(addr);
            // w_init: (addr, value, t=0): h_γ(addr,value,0) = addr·γ² + value·γ + 0
            w_init[addr] = gamma_squared * addr_fp + value * gamma - tau;
            // s: (addr, final_value, final_ts): h_γ(addr,value,final_ts) = addr·γ² + value·γ + final_ts
            // Assume final value is same as initial value from table for read-only memory
            s[addr] = gamma_squared * addr_fp + value * gamma + final_ts[addr] - tau;
        }

        // r and w: Read and write operation fingerprints (same size: n_reads)
        let mut r = vec![Fp4::ZERO; n_reads];
        let mut w = vec![Fp4::ZERO; n_reads];
        for i in 0..n_reads {
            let write_ts = read_ts[i] + Fp::ONE;
            // r: (addr, value, read_ts): h_γ(addr,value,read_ts) = addr·γ² + value·γ + read_ts
            r[i] = gamma_squared * indices[i] + gamma * values[i] + read_ts[i] - tau;
            // w: (addr, value, write_ts): h_γ(addr,value,write_ts) = addr·γ² + value·γ + (read_ts + 1)
            w[i] = gamma_squared * indices[i] + gamma * values[i] + write_ts - tau;
        }

        Self { w_init, w, r, s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_memory_consistency_property() {
        use crate::spartan::spark::sparse::TimeStamps;

        // Step 1: Create memory table and access pattern
        let memory_table = vec![
            Fp4::from(Fp::from_usize(10)), // table[0] = 10
            Fp4::from(Fp::from_usize(20)), // table[1] = 20
            Fp4::from(Fp::from_usize(30)), // table[2] = 30
            Fp4::from(Fp::from_usize(40)), // table[3] = 40
        ];

        // Memory access pattern: read from addresses [0, 1, 0, 2]
        let read_addresses_fp = vec![
            Fp::from_usize(0), // Access address 0 first time
            Fp::from_usize(1), // Access address 1 first time
            Fp::from_usize(0), // Access address 0 second time
            Fp::from_usize(2), // Access address 2 first time
        ];
        let read_values = vec![
            Fp::from_usize(10), // Value at address 0
            Fp::from_usize(20), // Value at address 1
            Fp::from_usize(10), // Value at address 0 (unchanged)
            Fp::from_usize(30), // Value at address 2
        ];

        // Step 2: Use TimeStamps::compute to generate correct timestamps
        let max_address_space = 4; // Power of 2 containing addresses 0,1,2,3
        let timestamps = TimeStamps::compute(&read_addresses_fp, max_address_space).unwrap();

        // Extract read_ts and final_ts from TimeStamps using getter methods
        let read_timestamps: Vec<Fp> = timestamps
            .read_ts()
            .coeffs()
            .iter()
            .take(read_addresses_fp.len())
            .cloned()
            .collect();
        let final_timestamps: &[Fp] = timestamps.final_ts().coeffs();

        // Step 3: Set up challenges
        let gamma = Fp4::from(Fp::from_usize(7));
        let tau = Fp4::from(Fp::from_usize(11));

        // Step 4: Generate all four fingerprint multisets using correct timestamps
        let fingerprints = Fingerprints::generate(
            &read_addresses_fp,
            &read_values,
            &memory_table,
            &read_timestamps,
            &final_timestamps,
            gamma,
            tau,
        );

        // Step 5: Compute products of the multisets
        // Memory consistency property: ∏(I ∪ W) = ∏(R ∪ F)

        // Product of w_init ∪ w (initial state ∪ writes)
        let mut product_initial_and_writes = Fp4::ONE;
        for &val in &fingerprints.w_init {
            product_initial_and_writes *= val;
        }
        for &val in &fingerprints.w {
            product_initial_and_writes *= val;
        }

        // Product of r ∪ s (reads ∪ final state)
        let mut product_reads_and_final = Fp4::ONE;
        for &val in &fingerprints.r {
            product_reads_and_final *= val;
        }
        for &val in &fingerprints.s {
            product_reads_and_final *= val;
        }

        // Step 6: Test the core memory consistency property
        // This should hold for any valid memory trace with correct timestamps
        assert_eq!(
            product_initial_and_writes, product_reads_and_final,
            "Memory consistency failed: ∏(w_init ∪ w) ≠ ∏(r ∪ s)"
        );

        // Step 7: Verify the fingerprint vectors have correct sizes
        assert_eq!(fingerprints.w_init.len(), 4); // 4 memory locations
        assert_eq!(fingerprints.r.len(), 4); // 4 read operations
        assert_eq!(fingerprints.w.len(), 4); // 4 write operations
        assert_eq!(fingerprints.s.len(), 4); // 4 memory locations (final state)
    }

    #[test]
    fn test_larger_memory_trace() {
        use crate::spartan::spark::sparse::TimeStamps;

        // Step 1: Create a larger memory trace with multiple operations
        // Memory table: 8 locations with initial values
        let memory_table = vec![
            Fp4::from(Fp::from_usize(100)), // table[0] = 100
            Fp4::from(Fp::from_usize(200)), // table[1] = 200
            Fp4::from(Fp::from_usize(300)), // table[2] = 300
            Fp4::from(Fp::from_usize(400)), // table[3] = 400
            Fp4::from(Fp::from_usize(500)), // table[4] = 500
            Fp4::from(Fp::from_usize(600)), // table[5] = 600
            Fp4::from(Fp::from_usize(700)), // table[6] = 700
            Fp4::from(Fp::from_usize(800)), // table[7] = 800
        ];

        // Complex memory access pattern: [0, 1, 0, 2, 3, 1, 4, 0]
        let read_addresses_fp = vec![
            Fp::from_usize(0), // Access address 0 first time
            Fp::from_usize(1), // Access address 1 first time
            Fp::from_usize(0), // Access address 0 second time
            Fp::from_usize(2), // Access address 2 first time
            Fp::from_usize(3), // Access address 3 first time
            Fp::from_usize(1), // Access address 1 second time
            Fp::from_usize(4), // Access address 4 first time
            Fp::from_usize(0), // Access address 0 third time
        ];
        let read_values = vec![
            Fp::from_usize(100), // Value at address 0
            Fp::from_usize(200), // Value at address 1
            Fp::from_usize(100), // Value at address 0 (unchanged)
            Fp::from_usize(300), // Value at address 2
            Fp::from_usize(400), // Value at address 3
            Fp::from_usize(200), // Value at address 1 (unchanged)
            Fp::from_usize(500), // Value at address 4
            Fp::from_usize(100), // Value at address 0 (unchanged)
        ];

        // Step 2: Use TimeStamps::compute to generate correct timestamps
        let max_address_space = 8; // Power of 2 containing all addresses
        let timestamps = TimeStamps::compute(&read_addresses_fp, max_address_space).unwrap();

        // Extract read_ts and final_ts from TimeStamps using getter methods
        let read_timestamps: Vec<Fp> = timestamps
            .read_ts()
            .coeffs()
            .iter()
            .take(read_addresses_fp.len())
            .cloned()
            .collect();
        let final_timestamps: &[Fp] = timestamps.final_ts().coeffs();

        // Step 3: Set up challenges
        let gamma = Fp4::from(Fp::from_usize(13));
        let tau = Fp4::from(Fp::from_usize(17));

        // Step 4: Generate all four fingerprint multisets using correct timestamps
        let fingerprints = Fingerprints::generate(
            &read_addresses_fp,
            &read_values,
            &memory_table,
            &read_timestamps,
            &final_timestamps,
            gamma,
            tau,
        );

        // Step 5: Compute products of the multisets
        let mut product_initial_and_writes = Fp4::ONE;
        for &val in &fingerprints.w_init {
            product_initial_and_writes *= val;
        }
        for &val in &fingerprints.w {
            product_initial_and_writes *= val;
        }

        let mut product_reads_and_final = Fp4::ONE;
        for &val in &fingerprints.r {
            product_reads_and_final *= val;
        }
        for &val in &fingerprints.s {
            product_reads_and_final *= val;
        }

        // Step 6: Test the core memory consistency property
        assert_eq!(
            product_initial_and_writes, product_reads_and_final,
            "Memory consistency failed: ∏(w_init ∪ w) ≠ ∏(r ∪ s)"
        );

        // Step 7: Verify the fingerprint vectors have correct sizes
        assert_eq!(fingerprints.w_init.len(), 8); // 8 memory locations
        assert_eq!(fingerprints.r.len(), 8); // 8 read operations
        assert_eq!(fingerprints.w.len(), 8); // 8 write operations
        assert_eq!(fingerprints.s.len(), 8); // 8 memory locations (final state)
    }

    #[test]
    fn test_product_tree_generation() {
        use crate::spartan::spark::sparse::TimeStamps;

        // Use the same test data as our working memory consistency test
        let memory_table = vec![
            Fp4::from(Fp::from_usize(10)), // table[0] = 10
            Fp4::from(Fp::from_usize(20)), // table[1] = 20
            Fp4::from(Fp::from_usize(30)), // table[2] = 30
            Fp4::from(Fp::from_usize(40)), // table[3] = 40
        ];

        // Memory access pattern: read from addresses [0, 1, 0, 2]
        let read_addresses_fp = vec![
            Fp::from_usize(0), // Access address 0 first time
            Fp::from_usize(1), // Access address 1 first time
            Fp::from_usize(0), // Access address 0 second time
            Fp::from_usize(2), // Access address 2 first time
        ];
        let read_values = vec![
            Fp::from_usize(10), // Value at address 0
            Fp::from_usize(20), // Value at address 1
            Fp::from_usize(10), // Value at address 0 (unchanged)
            Fp::from_usize(30), // Value at address 2
        ];

        // Use TimeStamps::compute to generate correct timestamps
        let max_address_space = 4; // Power of 2 containing addresses 0,1,2,3
        let timestamps = TimeStamps::compute(&read_addresses_fp, max_address_space).unwrap();

        // Extract read_ts and final_ts from TimeStamps using getter methods
        let read_timestamps: Vec<Fp> = timestamps
            .read_ts()
            .coeffs()
            .iter()
            .take(read_addresses_fp.len())
            .cloned()
            .collect();
        let final_timestamps: &[Fp] = timestamps.final_ts().coeffs();

        let gamma = Fp4::from(Fp::from_usize(7));
        let tau = Fp4::from(Fp::from_usize(11));

        // Step 2: Generate fingerprints using the same parameters as the working test
        let fingerprints = Fingerprints::generate(
            &read_addresses_fp,
            &read_values,
            &memory_table,
            &read_timestamps,
            &final_timestamps,
            gamma,
            tau,
        );

        // Step 3: Generate product trees
        let (left_tree, right_tree) = ProductTree::generate(&fingerprints);

        // Step 4: Verify that root values represent the products we computed before
        // Manual product calculation for verification
        let mut manual_left_product = Fp4::ONE;
        for &val in &fingerprints.w {
            manual_left_product *= val;
        }
        for &val in &fingerprints.r {
            manual_left_product *= val;
        }

        let mut manual_right_product = Fp4::ONE;
        for &val in &fingerprints.w_init {
            manual_right_product *= val;
        }
        for &val in &fingerprints.s {
            manual_right_product *= val;
        }

        // Step 5: Verify tree properties
        assert_eq!(
            left_tree.root_value(),
            manual_left_product,
            "Left tree root should match manual product of w ∪ r"
        );
        assert_eq!(
            right_tree.root_value(),
            manual_right_product,
            "Right tree root should match manual product of w_init ∪ s"
        );

        // Note: For this test case, the products should NOT be equal since this is not from a memory-consistent trace
        // The memory consistency property would hold only for valid memory traces

        // Step 6: Verify tree structure properties
        assert_eq!(
            left_tree.depth(),
            right_tree.depth(),
            "Both trees should have same depth"
        );
        assert_eq!(
            left_tree.input_size(),
            right_tree.input_size(),
            "Both trees should have same input size"
        );
        assert!(
            left_tree.input_size().is_power_of_two(),
            "Input size should be power of 2"
        );

        // Step 7: Verify tree dimensions
        let expected_leaves = (fingerprints.w.len() + fingerprints.r.len())
            .max(fingerprints.w_init.len() + fingerprints.s.len());
        let expected_size = expected_leaves.next_power_of_two();
        assert_eq!(left_tree.input_size(), expected_size);
        assert_eq!(right_tree.input_size(), expected_size);
    }
}
