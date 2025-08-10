use crate::{Fp, Fp4};
use p3_field::PrimeCharacteristicRing;

//Offline Memory Check
pub struct OfflineMemoryCheck {
    fingerprints: Vec<Fingerprints>,
    product_trees: Vec<ProductTree>,
}

pub struct Fingerprints {
    w_init: Vec<Fp4>,
    w: Vec<Fp4>,
    r: Vec<Fp4>,
    s: Vec<Fp4>,
}

pub struct ProductTree {
    layer_left: Vec<Vec<Fp4>>,
    layer_right: Vec<Vec<Fp4>>,
}

impl ProductTree {
    pub fn new(layer_left: Vec<Vec<Fp4>>, layer_right: Vec<Vec<Fp4>>) -> Self {
        Self {
            layer_left,
            layer_right,
        }
    }
}

impl Fingerprints {
    pub fn generate(
        indices: Vec<Fp>,
        values: Vec<Fp>,
        table: Vec<Fp4>,
        read_ts: Vec<Fp>,
        final_ts: Vec<Fp>,
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
        let read_timestamps: Vec<Fp> = timestamps.read_ts().iter().take(read_addresses_fp.len()).cloned().collect();
        let final_timestamps: Vec<Fp> = timestamps.final_ts().to_vec();
        
        // Step 3: Set up challenges
        let gamma = Fp4::from(Fp::from_usize(7));
        let tau = Fp4::from(Fp::from_usize(11));

        // Step 4: Generate all four fingerprint multisets using correct timestamps
        let fingerprints = Fingerprints::generate(
            read_addresses_fp,
            read_values,
            memory_table,
            read_timestamps, 
            final_timestamps,
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
        assert_eq!(product_initial_and_writes, product_reads_and_final,
                   "Memory consistency failed: ∏(w_init ∪ w) ≠ ∏(r ∪ s)");
        
        // Step 7: Verify the fingerprint vectors have correct sizes
        assert_eq!(fingerprints.w_init.len(), 4); // 4 memory locations
        assert_eq!(fingerprints.r.len(), 4);      // 4 read operations
        assert_eq!(fingerprints.w.len(), 4);      // 4 write operations
        assert_eq!(fingerprints.s.len(), 4);      // 4 memory locations (final state)
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
        let read_timestamps: Vec<Fp> = timestamps.read_ts().iter().take(read_addresses_fp.len()).cloned().collect();
        let final_timestamps: Vec<Fp> = timestamps.final_ts().to_vec();
        
        // Step 3: Set up challenges 
        let gamma = Fp4::from(Fp::from_usize(13));
        let tau = Fp4::from(Fp::from_usize(17));

        // Step 4: Generate all four fingerprint multisets using correct timestamps
        let fingerprints = Fingerprints::generate(
            read_addresses_fp,
            read_values, 
            memory_table,
            read_timestamps,
            final_timestamps,
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
        assert_eq!(product_initial_and_writes, product_reads_and_final,
                   "Memory consistency failed: ∏(w_init ∪ w) ≠ ∏(r ∪ s)");
        
        // Step 7: Verify the fingerprint vectors have correct sizes
        assert_eq!(fingerprints.w_init.len(), 8); // 8 memory locations
        assert_eq!(fingerprints.r.len(), 8);      // 8 read operations
        assert_eq!(fingerprints.w.len(), 8);      // 8 write operations  
        assert_eq!(fingerprints.s.len(), 8);      // 8 memory locations (final state)
    }
}
