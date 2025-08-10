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
    fn test_fingerprints_generate_simple() {
        // Simple test: 2 reads from a 3-entry table
        let indices = vec![Fp::from_usize(0), Fp::from_usize(2)]; // Read from addr 0 and 2
        let values = vec![Fp::from_usize(10), Fp::from_usize(30)]; // Values at those addresses
        let table = vec![
            Fp4::from(Fp::from_usize(10)), // table[0] = 10
            Fp4::from(Fp::from_usize(20)), // table[1] = 20
            Fp4::from(Fp::from_usize(30)), // table[2] = 30
        ];
        let read_ts = vec![Fp::from_usize(1), Fp::from_usize(3)]; // Read timestamps
        let final_ts = vec![
            Fp::from_usize(2), // Final timestamp for addr 0
            Fp::from_usize(0), // Final timestamp for addr 1 (not accessed)
            Fp::from_usize(4), // Final timestamp for addr 2
        ];
        let gamma = Fp4::from(Fp::from_usize(7)); // Random challenge γ
        let tau = Fp4::from(Fp::from_usize(13)); // Random challenge τ

        let fingerprints = Fingerprints::generate(indices, values, table, read_ts, final_ts, gamma, tau);

        // Check that all fingerprint vectors have expected lengths
        assert_eq!(fingerprints.w_init.len(), 3); // Initial state for 3 table entries
        assert_eq!(fingerprints.r.len(), 2); // 2 read operations
        assert_eq!(fingerprints.w.len(), 2); // 2 write operations
        assert_eq!(fingerprints.s.len(), 3); // Final state for 3 table entries

        // Verify some specific fingerprint calculations using h_γ(a,v,t) = a·γ² + v·γ + t - τ
        let gamma_squared = gamma * gamma;
        
        // w_init[0] should be: (0·γ² + 10·γ + 0) - τ = (0·49 + 10·7 + 0) - 13 = 70 - 13 = 57
        let addr_0_fp = Fp::ZERO;
        let value_0_fp4 = Fp4::from(Fp::from_usize(10));
        let expected_w_init_0 = gamma_squared * addr_0_fp + value_0_fp4 * gamma - tau;
        assert_eq!(fingerprints.w_init[0], expected_w_init_0);

        // r[0] should be: (0·γ² + 10·γ + 1) - τ = (0·49 + 10·7 + 1) - 13 = 71 - 13 = 58
        let read_ts_0_fp = Fp::from_usize(1);
        let value_0_fp = Fp::from_usize(10);
        let expected_r_0 = gamma_squared * addr_0_fp + gamma * value_0_fp + read_ts_0_fp - tau;
        assert_eq!(fingerprints.r[0], expected_r_0);

        // w[0] should be: (0·γ² + 10·γ + 2) - τ = (0·49 + 10·7 + 2) - 13 = 72 - 13 = 59
        let write_ts_0_fp = Fp::from_usize(2);
        let expected_w_0 = gamma_squared * addr_0_fp + gamma * value_0_fp + write_ts_0_fp - tau;
        assert_eq!(fingerprints.w[0], expected_w_0);
    }
}
