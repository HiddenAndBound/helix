//! Parallel four-step NTT over the BabyBear field.
//!
//! This module exposes [`ParallelFourStepNtt`], an implementation of the natural-ordering
//! four-step Number-Theoretic Transform specialised to the BabyBear field (`Fp`). The design
//! follows the repo's implementation plan (`docs/ntt_plan`) and keeps the public API free of
//! bit-reversed layouts while leveraging Rayon for large inputs.

use std::sync::Mutex;

use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use rayon::prelude::*;

use crate::utils::Fp;

/// Default length threshold above which Rayon-based parallelism is enabled.
const DEFAULT_PARALLEL_THRESHOLD: usize = 1 << 12;

/// Parallel four-step NTT specialised for BabyBear field elements.
#[derive(Debug)]
pub struct ParallelFourStepNtt {
    len: usize,
    n1: usize,
    n2: usize,
    twiddles: Vec<Fp>,
    column_twiddles: Vec<Fp>,
    row_twiddles: Vec<Fp>,
    scratch: Mutex<Vec<Fp>>,
    parallel_threshold: usize,
}

impl ParallelFourStepNtt {
    /// Construct a new transform for vectors of the given power-of-two length.
    pub fn new(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "four-step NTT length must be a power of two"
        );
        assert!(len > 0, "four-step NTT length must be non-zero");

        let log_n = len.trailing_zeros();
        let mut n1 = 1usize << (log_n / 2);
        let mut n2 = len / n1;
        if n1 < n2 {
            n1 <<= 1;
            n2 >>= 1;
        }

        let omega = Fp::two_adic_generator(log_n as usize);
        let column_root = omega.exp_u64(n2 as u64);
        let row_root = omega.exp_u64(n1 as u64);

        let column_twiddles = compute_powers(n1, column_root);
        let row_twiddles = compute_powers(n2, row_root);
        let twiddles = compute_twiddle_grid(n1, n2, omega);

        let scratch = Mutex::new(vec![Fp::ZERO; len]);

        Self {
            len,
            n1,
            n2,
            twiddles,
            column_twiddles,
            row_twiddles,
            scratch,
            parallel_threshold: DEFAULT_PARALLEL_THRESHOLD,
        }
    }

    /// Update the minimum length above which Rayon parallelism is used.
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Total length covered by this transform.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Forward NTT in place, producing natural-order outputs.
    pub fn forward_in_place(&self, values: &mut [Fp]) {
        let input_len = values.len();
        assert!(
            input_len == self.len,
            "input length {input_len} does not match configured NTT length {}",
            self.len
        );

        if self.len <= 1 {
            return;
        }

        let mut scratch = self.scratch.lock().expect("transpose scratch poisoned");
        if scratch.len() < self.len {
            scratch.resize(self.len, Fp::ZERO);
        }

        // Stage 0: reshape into column-major order for contiguous column FFTs.
        for c in 0..self.n2 {
            for r in 0..self.n1 {
                scratch[c * self.n1 + r] = values[r * self.n2 + c];
            }
        }

        // Stage 1: column FFTs of size n1.
        if self.len >= self.parallel_threshold {
            scratch
                .par_chunks_mut(self.n1)
                .for_each(|column| radix2_ntt_in_place(column, &self.column_twiddles));
        } else {
            for column in scratch.chunks_mut(self.n1) {
                radix2_ntt_in_place(column, &self.column_twiddles);
            }
        }

        // Stage 2: twiddle multiplication.
        for c in 0..self.n2 {
            for r in 0..self.n1 {
                let idx = c * self.n1 + r;
                let twiddle_idx = r * self.n2 + c;
                scratch[idx] *= self.twiddles[twiddle_idx];
            }
        }

        // Stage 3 setup: transpose back to row-major layout.
        for r in 0..self.n1 {
            for c in 0..self.n2 {
                values[r * self.n2 + c] = scratch[c * self.n1 + r];
            }
        }

        // Stage 3: row FFTs of size n2.
        if self.len >= self.parallel_threshold {
            values
                .par_chunks_mut(self.n2)
                .for_each(|row| radix2_ntt_in_place(row, &self.row_twiddles));
        } else {
            for row in values.chunks_mut(self.n2) {
                radix2_ntt_in_place(row, &self.row_twiddles);
            }
        }
    }

    /// Borrowing convenience that clones the input before running the transform.
    pub fn forward(&self, input: &[Fp]) -> Vec<Fp> {
        let mut buffer = input.to_vec();
        self.forward_in_place(&mut buffer);
        buffer
    }
}

/// Compute the table of consecutive powers of `base`, starting at 1, of length `size`.
fn compute_powers(size: usize, base: Fp) -> Vec<Fp> {
    let mut acc = Vec::with_capacity(size);
    let mut current = Fp::ONE;
    for _ in 0..size {
        acc.push(current);
        current *= base;
    }
    acc
}

/// Compute stage-two twiddle factors arranged in column-major order.
fn compute_twiddle_grid(n1: usize, n2: usize, omega: Fp) -> Vec<Fp> {
    let mut grid = Vec::with_capacity(n1 * n2);
    let mut omega_i = Fp::ONE;
    for _i in 0..n1 {
        let mut current = Fp::ONE;
        for _j in 0..n2 {
            grid.push(current);
            current *= omega_i;
        }
        omega_i *= omega;
    }

    grid
}

/// Iterative radix-2 decimation-in-time NTT specialised for BabyBear scalars.
fn radix2_ntt_in_place(values: &mut [Fp], twiddles: &[Fp]) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    debug_assert!(n.is_power_of_two());
    bit_reverse_permute(values);

    let mut m = 1;
    let stages = n.trailing_zeros() as usize;
    for _stage in 0..stages {
        let step = n / (2 * m);
        for k in (0..n).step_by(2 * m) {
            for j in 0..m {
                let twiddle = twiddles[j * step];
                let t = twiddle * values[k + j + m];
                let u = values[k + j];
                values[k + j] = u + t;
                values[k + j + m] = u - t;
            }
        }
        m *= 2;
    }
}

fn bit_reverse_permute(values: &mut [Fp]) {
    let n = values.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits_usize(i, bits);
        if j > i {
            values.swap(i, j);
        }
    }
}

fn reverse_bits_usize(mut value: usize, bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use p3_dft::TwoAdicSubgroupDft;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_monty_31::dft::RecursiveDft;

    const TEST_SEED: u64 = 0xDEADBEEF;

    #[test]
    fn forward_matches_recursive_dft_small() {
        let mut rng = StdRng::seed_from_u64(TEST_SEED);

        for &log_len in &[4usize, 6, 8] {
            let len = 1usize << log_len;
            let ntt = ParallelFourStepNtt::new(len);

            let mut input = Vec::with_capacity(len);
            for _ in 0..len {
                input.push(Fp::from_u32(rng.next_u32()));
            }

            let naive = naive_dft(&input);
            let mut ours = input.clone();
            ntt.forward_in_place(&mut ours);

            let dft = RecursiveDft::new(len);
            let matrix = RowMajorMatrix::new(input.clone(), 1);
            let reference = dft.dft_batch(matrix);
            let expected = reference.to_row_major_matrix().values;

            assert_eq!(
                expected, naive,
                "recursive DFT reference diverges from naive computation for length {len}"
            );
            if ours != expected {
                let mut bitrev = ours.clone();
                bit_reverse_in_place(&mut bitrev);
                assert_eq!(
                    bitrev, expected,
                    "bit-reversed output does not match expected for length {len}"
                );
            }
            assert_eq!(ours, expected, "length {len}");
        }
    }

    #[test]
    fn forward_matches_babybear_fft() {
        let mut rng = StdRng::seed_from_u64(TEST_SEED);

        for &log_len in &[4usize, 8, 12] {
            let len = 1usize << log_len;
            let ntt = ParallelFourStepNtt::new(len);

            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(Fp::from_u32(rng.next_u32()));
            }

            let naive = naive_dft(&values);
            let mut ours = values.clone();
            ntt.forward_in_place(&mut ours);

            let mut reference = values;
            let roots = Fp::roots_of_unity_table(len);
            Fp::forward_fft(&mut reference, &roots);
            bit_reverse_in_place(&mut reference);

            assert_eq!(
                reference, naive,
                "forward_fft reference diverges from naive computation for length {len}"
            );
            assert_eq!(ours, reference, "length {len}");
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        let _ = ParallelFourStepNtt::new(12);
    }

    #[test]
    fn radix2_kernel_matches_naive() {
        let len = 8usize;
        let omega = Fp::two_adic_generator(len.trailing_zeros() as usize);
        let twiddles = compute_powers(len, omega);
        let mut data: Vec<Fp> = (0..len).map(|i| Fp::from_u32(i as u32)).collect();
        let naive = naive_dft(&data);

        radix2_ntt_in_place(&mut data, &twiddles);

        assert_eq!(data, naive);
    }

    #[test]
    fn stage3_input_matches_theory() {
        let len = 16usize;
        let ntt = ParallelFourStepNtt::new(len);
        let mut input = Vec::with_capacity(len);
        for i in 0..len {
            input.push(Fp::from_u32(i as u32));
        }

        let stage_input = stage3_input_snapshot(&ntt, &input);
        let theoretical = theoretical_stage3_input(&input, ntt.n1, ntt.n2);

        assert_eq!(stage_input, theoretical);
    }

    fn bit_reverse_in_place(values: &mut [Fp]) {
        super::bit_reverse_permute(values);
    }

    fn stage3_input_snapshot(ntt: &ParallelFourStepNtt, input: &[Fp]) -> Vec<Fp> {
        let mut scratch = vec![Fp::ZERO; ntt.len];
        for c in 0..ntt.n2 {
            for r in 0..ntt.n1 {
                scratch[c * ntt.n1 + r] = input[r * ntt.n2 + c];
            }
        }

        for column in scratch.chunks_mut(ntt.n1) {
            radix2_ntt_in_place(column, &ntt.column_twiddles);
        }

        for c in 0..ntt.n2 {
            for r in 0..ntt.n1 {
                let idx = c * ntt.n1 + r;
                let twiddle_idx = r * ntt.n2 + c;
                scratch[idx] *= ntt.twiddles[twiddle_idx];
            }
        }

        let mut stage_input = vec![Fp::ZERO; ntt.len];
        for r in 0..ntt.n1 {
            for c in 0..ntt.n2 {
                stage_input[r * ntt.n2 + c] = scratch[c * ntt.n1 + r];
            }
        }

        stage_input
    }

    fn theoretical_stage3_input(input: &[Fp], n1: usize, n2: usize) -> Vec<Fp> {
        let len = input.len();
        let omega = Fp::two_adic_generator(len.trailing_zeros() as usize);
        let column_root = omega.exp_u64(n2 as u64);
        let mut stage = vec![Fp::ZERO; len];

        // Column FFTs (naive) producing Y[r,s]
        for s in 0..n2 {
            for k1 in 0..n1 {
                let mut acc = Fp::ZERO;
                for r in 0..n1 {
                    let twiddle = column_root.exp_u64((r * k1) as u64);
                    acc += input[r * n2 + s] * twiddle;
                }
                stage[k1 * n2 + s] = acc;
            }
        }

        // Twiddle multiplication omega^{k1 * s}
        for s in 0..n2 {
            for k1 in 0..n1 {
                let exponent = (k1 * s) as u64;
                stage[k1 * n2 + s] *= omega.exp_u64(exponent);
            }
        }

        stage
    }

    fn naive_dft(input: &[Fp]) -> Vec<Fp> {
        let len = input.len();
        if len == 0 {
            return Vec::new();
        }
        let omega = Fp::two_adic_generator(len.trailing_zeros() as usize);
        let mut output = Vec::with_capacity(len);
        for k in 0..len {
            let mut acc = Fp::ZERO;
            for (t, &value) in input.iter().enumerate() {
                let exponent = (t * k) % len;
                acc += value * omega.exp_u64(exponent as u64);
            }
            output.push(acc);
        }
        output
    }
}
