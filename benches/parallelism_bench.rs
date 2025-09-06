use helix::Fp;
use p3_field::PrimeCharacteristicRing;
use std::thread::{self, available_parallelism, scope, Builder};

fn benchmark_unsafe_unchecked(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    // Debug assertions to verify our bounds proof in development
    debug_assert!(
        data.len() >= iterations * 2,
        "Bounds failed: data.len()={} < iterations*2={}",
        data.len(),
        iterations * 2
    );
    debug_assert!(
        eq_data.len() >= iterations,
        "Bounds proof failed: eq_data.len()={} < iterations={}",
        eq_data.len(),
        iterations
    );
    let n_threads = available_parallelism()
        .expect("Failed to get number of threads")
        .get();
    debug_assert!(
        iterations >= n_threads,
        "Iterations must be greater than or equal to the number of threads"
    );
    let mut result = Fp::ZERO;
    for i in 0..iterations {
        unsafe {
            // SAFETY: We're sure that:
            // - i < iterations ≤ eq_data.len(), so i is in bounds for eq_data
            // - (i << 1) = i * 2 < iterations * 2 ≤ data.len(), so (i << 1) is in bounds for data
            result += *eq_data.get_unchecked(i) * *data.get_unchecked(i << 1);
        }
    }
    result
}
