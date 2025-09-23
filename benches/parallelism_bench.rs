use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use helix::Fp;
use itertools::Itertools;
use p3_baby_bear::PackedBabyBearNeon;
use p3_field::{PackedValue, PrimeCharacteristicRing};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};
use std::{
    cell::UnsafeCell,
    cmp::min,
    sync::mpsc,
    thread::{ScopedJoinHandle, available_parallelism, scope},
};

fn benchmark_std_parallelism(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
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

    let res = scope(|s| {
        let mut handles: Vec<ScopedJoinHandle<'_, PackedBabyBearNeon>> = Vec::new();
        let chunk_size = iterations / (n_threads * PackedBabyBearNeon::WIDTH);
        let packed_eq_slice = PackedBabyBearNeon::pack_slice(&eq_data);
        let packed_data_slice = PackedBabyBearNeon::pack_slice(&data);
        for thread_chunk in 0..n_threads {
            let start = thread_chunk * chunk_size;
            let end = min(iterations / PackedBabyBearNeon::WIDTH, start + chunk_size);
            let data_slice = &packed_data_slice[start..end];
            let eq_slice = &packed_eq_slice[start..end];
            let handle = s.spawn(move || {
                data_slice
                    .iter()
                    .zip(eq_slice.iter())
                    .map(|(&d, &e)| d * e)
                    .sum()
            });
            handles.push(handle);
        }

        // Collect results from all threads
        let mut total_result = PackedBabyBearNeon::ZERO;
        for handle in handles {
            total_result += handle.join().unwrap();
        }

        total_result.0.into_iter().sum()
    });

    res
}
fn benchmark_rayon_parallelism(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
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

    let data_pack = PackedBabyBearNeon::pack_slice(data);
    let eq_pack = PackedBabyBearNeon::pack_slice(eq_data);
    let res: PackedBabyBearNeon = data_pack
        .par_iter()
        .zip(eq_pack.par_iter())
        .map(|(&d, &e)| d * e)
        .sum();
    res.0.into_iter().sum()
}

fn create_test_data(size: usize) -> (Vec<Fp>, Vec<Fp>) {
    let poly_data: Vec<Fp> = (0..size).map(|i| Fp::from_u32((i as u32) + 1)).collect();
    let eq_data: Vec<Fp> = (0..size / 2)
        .map(|i| Fp::from_u32((i * 3 + 7) as u32))
        .collect();
    (poly_data, eq_data)
}

fn bench_parallelism_comparison(c: &mut Criterion) {
    let sizes = vec![
        (1 << 16, "64K"),  // ~64k elements
        (1 << 18, "256K"), // ~256k elements
        (1 << 20, "1M"),   // ~1 million elements
        (1 << 22, "4M"),   // ~4 million elements
        (1 << 24, "16M"),  // ~16 million elements
    ];

    for (size, size_name) in sizes {
        let iterations = size / 2; // We're accessing even indices, so half the iterations
        let (poly_data, eq_data) = create_test_data(size);

        let mut group = c.benchmark_group(format!("parallelism_comparison_{}", size_name));

        group.bench_with_input(
            BenchmarkId::new("std_threads", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| b.iter(|| benchmark_std_parallelism(poly, eq, *iter_count)),
        );

        group.bench_with_input(
            BenchmarkId::new("rayon", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| {
                b.iter(|| benchmark_rayon_parallelism(poly, eq, *iter_count))
            },
        );

        group.finish();
    }
}

criterion_group!(benches, bench_parallelism_comparison);
criterion_main!(benches);
