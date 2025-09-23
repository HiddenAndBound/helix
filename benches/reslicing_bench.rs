use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use helix::Fp;
use p3_field::PrimeCharacteristicRing;

// Baseline: Direct indexing (current implementation)
fn benchmark_direct_indexing(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    let mut result = Fp::ZERO;
    for i in 0..iterations {
        result += eq_data[i] * data[i << 1]; // Equivalent to data[i * 2]
    }
    result
}

// New: Reslicing technique for guaranteed bounds check elimination
fn benchmark_reslicing(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    // Reslice to make bounds obvious to LLVM
    let data_slice = &data[..iterations * 2]; // Ensure we have enough elements
    let eq_slice = &eq_data[..iterations];

    let mut result = Fp::ZERO;
    for i in 0..iterations {
        result += eq_slice[i] * data_slice[i << 1]; // Now bounds-check free!
    }
    result
}

fn create_test_data(size: usize) -> (Vec<Fp>, Vec<Fp>) {
    let poly_data: Vec<Fp> = (0..size).map(|i| Fp::from_u32(i as u32 + 1)).collect();
    let eq_data: Vec<Fp> = (0..size / 2)
        .map(|i| Fp::from_u32((i * 3 + 7) as u32))
        .collect();
    (poly_data, eq_data)
}

fn bench_reslicing_comparison(c: &mut Criterion) {
    let sizes = vec![
        (1 << 20, "1M"), // ~1 million elements
        (1 << 21, "2M"), // ~2 million elements
        (1 << 22, "4M"), // ~4 million elements
    ];

    for (size, size_name) in sizes {
        let iterations = size / 2; // We're accessing even indices, so half the iterations
        let (poly_data, eq_data) = create_test_data(size);

        let mut group = c.benchmark_group(format!("reslicing_comparison_{}", size_name));

        group.bench_with_input(
            BenchmarkId::new("direct_indexing", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| b.iter(|| benchmark_direct_indexing(poly, eq, *iter_count)),
        );

        group.bench_with_input(
            BenchmarkId::new("reslicing", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| b.iter(|| benchmark_reslicing(poly, eq, *iter_count)),
        );

        group.finish();
    }
}

// Realistic sumcheck scenario with 10 iterations
fn bench_sumcheck_realistic_reslicing(c: &mut Criterion) {
    let iterations = 10; // Realistic sumcheck inner loop
    let poly_size = 1 << 16; // 64k elements

    let (poly_data, eq_data) = create_test_data(poly_size);

    let mut group = c.benchmark_group("sumcheck_realistic_reslicing");

    group.bench_function("direct_indexing_sumcheck", |b| {
        b.iter(|| benchmark_direct_indexing(&poly_data, &eq_data, iterations))
    });

    group.bench_function("reslicing_sumcheck", |b| {
        b.iter(|| benchmark_reslicing(&poly_data, &eq_data, iterations))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reslicing_comparison,
    bench_sumcheck_realistic_reslicing
);
criterion_main!(benches);

// Assembly verification commands:
// cargo asm --rust --bin reslicing_bench benchmark_direct_indexing
// cargo asm --rust --bin reslicing_bench benchmark_reslicing
// The reslicing version should show no bounds checks in the hot loop
