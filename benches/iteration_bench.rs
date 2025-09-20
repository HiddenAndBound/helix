use criterion::{ Criterion, criterion_group, criterion_main, BenchmarkId };
use helix::Fp;
use p3_field::PrimeCharacteristicRing;

// Simulate the exact computation from process_sum_check_round
fn benchmark_direct_indexing(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    let mut result = Fp::ZERO;
    for i in 0..iterations {
        result += eq_data[i] * data[i << 1]; // Equivalent to data[i * 2]
    }
    result
}

fn benchmark_step_by(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    let mut result = Fp::ZERO;
    data.iter()
        .step_by(2)
        .zip(eq_data.iter())
        .map(|(&d, &e)| { d * e })
        .sum()
}

fn benchmark_chunks_exact(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    let mut result = Fp::ZERO;
    data.chunks(2)
        .zip(eq_data.iter())
        .map(|(d, &e)| { d[0] * e })
        .sum()
}

fn create_test_data(size: usize) -> (Vec<Fp>, Vec<Fp>) {
    let poly_data: Vec<Fp> = (0..size).map(|i| Fp::from_u32((i as u32) + 1)).collect();
    let eq_data: Vec<Fp> = (0..size / 2).map(|i| Fp::from_u32((i * 3 + 7) as u32)).collect();
    (poly_data, eq_data)
}

fn bench_iteration_methods(c: &mut Criterion) {
    let sizes = vec![
        (1 << 20, "1M"), // ~1 million elements
        (1 << 21, "2M"), // ~2 million elements
        (1 << 22, "4M") // ~4 million elements
    ];

    for (size, size_name) in sizes {
        let iterations = size / 2; // We're accessing even indices, so half the iterations
        let (poly_data, eq_data) = create_test_data(size);

        let mut group = c.benchmark_group(format!("iteration_comparison_{}", size_name));

        group.bench_with_input(
            BenchmarkId::new("direct_indexing", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| {
                b.iter(|| { benchmark_direct_indexing(poly, eq, *iter_count) })
            }
        );

        group.bench_with_input(
            BenchmarkId::new("step_by", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| { b.iter(|| { benchmark_step_by(poly, eq, *iter_count) }) }
        );

        group.bench_with_input(
            BenchmarkId::new("chunks_exact", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| {
                b.iter(|| { benchmark_chunks_exact(poly, eq, *iter_count) })
            }
        );

        group.finish();
    }
}

// Additional benchmark for the specific sumcheck scenario
fn bench_sumcheck_realistic(c: &mut Criterion) {
    // Typical parameters for sumcheck rounds
    let rounds = 20;
    let round = 10; // Mid-round scenario
    let iterations = 1 << (rounds - round - 1); // 2^9 = 512 iterations
    let poly_size = 1 << rounds; // 2^20 = ~1M elements

    let (poly_data, eq_data) = create_test_data(poly_size);

    let mut group = c.benchmark_group("sumcheck_realistic");

    group.bench_function("direct_indexing_sumcheck", |b| {
        b.iter(|| benchmark_direct_indexing(&poly_data, &eq_data, iterations))
    });

    group.bench_function("step_by_sumcheck", |b| {
        b.iter(|| benchmark_step_by(&poly_data, &eq_data, iterations))
    });

    group.bench_function("chunks_exact_sumcheck", |b| {
        b.iter(|| benchmark_chunks_exact(&poly_data, &eq_data, iterations))
    });

    group.finish();
}

criterion_group!(benches, bench_iteration_methods, bench_sumcheck_realistic);
criterion_main!(benches);
