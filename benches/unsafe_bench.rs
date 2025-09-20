use criterion::{Criterion, criterion_group, criterion_main, BenchmarkId};
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

// New: Unsafe get_unchecked for maximum performance
// SAFETY REQUIREMENTS:
// - Must prove mathematically that i < eq_data.len()
// - Must prove mathematically that (i << 1) < data.len()
// - In sumcheck context: iterations = 1 << (rounds - round - 1)
// - poly.len() = 1 << rounds, so max index = iterations * 2 - 2 < poly.len()
fn benchmark_unsafe_unchecked(data: &[Fp], eq_data: &[Fp], iterations: usize) -> Fp {
    // Debug assertions to verify our bounds proof in development
    debug_assert!(data.len() >= iterations * 2, 
        "Bounds proof failed: data.len()={} < iterations*2={}", data.len(), iterations * 2);
    debug_assert!(eq_data.len() >= iterations,
        "Bounds proof failed: eq_data.len()={} < iterations={}", eq_data.len(), iterations);
    
    let mut result = Fp::ZERO;
    for i in 0..iterations {
        unsafe {
            // SAFETY: We've proven above that:
            // - i < iterations ≤ eq_data.len(), so i is in bounds for eq_data
            // - (i << 1) = i * 2 < iterations * 2 ≤ data.len(), so (i << 1) is in bounds for data
            result += *eq_data.get_unchecked(i) * *data.get_unchecked(i << 1);
        }
    }
    result
}

fn create_test_data(size: usize) -> (Vec<Fp>, Vec<Fp>) {
    let poly_data: Vec<Fp> = (0..size).map(|i| Fp::from_u32(i as u32 + 1)).collect();
    let eq_data: Vec<Fp> = (0..size/2).map(|i| Fp::from_u32((i * 3 + 7) as u32)).collect();
    (poly_data, eq_data)
}

fn bench_unsafe_comparison(c: &mut Criterion) {
    let sizes = vec![
        (1 << 20, "1M"),    // ~1 million elements
        (1 << 21, "2M"),    // ~2 million elements  
        (1 << 22, "4M"),    // ~4 million elements
    ];

    for (size, size_name) in sizes {
        let iterations = size / 2; // We're accessing even indices, so half the iterations
        let (poly_data, eq_data) = create_test_data(size);

        let mut group = c.benchmark_group(format!("unsafe_comparison_{}", size_name));
        
        group.bench_with_input(
            BenchmarkId::new("direct_indexing", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| {
                b.iter(|| {
                    benchmark_direct_indexing(poly, eq, *iter_count)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("unsafe_unchecked", size_name),
            &(&poly_data, &eq_data, iterations),
            |b, (poly, eq, iter_count)| {
                b.iter(|| {
                    benchmark_unsafe_unchecked(poly, eq, *iter_count)
                })
            },
        );

        group.finish();
    }
}

// Realistic sumcheck scenario with 10 iterations
fn bench_sumcheck_realistic_unsafe(c: &mut Criterion) {
    let iterations = 10; // Realistic sumcheck inner loop
    let poly_size = 1 << 16; // 64k elements
    
    let (poly_data, eq_data) = create_test_data(poly_size);

    let mut group = c.benchmark_group("sumcheck_realistic_unsafe");

    group.bench_function("direct_indexing_sumcheck", |b| {
        b.iter(|| {
            benchmark_direct_indexing(&poly_data, &eq_data, iterations)
        })
    });

    group.bench_function("unsafe_unchecked_sumcheck", |b| {
        b.iter(|| {
            benchmark_unsafe_unchecked(&poly_data, &eq_data, iterations)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_unsafe_comparison, bench_sumcheck_realistic_unsafe);
criterion_main!(benches);

// Assembly verification commands:
// cargo asm --rust --bin unsafe_bench benchmark_direct_indexing
// cargo asm --rust --bin unsafe_bench benchmark_unsafe_unchecked
// The unsafe version should show minimal assembly with no bounds checks

/* MATHEMATICAL BOUNDS PROOF for sumcheck usage:
 *
 * In process_sum_check_round:
 * - rounds = eval_point.len() 
 * - iterations = 1 << (rounds - round - 1)
 * - poly.len() = 1 << rounds (by MLE definition)
 * - eq.len() >= iterations (by EqEvals invariant)
 *
 * For i ∈ [0, iterations):
 * - i < iterations ≤ eq.len(), so eq[i] is in bounds ✓
 * - i << 1 = i * 2 < iterations * 2 = 2 * 2^(rounds-round-1) = 2^(rounds-round) ≤ 2^rounds = poly.len()
 *   so poly[i << 1] is in bounds ✓
 *
 * Therefore unsafe access is mathematically proven safe in sumcheck context.
 */