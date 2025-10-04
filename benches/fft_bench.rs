use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helix::Fp;
use helix::pcs::ntt::ParallelFourStepNtt;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_monty_31::dft::RecursiveDft;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn benchmark_fft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    let mut rng = StdRng::seed_from_u64(0xFEED_BEEF);

    for log_len in 18..=25 {
        let len = 1usize << log_len;
        let mut baseline = Vec::with_capacity(len);
        for _ in 0..len {
            baseline.push(Fp::from_u32(rng.next_u32()));
        }

        let ntt = ParallelFourStepNtt::new(len);
        let recursive = RecursiveDft::new(len);

        if cfg!(debug_assertions) {
            let mut ours = baseline.clone();
            ntt.forward_in_place(&mut ours);
            let reference = recursive
                .dft_batch(RowMajorMatrix::new(baseline.clone(), 1))
                .bit_reverse_rows();
            assert_eq!(ours, reference.values, "length {len}");
        }

        group.bench_with_input(BenchmarkId::new("RecursiveDft", len), &len, |b, _| {
            b.iter(|| {
                let matrix = RowMajorMatrix::new(baseline.clone(), 1);
                let result = recursive.dft_batch(matrix);
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("ParallelFourStep", len), &len, |b, _| {
            b.iter(|| {
                let mut data = baseline.clone();
                ntt.forward_in_place(&mut data);
                black_box(data);
            });
        });
    }
}

criterion_group!(benches, benchmark_fft_forward);
criterion_main!(benches);
