use criterion::{ criterion_group, criterion_main, BenchmarkId, Criterion };
use helix::{ pcs::utils::bit_reverse_sort, Fp };
use p3_dft::{ Radix2Bowers, Radix2Dit, Radix2DitParallel, TwoAdicSubgroupDft };
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{ bitrev, dense::{ DenseMatrix, RowMajorMatrix, RowMajorMatrixViewMut }, Matrix };
use p3_monty_31::dft::RecursiveDft;
use rayon::{ iter::ParallelIterator, slice::ParallelSliceMut, vec };

fn benchmark_fft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    for i in 20..30 {
        group.bench_with_input(format!("FFT 2^{i}"), &i, |b, i| {
            let mut vector = vec![Fp::from_u32(50); 1 << i];
            let matrix = RowMajorMatrix::new(vector, 1 << (i - 1));
            let roots = Fp::roots_of_unity_table(1 << (i - 3));
            let fft = RecursiveDft::new(1 << i);
            b.iter(|| { fft.dft_batch(matrix.clone()) });
        });
    }
}

criterion_group!(benches, benchmark_fft_forward);
criterion_main!(benches);
