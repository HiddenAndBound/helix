use criterion::{Criterion, criterion_group, criterion_main};
use helix::Fp;
use p3_field::PrimeCharacteristicRing;
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};

fn benchmark_fft_forward(c: &mut Criterion) {
    c.bench_function("fft_forward", |b| {
        let mut vector = vec![Fp::from_u32(50); 1 << 11];
        let roots = Fp::roots_of_unity_table(1 << 5);

        b.iter(|| {
            vector
                .par_chunks_mut(1 << 5)
                .for_each(|chunk| Fp::forward_fft(chunk, &roots))
        });
    });
}

criterion_group!(benches, benchmark_fft_forward);
criterion_main!(benches);
