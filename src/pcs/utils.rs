//! BaseFold-style utility helpers (encoding, folding, hashing, Merkle).

use blake3::Hasher;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::PrimeCharacteristicRing;
use p3_field::{ExtensionField, Field, RawDataSerializable};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_monty_31::dft::RecursiveDft;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::Fp;
use crate::Fp4;
use crate::merkle::{HashOutput, MerklePath, MerkleTree};
use crate::pcs::BaseFoldConfig;
use crate::poly::MLE;

/// Cryptographic commitment as 32-byte Blake3 hash.
pub type Commitment = HashOutput;

/// Reed-Solomon encoded polynomial.
pub type Encoding = Vec<Fp>;

#[inline(always)]
pub fn fill_buf<T>(left: T, right: T, buf: &mut [u8; 32])
where
    T: Copy + Sync + Send,
    Fp4: RawDataSerializable + From<T>,
{
    Fp4::from(left)
        .into_bytes()
        .into_iter()
        .zip(&mut buf[0..16])
        .for_each(|(byte, dst)| *dst = byte);

    Fp4::from(right)
        .into_bytes()
        .into_iter()
        .zip(&mut buf[16..32])
        .for_each(|(byte, dst)| *dst = byte);
}

/// Computes Blake3 hash of a field element pair.
pub fn hash_field_pair<T>(left: T, right: T) -> HashOutput
where
    T: Copy + Sync + Send,
    Fp4: RawDataSerializable + From<T>,
{
    let mut buffer = [0u8; 32];
    fill_buf(left, right, &mut buffer);
    blake3::hash(&buffer).into()
}

/// Creates hash leaves from field element pairs (split into left/right halves).
pub fn create_hash_leaves_std<T>(data: &[T]) -> Vec<HashOutput>
where
    T: Copy + Sync + Send,
    Fp4: RawDataSerializable + From<T>,
{
    let (left, right) = data.split_at(data.len() / 2);
    left.par_iter()
        .zip(right.par_iter())
        .map(|(&l, &r)| hash_field_pair(l, r))
        .collect()
}

/// Creates hash leaves while skipping `config.round_skip` folding rounds.
pub fn create_hash_leaves_skip<T>(data: &[T], config: &BaseFoldConfig) -> Vec<HashOutput>
where
    T: ExtensionField<Fp> + RawDataSerializable + Copy + Sync + Send,
{
    let rounds_skipped = config.round_skip;
    let partitions = 1 << (rounds_skipped + 1);
    println!("rounds skipped {rounds_skipped}");
    println!("partitions {partitions}");

    let partition_size = data.len() / partitions;
    let matrix = RowMajorMatrixView::new(data, partition_size);
    let transpose = matrix.transpose();

    transpose
        .par_row_chunks(1024)
        .flat_map(|chunk| {
            let mut hasher = Hasher::new();
            let mut buffer = vec![0; T::NUM_BYTES * partitions];
            chunk
                .rows()
                .into_iter()
                .map(|row| {
                    hasher.reset();
                    T::into_byte_stream(row)
                        .into_iter()
                        .zip(buffer.iter_mut())
                        .for_each(|(byte, dst)| *dst = byte);
                    hasher.update(&buffer).finalize().into()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Encodes an MLE via a recursive DFT implementation.
pub fn encode_mle_rec(poly: &MLE<Fp>, dft: &RecursiveDft<Fp>, _rate: usize) -> Encoding {
    let mut buffer = poly.coeffs().to_vec();
    buffer.extend(std::iter::repeat(Fp::ZERO).take(poly.len()));
    dft.dft(buffer)
}

pub fn decode_mle_ext(encoding: Vec<Fp4>, rate: usize) -> Vec<Fp4> {
    let fft: RecursiveDft<Fp> = RecursiveDft::new(encoding.len() / rate);
    let buffer = fft.idft_algebra(encoding);
    buffer[..buffer.len() / rate].to_vec()
}

/// Folds a pair of codewords using challenge and twiddle factor.
#[inline(always)]
pub fn fold_pair<F>(codewords: (F, F), r: Fp4, twiddle: Fp) -> Fp4
where
    F: Field,
    F: std::ops::Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let (a0, a1) = codewords;
    let (g0, g1) = ((a0 + a1).halve(), (a0 - a1).halve() * twiddle);
    r * (g1 - g0) + g0
}

/// Folds an encoding by applying `fold_pair` to adjacent pairs, reducing size by half.
pub fn fold<F>(code: &[F], random_challenge: Fp4, roots: &[Fp]) -> Vec<Fp4>
where
    F: Field + std::ops::Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let half_size = code.len() >> 1;
    assert_eq!(
        roots.len(),
        half_size,
        "roots length must equal half of code length"
    );

    let powers: Vec<_> = roots[1].inverse().powers().take(half_size).collect();
    let (left, right) = code.split_at(half_size);

    (left, right, &powers)
        .into_par_iter()
        .map(|(&l, &r, &root)| fold_pair((l, r), random_challenge, root))
        .collect()
}

/// Generates Merkle authentication paths for query positions.
pub fn get_merkle_paths(
    queries: &[usize],
    merkle_tree: &MerkleTree,
) -> anyhow::Result<Vec<MerklePath>> {
    queries
        .iter()
        .copied()
        .map(|i| merkle_tree.get_path(i))
        .collect()
}
