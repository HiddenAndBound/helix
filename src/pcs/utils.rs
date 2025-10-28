//! BaseFold utility functions for Reed-Solomon encoding, polynomial folding,
//! Merkle tree operations, and cryptographic hashing.

use core::hash;
use std::cell::UnsafeCell;
use std::io::{ IoSlice, Write };
use std::thread::{ self, scope };
use std::{ iter::zip, thread::available_parallelism };
use std::ops::Mul;

use blake3::Hasher;
use itertools::{ multizip, Itertools };
use p3_baby_bear::BabyBear;
use p3_field::{ ExtensionField, Field, PrimeCharacteristicRing, RawDataSerializable };
use p3_monty_31::dft::RecursiveDft;
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    ParallelBridge,
    ParallelIterator,
};
use serde::Serialize;
use sha3::{ Digest, Keccak256 };
use tracing::instrument;

use crate::pcs::BaseFoldConfig;
use crate::{ merkle_tree::{ HashOutput, MerklePath, MerkleTree }, polynomial::MLE, Fp, Fp4 };

/// Cryptographic commitment as 32-byte Blake3 hash.
pub type Commitment = HashOutput;

/// Reed-Solomon encoded polynomial.
pub type Encoding = Vec<Fp>;

/// Number of random queries for soundness (≈2⁻¹⁰⁰ security).
pub const QUERIES: usize = 144;

/// Reed-Solomon encoding rate.
pub const RATE: usize = 2;

/// Precomputed constant 1/2 in BabyBear field.
pub const HALF: Fp = Fp::new(1006632961);

/// Encodes a multilinear polynomial using Reed-Solomon codes via forward FFT.
/// Zero-pads coefficients to 2 * poly.len(), applies FFT, and bit-reverses result.
#[instrument(level = "debug", skip_all)]
pub fn encode_mle(poly: &MLE<Fp>, roots: &[Vec<Fp>], rate: usize) -> Encoding {
    let mut buffer = vec![Fp::ZERO; poly.len() * rate];
    buffer[0..poly.len()].copy_from_slice(poly.coeffs());

    assert_eq!(
        roots.len(),
        (buffer.len().trailing_zeros() as usize) - (rate.trailing_zeros() as usize),
        "Root table not large enough to encode the MLE."
    );

    BabyBear::forward_fft(&mut buffer, roots);
    bit_reverse_sort(&mut buffer);
    buffer
}

/// Encodes a multilinear polynomial using Reed-Solomon codes via forward FFT.
/// Zero-pads coefficients to 2 * poly.len(), applies FFT, and bit-reverses result.
pub fn encode_mle_ext(poly: &MLE<Fp4>, rate: usize) -> Vec<Fp4> {
    use p3_dft::TwoAdicSubgroupDft;
    let mut buffer = vec![Fp4::ZERO; poly.len() * rate];
    buffer[0..poly.len()].copy_from_slice(poly.coeffs());
    let fft: RecursiveDft<Fp> = RecursiveDft::new(buffer.len());

    let buffer = fft.dft_algebra(buffer);
    buffer
}

pub fn decode_mle_ext(encoding: Vec<Fp4>, rate: usize) -> Vec<Fp4> {
    use p3_dft::TwoAdicSubgroupDft;
    let fft: RecursiveDft<Fp> = RecursiveDft::new(encoding.len() / rate);

    let buffer = fft.idft_algebra(encoding);
    buffer[..buffer.len() / rate].to_vec()
}

/// Folds a pair of codewords using challenge and twiddle factor.
/// Computes g₀ = (a₀ + a₁)/2, g₁ = (a₀ - a₁)/2 * ω⁻¹, returns g₀ + r*(g₁ - g₀).
#[inline(always)]
pub fn fold_pair<F>(codewords: (F, F), r: Fp4, twiddle: Fp) -> Fp4
    where F: Field + Mul<Fp, Output = F>, Fp4: ExtensionField<F>
{
    let (a0, a1) = codewords;
    let (g0, g1) = ((a0 + a1).halve(), (a0 - a1).halve() * twiddle);
    r * (g1 - g0) + g0
}

/// Folds an encoding by applying fold_pair to adjacent pairs, reducing size by half.
/// Uses slice splitting to eliminate manual offset calculation and prevent out-of-bounds access.
/// Applies fold_pair to pairs from left and right halves of the input slice.
#[tracing::instrument(level = "debug", skip_all)]
pub fn fold<F>(code: &[F], random_challenge: Fp4, roots: &[Fp]) -> Vec<Fp4>
    where F: Field + Mul<Fp, Output = F>, Fp4: ExtensionField<F>
{
    let half_size = code.len() >> 1;
    assert_eq!(roots.len(), half_size, "roots length must equal half of code length");

    let powers: Vec<_> = roots[1].inverse().powers().take(half_size).collect();

    let (left, right) = code.split_at(half_size);
    (left, right, &powers)
        .into_par_iter()
        .map(|(&l, &r, &root)| fold_pair((l, r), random_challenge, root))
        .collect()
}

/// Retrieves codeword pairs from an encoding at query positions.
/// Uses slice splitting to eliminate manual offset calculation.
pub fn get_codewords<F: Into<Fp4> + Copy>(queries: &[usize], encoding: &[F]) -> Vec<(Fp4, Fp4)> {
    let halfsize = encoding.len() >> 1;
    let (left, right) = encoding.split_at(halfsize);

    queries
        .iter()
        .copied()
        .map(|i| (left[i].into(), right[i].into()))
        .collect()
}

/// Generates Merkle authentication paths for query positions.
/// Assumes indices are valid leaf indices.
pub fn get_merkle_paths(queries: &[usize], merkle_tree: &MerkleTree) -> Vec<MerklePath> {
    queries
        .iter()
        .copied()
        .map(|i| merkle_tree.get_path(i))
        .collect()
}

/// Computes Blake3 hash of a field element pair.
/// Converts elements to Fp4, serializes to bytes, and hashes with Blake3.
pub fn hash_field_pair<T>(left: T, right: T) -> HashOutput
    where T: Copy + Sync + Send, Fp4: RawDataSerializable + From<T>
{
    let mut buffer = [0u8; 32];
    fill_buf(left, right, &mut buffer);
    blake3::hash(&buffer).into()
}

/// Creates hash leaves from field element pairs.
#[tracing::instrument(level = "debug", skip_all)]
pub fn create_hash_leaves<T>(data: &[T]) -> Vec<HashOutput>
    where T: Copy + Sync + Send, Fp4: RawDataSerializable + From<T>
{
    let (left, right) = data.split_at(data.len() / 2);
    left.par_iter()
        .zip(right.par_iter())
        .map(|(&l, &r)| hash_field_pair(l, r))
        .collect()
}

/// Creates hash leaves from field element pairs.
#[tracing::instrument(level = "debug", skip_all)]
pub fn create_hash_leaves_std<T>(data: &[T]) -> Vec<HashOutput>
    where T: Copy + Sync + Send, Fp4: RawDataSerializable + From<T>
{
    let (left, right) = data.split_at(data.len() / 2);
    let mut leaves = vec![[0;32]; data.len()/2];
    let threads = available_parallelism().unwrap();
    let chunk_size = leaves.len().div_ceil(threads.get());
    let mut leaf_chunks = leaves.chunks_mut(chunk_size);

    scope(|s| {
        for thread in 0..threads.into() {
            // Creating multiple mutable pointers, this is ok
            let leaves_ref = leaf_chunks.next().unwrap();
            let left_ref = left.chunks(chunk_size).nth(thread).unwrap();
            let right_ref = right.chunks(chunk_size).nth(thread).unwrap();

            s.spawn(move || {
                let mut hasher = blake3::Hasher::new();
                let mut buffer = [0u8; 32];
                for (parent, (&left, &right)) in leaves_ref
                    .iter_mut()
                    .zip(zip(left_ref.iter(), right_ref.iter())) {
                    fill_buf(left, right, &mut buffer);
                    *parent = hasher.update(&buffer).finalize().into();
                    hasher.reset();
                }
            });
        }
    });
    leaves
}

#[inline(always)]
pub fn fill_buf<T>(left: T, right: T, buf: &mut [u8; 32])
    where T: Copy + Sync + Send, Fp4: RawDataSerializable + From<T>
{
    Fp4::from(left)
        .into_bytes()
        .into_iter()
        .zip(&mut buf[0..16])
        .for_each(|(byte, buf)| {
            *buf = byte;
        });

    Fp4::from(right)
        .into_bytes()
        .into_iter()
        .zip(&mut buf[16..32])
        .for_each(|(byte, buf)| {
            *buf = byte;
        });
}

/// Creates hash leaves from field element pairs.
#[tracing::instrument(level = "debug", skip_all)]
pub fn create_hash_leaves_skip<T>(data: &[T], config: &BaseFoldConfig) -> Vec<HashOutput>
    where T: Copy + Sync + Send + RawDataSerializable
{
    let rounds_skipped = config.round_skip;
    let partitions = 2 << rounds_skipped;
    let partition_size = data.len() / partitions;
    let data_chunks = data.chunks(partition_size);
    let mut leaves = vec![[0;32]; partition_size];
    let threads = available_parallelism().unwrap();
    let chunk_size = leaves.len().div_ceil(threads.get());
    let mut leaf_chunks = leaves.chunks_mut(chunk_size);

    scope(|s| {
        for thread in 0..threads.get() {
            let buffer_size = size_of::<T>() * partitions;
            let chunk = leaf_chunks.next().unwrap();

            s.spawn(move || {
                let mut hasher = Hasher::new();
                let mut buffer = vec![0u8; buffer_size.clone()];
                for i in 0..chunk.len() {
                    for j in 0..partitions {
                        buffer.chunks_mut(size_of::<T>()).for_each(|buf|
                            data[i + j * partition_size]
                                .into_bytes()
                                .into_iter()
                                .zip(buf)
                                .for_each(|(b, byte)| {
                                    *byte = b;
                                })
                        );
                    }
                    chunk[i] = hasher.update(&buffer).finalize().into();
                    hasher.reset();
                }
            });
        }
    });

    leaves
}

/// Performs in-place bit-reverse sort on a vector.
/// Swaps element at position i with element at bit_reverse(i).
/// Vector length must be a power of 2.
pub fn bit_reverse_sort<T>(vec: &mut [T]) {
    let len = vec.len();
    assert!(len == 0 || len.is_power_of_two(), "Vector length must be a power of 2, got {}", len);

    if len <= 1 {
        return;
    }

    let num_bits = len.trailing_zeros();

    for i in 0..len {
        // Use the highly-optimized builtin reverse_bits which typically maps to a single
        // CPU instruction on supported targets. This provides a large speedup over a
        // manual loop while remaining portable and stable.
        let j = (i.reverse_bits() >> (usize::BITS - num_bits)) as usize;
        if i < j {
            vec.swap(i, j);
        }
    }
}

/// Returns a new vector with elements in bit-reversed order.
/// Creates new vector placing element at position i into position bit_reverse(i).
/// Vector length must be a power of 2.
pub fn bit_reverse_sorted<T: Clone>(vec: &Vec<T>) -> Vec<T> {
    let len = vec.len();
    assert!(len == 0 || len.is_power_of_two(), "Vector length must be a power of 2, got {}", len);

    if len <= 1 {
        return vec.clone();
    }

    let num_bits = len.trailing_zeros();
    let mut result = vec![vec[0].clone(); len];

    for i in 0..len {
        // Fast reverse via builtin instruction
        let j = (i.reverse_bits() >> (usize::BITS - num_bits)) as usize;
        result[j] = vec[i].clone();
    }

    result
}

#[cfg(test)]
mod tests {
    use std::default;

    use p3_matrix::{ Matrix, dense::RowMajorMatrix };
    use p3_monty_31::dft::RecursiveDft;
    use rand::{ RngCore, SeedableRng, rngs::StdRng };

    use super::*;

    #[test]
    fn test_bit_reverse_sort_basic() {
        let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse_sort(&mut vec);

        // For length 8 (3 bits), bit-reverse mapping:
        // 0 (000) -> 0 (000), 1 (001) -> 4 (100), 2 (010) -> 2 (010), 3 (011) -> 6 (110)
        // 4 (100) -> 1 (001), 5 (101) -> 5 (101), 6 (110) -> 3 (011), 7 (111) -> 7 (111)
        assert_eq!(vec, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn test_bit_reverse_sort_length_4() {
        let mut vec = vec!['a', 'b', 'c', 'd'];
        bit_reverse_sort(&mut vec);

        // For length 4 (2 bits), bit-reverse mapping:
        // 0 (00) -> 0 (00), 1 (01) -> 2 (10), 2 (10) -> 1 (01), 3 (11) -> 3 (11)
        assert_eq!(vec, vec!['a', 'c', 'b', 'd']);
    }

    #[test]
    fn test_bit_reverse_sort_length_2() {
        let mut vec = vec![10, 20];
        bit_reverse_sort(&mut vec);

        // For length 2 (1 bit), bit-reverse mapping:
        // 0 (0) -> 0 (0), 1 (1) -> 1 (1)
        assert_eq!(vec, vec![10, 20]);
    }

    #[test]
    fn test_bit_reverse_sort_length_1() {
        let mut vec = vec![42];
        bit_reverse_sort(&mut vec);
        assert_eq!(vec, vec![42]);
    }

    #[test]
    fn test_bit_reverse_sort_empty() {
        let mut vec: Vec<i32> = vec![];
        bit_reverse_sort(&mut vec);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_bit_reverse_sort_large() {
        let mut vec: Vec<usize> = (0..16).collect();
        bit_reverse_sort(&mut vec);

        // For length 16 (4 bits), verify a few key mappings:
        // 0 (0000) -> 0 (0000), 1 (0001) -> 8 (1000), 2 (0010) -> 4 (0100), 3 (0011) -> 12 (1100)
        assert_eq!(vec[0], 0); // 0 -> 0
        assert_eq!(vec[8], 1); // 1 -> 8
        assert_eq!(vec[4], 2); // 2 -> 4
        assert_eq!(vec[12], 3); // 3 -> 12

        // Check that all original elements are present
        let mut sorted_vec = vec.clone();
        sorted_vec.sort();
        assert_eq!(sorted_vec, (0..16).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "Vector length must be a power of 2")]
    fn test_bit_reverse_sort_non_power_of_two() {
        let mut vec = vec![1, 2, 3]; // length 3 is not a power of 2
        bit_reverse_sort(&mut vec);
    }

    #[test]
    fn test_bit_reverse_sorted_basic() {
        let vec = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let result = bit_reverse_sorted(&vec);

        // Should produce same result as in-place version
        assert_eq!(result, vec![0, 4, 2, 6, 1, 5, 3, 7]);

        // Original should be unchanged
        assert_eq!(vec, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_bit_reverse_sorted_strings() {
        let vec = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let result = bit_reverse_sorted(&vec);

        assert_eq!(
            result,
            vec!["a".to_string(), "c".to_string(), "b".to_string(), "d".to_string()]
        );
        assert_eq!(vec, vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]);
    }

    #[test]
    fn test_bit_reverse_sorted_field_elements() {
        let vec: Vec<Fp> = (0..8).map(|i| Fp::from_u32(i)).collect();
        let result = bit_reverse_sorted(&vec);

        let expected: Vec<Fp> = vec![0, 4, 2, 6, 1, 5, 3, 7]
            .into_iter()
            .map(|i| Fp::from_u32(i))
            .collect();

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Vector length must be a power of 2")]
    fn test_bit_reverse_sorted_non_power_of_two() {
        let vec = vec![1, 2, 3, 4, 5]; // length 5 is not a power of 2
        bit_reverse_sorted(&vec);
    }

    #[test]
    fn test_bit_reverse_inverse_property() {
        // Test that applying bit-reverse twice returns to original order
        let original = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut vec = original.clone();

        // Apply bit-reverse twice
        bit_reverse_sort(&mut vec);
        bit_reverse_sort(&mut vec);

        assert_eq!(vec, original);
    }

    #[test]
    fn test_bit_reverse_consistency() {
        // Test that in-place and copying versions produce same result
        let original = vec![100, 200, 300, 400, 500, 600, 700, 800];

        let mut in_place = original.clone();
        bit_reverse_sort(&mut in_place);

        let copied = bit_reverse_sorted(&original);

        assert_eq!(in_place, copied);
    }

    #[test]
    fn test_bit_reverse_with_complex_types() {
        #[derive(Debug, Clone, PartialEq)]
        struct ComplexData {
            id: usize,
            name: String,
        }

        let vec = vec![
            ComplexData {
                id: 0,
                name: "zero".to_string(),
            },
            ComplexData {
                id: 1,
                name: "one".to_string(),
            },
            ComplexData {
                id: 2,
                name: "two".to_string(),
            },
            ComplexData {
                id: 3,
                name: "three".to_string(),
            }
        ];

        let result = bit_reverse_sorted(&vec);

        // Expected order: [0, 2, 1, 3]
        assert_eq!(result[0].id, 0);
        assert_eq!(result[1].id, 2);
        assert_eq!(result[2].id, 1);
        assert_eq!(result[3].id, 3);
    }

    #[test]
    fn fft_test() {}
}
