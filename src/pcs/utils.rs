use std::iter::zip;
use std::ops::Mul;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, RawDataSerializable};

use crate::{
    Fp, Fp4,
    merkle_tree::{MerklePath, MerkleTree},
    polynomial::MLE,
};

pub type Commitment = [u8; 32];
pub type Encoding = Vec<Fp>;

pub const QUERIES: usize = 144;
pub const RATE: usize = 2;
pub const HALF: Fp = Fp::new(134217727);

pub fn encode_mle(poly: &MLE<Fp>, roots: &[Vec<Fp>]) -> Encoding {
    let mut buffer = vec![Fp::ZERO; poly.len() * RATE];
    buffer[0..poly.len()].copy_from_slice(poly.coeffs());

    assert!(
        roots.len() > buffer.len().trailing_zeros() as usize,
        "Root table not large enough to encode the MLE."
    );

    BabyBear::forward_fft(&mut buffer, roots);
    buffer
}

pub fn fold_pair<F>(codewords: (F, F), r: Fp4, twiddle: Fp) -> Fp4
where
    F: Field + Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let (a0, a1) = codewords;
    let (g0, g1) = ((a0 - a1) * HALF, (a0 + a1) * HALF * twiddle);
    r * (g0 + g1) + g0
}

pub fn fold<F>(mle: &[F], random_challenge: Fp4, roots: &[Fp]) -> Vec<Fp4>
where
    F: Field + Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let half_size = mle.len() >> 1;
    let mut folded = vec![Fp4::ZERO; half_size];
    for i in 0..half_size {
        folded[i] = fold_pair((mle[i], mle[i + half_size]), random_challenge, roots[i])
    }

    folded
}

pub fn get_codewords<F: Into<Fp4> + Copy>(queries: &[usize], encoding: &[F]) -> Vec<(Fp4, Fp4)> {
    let halfsize = encoding.len() / 2;
    queries
        .iter()
        .copied()
        .map(|i| {
            if i > halfsize {
                (encoding[i ^ halfsize].into(), encoding[i].into())
            } else {
                (encoding[i].into(), encoding[i ^ halfsize].into())
            }
        })
        .collect()
}

//We assume that codewords to be folded are hashed together.
pub fn get_merkle_paths(queries: &[usize], merkle_tree: &MerkleTree) -> Vec<MerklePath> {
    let halfsize = 1 << merkle_tree.depth;
    queries
        .iter()
        .copied()
        .map(|i| {
            // As pairs that differ in their most dominant bit are hashed together, we only index leaf nodes by the trailing bits.
            // Thus in the case an index is above the halfsize, meaning its leading bit is 1, we toggle that bit to get the valid and appropriate index.
            if i > halfsize {
                merkle_tree.get_path(i ^ halfsize)
            } else {
                merkle_tree.get_path(i)
            }
        })
        .collect()
}

/// Hash a pair of field elements by converting them to Fp4, serializing to bytes, and hashing with Blake3.
pub fn hash_field_pair<T>(left: T, right: T) -> [u8; 32]
where
    Fp4: RawDataSerializable + From<T>,
{
    let buffer = Fp4::from(left)
        .into_bytes()
        .into_iter()
        .chain(Fp4::from(right).into_bytes().into_iter())
        .collect_vec();

    blake3::hash(&buffer).into()
}

/// Create hash leaves from pairs of left and right field elements for Merkle tree construction.
pub fn create_hash_leaves_from_pairs<T>(left: &[T], right: &[T]) -> Vec<[u8; 32]>
where
    T: Copy,
    Fp4: RawDataSerializable + From<T>,
{
    zip(left, right)
        .map(|(&l, &r)| hash_field_pair(l, r))
        .collect()
}

/// Create hash leaves from pairs of left and right field elements (by reference) for Merkle tree construction.
pub fn create_hash_leaves_from_pairs_ref<T>(left: &[T], right: &[T]) -> Vec<[u8; 32]>
where
    T: Clone,
    Fp4: RawDataSerializable + From<T>,
{
    zip(left, right)
        .map(|(l, r)| hash_field_pair(l.clone(), r.clone()))
        .collect()
}
