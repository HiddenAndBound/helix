//! # BaseFold Utility Functions
//!
//! This module provides the core utility functions that support the BaseFold polynomial commitment scheme.
//! These functions handle Reed-Solomon encoding, polynomial folding operations, Merkle tree utilities,
//! and cryptographic hash operations essential for the BaseFold protocol.
//!
//! ## Key Components
//!
//! ### Reed-Solomon Encoding
//! - `encode_mle()`: Transforms polynomial coefficients into Reed-Solomon encoded codewords
//! - Uses forward FFT for efficient evaluation at roots of unity
//! - Provides rate-1/2 encoding for error detection capabilities
//!
//! ### Folding Operations  
//! - `fold()` and `fold_pair()`: Core folding algorithms that reduce encoding size by half each round
//! - Maintains consistency between folded encodings and folded polynomials
//! - Uses field extension arithmetic for challenge integration
//!
//! ### Merkle Tree Support
//! - `get_codewords()` and `get_merkle_paths()`: Query response generation
//! - `hash_field_pair()`: Cryptographic hashing for Merkle leaves
//! - Optimized for the pairing structure used in BaseFold commitments
//!
//! ## Security Parameters
//!
//! - **QUERIES = 144**: Number of random queries for soundness (≈2⁻¹⁰⁰ security)
//! - **RATE = 2**: Reed-Solomon encoding rate (1/2 information rate)  
//! - **HALF = Fp::new(134217727)**: Precomputed constant 1/2 in BabyBear field

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

/// Cryptographic commitment represented as a 32-byte Blake3 hash.
pub type Commitment = [u8; 32];

/// Reed-Solomon encoded polynomial represented as a vector of field elements.
pub type Encoding = Vec<Fp>;

/// Number of random queries for soundness verification.
/// 144 queries provides approximately 2⁻¹⁰⁰ soundness error.
pub const QUERIES: usize = 144;

/// Reed-Solomon encoding rate (2x expansion for rate-1/2 code).
pub const RATE: usize = 2;

/// Precomputed constant 1/2 in BabyBear field for efficient folding operations.
pub const HALF: Fp = Fp::new(1006632961);

/// Encodes a multilinear polynomial using Reed-Solomon codes via forward FFT.
///
/// This function transforms polynomial coefficients into an error-correcting Reed-Solomon encoding
/// by evaluating the polynomial at twice as many points using FFT. The resulting encoding provides
/// the distance properties necessary for soundness in the BaseFold protocol.
///
/// # Mathematical Process
/// 1. **Zero-padding**: Extend coefficients to 2 * poly.len() with zeros (rate-1/2 encoding)
/// 2. **Forward FFT**: Apply forward_fft using provided roots of unity  
/// 3. **Result**: Reed-Solomon codeword with minimum distance ≥ poly.len() + 1
///
/// # Parameters
/// * `poly` - Multilinear polynomial with coefficients over base field Fp
/// * `roots` - Table of FFT roots of unity, must have sufficient depth for polynomial size
///
/// # Returns  
/// * `Encoding` - Reed-Solomon encoded codewords of length 2 * poly.len()
///
/// # Panics
/// * If root table doesn't have sufficient depth: roots.len() ≤ log₂(buffer.len())
///
/// # Security Properties
/// The Reed-Solomon encoding provides:
/// - **Minimum distance**: At least n/2 + 1 where n is codeword length
/// - **Error detection**: Can detect up to d-1 errors where d is minimum distance  
/// - **Unique decoding**: Up to (d-1)/2 errors can be corrected
///
/// # Implementation Notes
/// Uses the optimized forward_fft implementation from p3_baby_bear for efficiency.
/// The encoding is performed in-place on a zero-padded buffer for memory efficiency.
///
/// # Example
/// ```rust,ignore
/// let poly = MLE::new(vec![Fp::ONE, Fp::TWO, Fp::ZERO, Fp::ONE]);
/// let roots = generate_fft_roots_for_depth(poly.n_vars());
/// let encoding = encode_mle(&poly, &roots); // Length = 2 * poly.len()
/// ```
pub fn encode_mle(poly: &MLE<Fp>, roots: &[Vec<Fp>], rate: usize) -> Encoding {
    let mut buffer = vec![Fp::ZERO; poly.len() * rate];
    buffer[0..poly.len()].copy_from_slice(poly.coeffs());

    assert_eq!(
        roots.len(),
        buffer.len().trailing_zeros() as usize - rate.trailing_zeros() as usize,
        "Root table not large enough to encode the MLE."
    );

    BabyBear::forward_fft(&mut buffer, roots);
    bit_reverse_sort(&mut buffer);
    buffer
}

/// Folds a pair of Reed-Solomon codewords using a challenge and twiddle factor.
///
/// This is the fundamental folding operation that reduces encoding size by half in each BaseFold round.
/// The function combines two codewords using a random challenge and a twiddle factor to maintain
/// the Reed-Solomon structure while reducing the problem size.
///
/// # Mathematical Process
/// Given codewords (a₀, a₁), challenge r, and twiddle ω:
/// 1. Compute g₀ = (a₀ + a₁) * (1/2)
/// 2. Compute g₁ = (a₀ - a₁) * (1/2) * ω
/// 3. Return folded = r * (g₀ + g₁) + g₀ = g₀ + r * g₁
///
/// This preserves the polynomial evaluation structure while folding the encoding.
///
/// # Parameters
/// * `codewords` - Pair of codewords (a₀, a₁) to be folded  
/// * `r` - Random challenge from Fiat-Shamir challenger
/// * `twiddle` - Twiddle factor ω for maintaining FFT structure
///
/// # Returns
/// * `Fp4` - Folded codeword in extension field
///
/// # Type Parameters
/// * `F` - Base field type (typically Fp or Fp4)
///
/// # Mathematical Properties
/// - **Linearity**: fold_pair(a+b, r, ω) = fold_pair(a, r, ω) + fold_pair(b, r, ω)
/// - **Field extension**: Promotes base field elements to extension field as needed
/// - **Deterministic**: Same inputs always produce same output
///
/// # Example
/// ```rust,ignore
/// let pair = (Fp::ONE, Fp::TWO);
/// let challenge = Fp4::from_u32(42);
/// let twiddle = roots[round][index];
/// let folded = fold_pair(pair, challenge, twiddle);
/// ```
pub fn fold_pair<F>(codewords: (F, F), r: Fp4, twiddle: Fp) -> Fp4
where
    F: Field + Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let (a0, a1) = codewords;
    //todo: pass batch inverted twiddles
    let (g0, g1) = ((a0 + a1) * HALF, (a0 - a1) * HALF * twiddle.inverse());
    r*(g1 - g0) + g0
}

/// Folds an entire Reed-Solomon encoding by applying pairwise folding operations.
///
/// This function processes an encoding by folding each pair of codewords, reducing the
/// encoding size by half. This is the batch version of `fold_pair` applied to all
/// adjacent pairs in the encoding.
///
/// # Mathematical Process
/// For encoding E = [e₀, e₁, e₂, e₃, ..., e_{n-2}, e_{n-1}]:
/// 1. Pair elements: (e₀,e₁), (e₂,e₃), ..., (e_{n-2},e_{n-1})
/// 2. Apply fold_pair to each pair with corresponding twiddle factor
/// 3. Result: [fold_pair((e₀,e₁), r, ω₀), fold_pair((e₂,e₃), r, ω₁), ...]
///
/// # Parameters
/// * `mle` - Reed-Solomon encoding to fold (length must be even)
/// * `random_challenge` - Challenge r from Fiat-Shamir challenger  
/// * `roots` - Twiddle factors for each pair, length = mle.len()/2
///
/// # Returns
/// * `Vec<Fp4>` - Folded encoding of length mle.len()/2 in extension field
///
/// # Type Parameters  
/// * `F` - Base field type of input encoding
///
/// # Consistency Properties
/// The folding maintains the polynomial evaluation structure such that:
/// - If original encoding corresponds to polynomial P(x₁,...,xₙ)
/// - Then folded encoding corresponds to P.fold_in_place(r) over (x₂,...,xₙ)
///
/// # Performance
/// - Time complexity: O(mle.len())
/// - Space complexity: O(mle.len()/2) for result
/// - Uses vectorized operations where possible
///
/// # Example
/// ```rust,ignore  
/// let encoding = vec![Fp::ONE, Fp::TWO, Fp::ZERO, Fp::ONE]; // Length 4
/// let challenge = Fp4::from_u32(42);
/// let twiddle_factors = &roots[round]; // Length 2 for 4→2 folding
/// let folded = fold(&encoding, challenge, twiddle_factors); // Length 2
/// ```
pub fn fold<F>(code: &[F], random_challenge: Fp4, roots: &[Fp]) -> Vec<Fp4>
where
    F: Field + Mul<Fp, Output = F>,
    Fp4: ExtensionField<F>,
{
    let half_size = code.len() >> 1;
    let mut folded = Vec::with_capacity(half_size);
    for i in 0..half_size {
        folded.push(fold_pair(
            (code[i], code[i + half_size]),
            random_challenge,
            roots[i],
        ));
    }

    folded
}

/// Retrieves codeword pairs from an encoding at specified query positions.
///
/// This function extracts the Reed-Solomon codewords needed to answer verifier queries.
/// For each query position, it returns the corresponding paired codewords that will be
/// used in the folding verification process.
///
/// # Query Structure  
/// The function handles the pairing structure used in BaseFold:
/// - Encoding is conceptually split into two halves: [left_half | right_half]
/// - For query position i:
///   - If i ≤ halfsize: return (left[i], right[i])  
///   - If i > halfsize: return (right[i-halfsize], left[i-halfsize])
/// - This ensures consistent pairing for the folding verification
///
/// # Parameters
/// * `queries` - List of query positions to retrieve codewords for
/// * `encoding` - Reed-Solomon encoding to query from
///
/// # Returns
/// * `Vec<(Fp4, Fp4)>` - Paired codewords for each query position
///
/// # Type Parameters
/// * `F` - Field type of encoding elements (convertible to Fp4)
///
/// # Mathematical Properties  
/// The returned pairs satisfy:
/// - Each pair represents adjacent elements that would be folded together
/// - Pairing is consistent with the Merkle tree leaf construction
/// - Field elements are promoted to Fp4 for uniform handling
///
/// # Usage in Verification
/// These codeword pairs are used by the verifier to:
/// 1. Verify Merkle authentication paths  
/// 2. Check folding consistency across rounds
/// 3. Detect encoding manipulation by the prover
///
/// # Example
/// ```rust,ignore
/// let queries = vec![0, 2, 5]; // Query positions  
/// let encoding = vec![Fp::ONE, Fp::TWO, Fp::ZERO, Fp::ONE, ...];
/// let codewords = get_codewords(&queries, &encoding);
/// // Returns pairs like [(encoding[0], encoding[4]), (encoding[2], encoding[6]), ...]
/// ```
pub fn get_codewords<F: Into<Fp4> + Copy>(queries: &[usize], encoding: &[F]) -> Vec<(Fp4, Fp4)> {
    let halfsize = encoding.len() / 2;
    queries
        .iter()
        .copied()
        .map(|i| {
            if i >= halfsize {
                (encoding[i ^ halfsize].into(), encoding[i].into())
            } else {
                (encoding[i].into(), encoding[i ^ halfsize].into())
            }
        })
        .collect()
}

/// Generates Merkle authentication paths for queried positions in the encoding.
///
/// This function retrieves the cryptographic proofs needed to authenticate that queried
/// codewords are genuine parts of the committed encoding. The paths enable the verifier
/// to check Merkle tree membership without possessing the full encoding.
///
/// # Path Generation Process
/// For each query position i:
/// 1. **Index adjustment**: If i > halfsize, use i ⊕ halfsize to get the leaf index
/// 2. **Path extraction**: Get authentication path from leaf to root  
/// 3. **Optimization**: Adjacent pairs that differ in MSB are hashed together as leaves
///
/// # Parameters
/// * `queries` - Query positions to generate authentication paths for
/// * `merkle_tree` - Merkle tree built over the Reed-Solomon encoding
///
/// # Returns  
/// * `Vec<MerklePath>` - Authentication paths for each query position
///
/// # Index Mapping Logic
/// The function uses XOR operations to handle the pairing structure:
/// - Pairs differing in their MSB are hashed together as single leaves
/// - Query indices above halfsize toggle their MSB to get valid leaf indices
/// - This ensures consistent path generation for paired codewords
///
/// # Security Properties
/// - **Authenticity**: Paths cryptographically prove membership in committed encoding
/// - **Efficiency**: Log-depth paths provide succinct proofs
/// - **Binding**: Forged paths would require breaking collision-resistance of Blake3
///
/// # Usage in Verification
/// The generated paths are used to:
/// 1. Verify that returned codewords match the commitment
/// 2. Detect any tampering with the encoding between commit and query phases  
/// 3. Provide cryptographic assurance of encoding integrity
///
/// # Example
/// ```rust,ignore
/// let queries = vec![0, 3, 7]; // Query positions
/// let paths = get_merkle_paths(&queries, &merkle_tree);
/// // Each path proves membership: MerkleTree::verify_path(leaf_hash, query, path, root)
/// ```
pub fn get_merkle_paths(queries: &[usize], merkle_tree: &MerkleTree) -> Vec<MerklePath> {
    let halfsize = 1 << merkle_tree.depth;
    queries
        .iter()
        .copied()
        .map(|i| {
            // As pairs that differ in their most dominant bit are hashed together, we only index leaf nodes by the trailing bits.
            // Thus in the case an index is above the halfsize, meaning its leading bit is 1, we toggle that bit to get the valid and appropriate index.
            if i >= halfsize {
                merkle_tree.get_path(i ^ halfsize)
            } else {
                merkle_tree.get_path(i)
            }
        })
        .collect()
}

/// Computes a cryptographic hash of a pair of field elements using Blake3.
///
/// This function provides the cryptographic binding for Merkle tree construction by
/// hashing pairs of Reed-Solomon codewords. The hash function is used to build
/// Merkle tree leaves from adjacent encoding elements.
///
/// # Hashing Process
/// 1. **Field extension**: Convert both elements to Fp4 for uniform representation
/// 2. **Serialization**: Convert Fp4 elements to bytes using RawDataSerializable  
/// 3. **Concatenation**: Combine left and right element bytes
/// 4. **Cryptographic hash**: Apply Blake3 hash function to get 32-byte digest
///
/// # Parameters
/// * `left` - Left field element of the pair
/// * `right` - Right field element of the pair  
///
/// # Returns
/// * `[u8; 32]` - Blake3 hash digest of the serialized pair
///
/// # Type Parameters
/// * `T` - Field element type (must be convertible to Fp4)
///
/// # Security Properties
/// - **Collision resistance**: Breaking requires 2¹²⁸ operations under Blake3 assumptions
/// - **Preimage resistance**: Given hash, finding preimage requires 2²⁵⁶ operations
/// - **Consistency**: Same field element pairs always produce identical hashes
///
/// # Usage in BaseFold
/// Used to construct Merkle tree leaves by hashing adjacent codeword pairs:
/// ```text
/// leaves[i] = hash_field_pair(encoding[2*i], encoding[2*i+1])
/// ```
///
/// # Implementation Notes
/// - Uses Blake3 for fast, secure hashing optimized for modern hardware
/// - Serialization follows p3-field's canonical byte representation
/// - All field arithmetic performed in Fp4 for consistency
///
/// # Example
/// ```rust,ignore
/// let left = Fp::ONE;  
/// let right = Fp::TWO;
/// let leaf_hash = hash_field_pair(left, right);
/// // leaf_hash can now be used as a Merkle tree leaf
/// ```
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

/// Performs an in-place bit-reverse sort on a vector.
///
/// This function rearranges the elements in a vector so that the element at position `i`
/// is moved to position `bit_reverse(i)`. This reordering is commonly used in FFT algorithms
/// to optimize memory access patterns and eliminate the need for bit-reversal during the
/// transform process.
///
/// # Mathematical Process
/// For a vector of length n = 2^k:
/// 1. **Index computation**: For each position i, compute j = bit_reverse(i, k bits)
/// 2. **Element swapping**: If i < j, swap elements at positions i and j
/// 3. **Cycle completion**: Each element reaches its final bit-reversed position
///
/// The bit-reversal operation reverses the binary representation of the index within
/// k bits, where k = log₂(n).
///
/// # Parameters
/// * `vec` - Vector to sort in-place (length must be a power of 2)
///
/// # Panics
/// * If vector length is not a power of 2
///
/// # Performance
/// - **Time complexity**: O(n) where n is vector length
/// - **Space complexity**: O(1) additional space (in-place operation)
/// - **Optimization**: Uses `std::intrinsics::bitreverse` for maximum efficiency
///
/// # Usage in FFT Algorithms
/// Many FFT implementations require input data in bit-reversed order to optimize
/// the butterfly operations and memory access patterns. This function pre-processes
/// the data to eliminate bit-reversal overhead during the actual FFT computation.
///
/// # Example
/// ```rust,ignore
/// let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// bit_reverse_sort(&mut vec);
/// // Result: [0, 4, 2, 6, 1, 5, 3, 7] (indices 0,1,2,3,4,5,6,7 -> 0,4,2,6,1,5,3,7)
/// ```
pub fn bit_reverse_sort<T>(vec: &mut Vec<T>) {
    let len = vec.len();
    assert!(
        len == 0 || len.is_power_of_two(),
        "Vector length must be a power of 2, got {}",
        len
    );

    if len <= 1 {
        return;
    }

    let num_bits = len.trailing_zeros();

    for i in 0..len {
        let j = (std::intrinsics::bitreverse(i) >> (usize::BITS - num_bits)) as usize;
        if i < j {
            vec.swap(i, j);
        }
    }
}

/// Returns a new vector with elements arranged in bit-reversed order.
///
/// This function creates a new vector where each element from the original vector at position `i`
/// is placed at position `bit_reverse(i)`. Unlike `bit_reverse_sort`, this function does not
/// modify the original vector but returns a new one with the reordered elements.
///
/// # Mathematical Process
/// For a vector of length n = 2^k:
/// 1. **New vector creation**: Create a new vector of the same length
/// 2. **Index mapping**: For each position i, compute j = bit_reverse(i, k bits)  
/// 3. **Element copying**: Copy element from position i to position j in the new vector
///
/// # Parameters
/// * `vec` - Vector to reorder (length must be a power of 2)
///
/// # Returns
/// * `Vec<T>` - New vector with elements in bit-reversed order
///
/// # Panics
/// * If vector length is not a power of 2
///
/// # Performance
/// - **Time complexity**: O(n) where n is vector length
/// - **Space complexity**: O(n) for the new vector
/// - **Memory**: Requires cloning each element once
///
/// # Type Requirements
/// Elements must implement `Clone` since they are copied to the new vector.
///
/// # Usage
/// This function is useful when you need to maintain the original vector while also
/// having a bit-reversed version, or when working with immutable data structures.
///
/// # Example
/// ```rust,ignore
/// let vec = vec![0, 1, 2, 3, 4, 5, 6, 7];
/// let reversed = bit_reverse_sorted(&vec);
/// // Original: [0, 1, 2, 3, 4, 5, 6, 7]
/// // Result:   [0, 4, 2, 6, 1, 5, 3, 7]
/// ```
pub fn bit_reverse_sorted<T: Clone>(vec: &Vec<T>) -> Vec<T> {
    let len = vec.len();
    assert!(
        len == 0 || len.is_power_of_two(),
        "Vector length must be a power of 2, got {}",
        len
    );

    if len <= 1 {
        return vec.clone();
    }

    let num_bits = len.trailing_zeros();
    let mut result = vec![vec[0].clone(); len];

    for i in 0..len {
        let j = (std::intrinsics::bitreverse(i) >> (usize::BITS - num_bits)) as usize;
        result[j] = vec[i].clone();
    }

    result
}

#[cfg(test)]
mod tests {
    use p3_field::integers::QuotientMap;

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
        assert_eq!(vec, vec![]);
    }

    #[test]
    fn test_bit_reverse_sort_large() {
        let mut vec: Vec<usize> = (0..16).collect();
        bit_reverse_sort(&mut vec);
        
        // For length 16 (4 bits), verify a few key mappings:
        // 0 (0000) -> 0 (0000), 1 (0001) -> 8 (1000), 2 (0010) -> 4 (0100), 3 (0011) -> 12 (1100)
        assert_eq!(vec[0], 0);  // 0 -> 0
        assert_eq!(vec[8], 1);  // 1 -> 8
        assert_eq!(vec[4], 2);  // 2 -> 4  
        assert_eq!(vec[12], 3); // 3 -> 12
        
        // Check that all original elements are present
        let mut sorted_vec = vec.clone();
        sorted_vec.sort();
        assert_eq!(sorted_vec, (0..16).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "Vector length must be a power of 2")]
    fn test_bit_reverse_sort_non_power_of_two() {
        let mut vec = vec![1, 2, 3];  // length 3 is not a power of 2
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
        
        assert_eq!(result, vec!["a".to_string(), "c".to_string(), "b".to_string(), "d".to_string()]);
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
        let vec = vec![1, 2, 3, 4, 5];  // length 5 is not a power of 2
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
            ComplexData { id: 0, name: "zero".to_string() },
            ComplexData { id: 1, name: "one".to_string() },
            ComplexData { id: 2, name: "two".to_string() },
            ComplexData { id: 3, name: "three".to_string() },
        ];
        
        let result = bit_reverse_sorted(&vec);
        
        // Expected order: [0, 2, 1, 3]
        assert_eq!(result[0].id, 0);
        assert_eq!(result[1].id, 2);
        assert_eq!(result[2].id, 1);
        assert_eq!(result[3].id, 3);
    }
}
