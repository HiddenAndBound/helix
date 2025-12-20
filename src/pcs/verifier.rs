//! Verifier-side helpers for the BaseFold-style commitment layer.

use anyhow::bail;
use itertools::multizip;
use p3_field::{ExtensionField, Field, RawDataSerializable};

use crate::Fp;
use crate::Fp4;
use crate::merkle::{MerklePath, MerkleTree};
use crate::pcs::utils::{fold_pair, hash_field_pair};

/// Folds queried codeword pairs (provided as `Vec<F>`) using the round challenge.
pub fn fold_codewords_vec<F>(
    folded_codewords: &mut [Fp4],
    codewords: &[Vec<F>],
    queries: &[usize],
    r: Fp4,
    roots: &[Fp],
) where
    F: ExtensionField<Fp>,
    Fp4: ExtensionField<F>,
{
    for (query, fold, codeword_pair) in multizip((queries, folded_codewords, codewords)) {
        *fold = fold_pair(
            (codeword_pair[0], codeword_pair[1]),
            r,
            roots[*query].inverse(),
        );
    }
}

/// Folds each query's `2^k`-sized fiber across `k = challenges.len()` skipped rounds.
pub fn fold_codewords_vec_skip<F>(
    folded_codewords: &mut [Fp4],
    codewords: &[Vec<F>],
    queries: &[usize],
    challenges: &[Fp4],
    roots: &[Vec<Fp>],
    domain_size: usize,
) where
    F: ExtensionField<Fp>,
    Fp4: ExtensionField<F>,
{
    let rounds = challenges.len();
    let log_domain_size = domain_size.trailing_zeros() as usize;

    for codeword in codewords {
        assert_eq!(1 << rounds, codeword.len());
    }

    for (&query, fold, query_codewords) in multizip((queries, folded_codewords, codewords)) {
        let mut buff = query_codewords
            .iter()
            .map(|&cw| Fp4::from(cw))
            .collect::<Vec<_>>();

        for round in 0..rounds {
            let offset = 1 << (log_domain_size - rounds);
            for i in 0..buff.len() / 2 {
                buff[i] = fold_pair::<Fp4>(
                    (buff[i], buff[i + (buff.len() / 2)]),
                    challenges[round],
                    roots[round][query + i * offset].inverse(),
                );
            }
            buff.truncate(buff.len() / 2);
        }

        *fold = buff[0];
    }
}

/// Verifies Merkle authentication paths for `(Fp4, Fp4)` codeword pairs.
pub fn verify_paths(
    codewords: &[(Fp4, Fp4)],
    paths: &[MerklePath],
    queries: &[usize],
    oracle_commitment: [u8; 32],
) -> anyhow::Result<()> {
    for (query, path, &codeword_pair) in multizip((queries, paths, codewords)) {
        let (left, right) = codeword_pair;
        let leaf_hash = hash_field_pair(left, right);
        MerkleTree::verify_path(leaf_hash, *query, path, oracle_commitment)?;
    }
    Ok(())
}

/// Verifies Merkle authentication paths for `Vec<F>` codeword "pairs" (length 2).
pub fn verify_paths_vec<F>(
    codewords: &[Vec<F>],
    paths: &[MerklePath],
    queries: &[usize],
    oracle_commitment: [u8; 32],
) -> anyhow::Result<()>
where
    F: ExtensionField<Fp> + RawDataSerializable,
    Fp4: From<F>,
{
    for (query, path, codeword_pair) in multizip((queries, paths, codewords)) {
        let leaf_hash = blake3::hash(
            &F::into_byte_stream(codeword_pair.clone())
                .into_iter()
                .collect::<Vec<_>>(),
        )
        .into();
        MerkleTree::verify_path(leaf_hash, *query, path, oracle_commitment)?;
    }
    Ok(())
}

/// Updates a single query index for the next folding round.
pub fn update_query(query: &mut usize, halfsize: usize) {
    debug_assert!(halfsize.is_power_of_two(), "halfsize must be a power of 2");
    *query &= halfsize - 1;
}

fn check_fold(
    folded_codeword: Fp4,
    query: usize,
    halfsize: usize,
    left: Fp4,
    right: Fp4,
) -> anyhow::Result<()> {
    debug_assert!(halfsize.is_power_of_two(), "halfsize must be a power of 2");

    if (query & halfsize) != 0 {
        if folded_codeword != right {
            bail!(
                "Folded codeword verification failed: expected {:?}, got {:?}",
                (left, right),
                folded_codeword
            );
        }
    } else if folded_codeword != left {
        bail!(
            "Folded codeword verification failed: expected {:?}, got {:?}",
            (left, right),
            folded_codeword
        );
    }

    Ok(())
}

/// Checks consistency between previously-folded codewords and newly opened codeword pairs.
pub fn check_query_consistency_vec(
    queries: &mut [usize],
    folded_codewords: &[Fp4],
    codewords: &[Vec<Fp4>],
    query_range: usize,
    round: usize,
) -> anyhow::Result<()> {
    if round > 0 {
        for (query, &folded_codeword, codeword_pair) in
            multizip((queries, folded_codewords, codewords))
        {
            if codeword_pair.len() != 2 {
                bail!("Codeword is not of correct length");
            }
            let (left, right) = (codeword_pair[0], codeword_pair[1]);
            check_fold(folded_codeword, *query, query_range, left, right)?;
            update_query(query, query_range);
        }
    }
    Ok(())
}
