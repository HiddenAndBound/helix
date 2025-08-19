use std::{
    ops::{Add, Mul},
    thread::current,
    vec,
};

use crate::{
    Fp, Fp4,
    challenger::{self, Challenger},
    commitment::PolynomialCommitment,
    eq::EqEvals,
    merkle_tree::{self, MerklePath, MerkleTree},
    polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};
use anyhow::Ok;
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField, PrimeField32, TwoAdicField};
pub struct Basefold;

type Commitment = [u8; 32];
type Encoding = Vec<Fp>;

const RATE: usize = 2;
const HALF: Fp = Fp::new(134217727);
pub struct EvalProof {
    pub sum_check_rounds: Vec<UnivariatePoly>,
    pub paths: Vec<Vec<MerklePath>>,
    pub commitments: Vec<Commitment>,
    pub codewords: Vec<Vec<(Fp, Fp)>>,
}

impl Basefold {
    fn encode(poly: &MLE<Fp>, roots: &[Vec<Fp>]) -> Encoding {
        let mut buffer = vec![Fp::ZERO; poly.len() * RATE];
        buffer[0..poly.len()].copy_from_slice(poly.coeffs());

        assert!(
            roots.len() > buffer.len().trailing_zeros() as usize,
            "Root table not large enough to encode the MLE."
        );

        BabyBear::forward_fft(&mut buffer, roots);
        buffer
    }

    pub fn commit(poly: &MLE<Fp>, roots: Vec<Vec<Fp>>) -> (MerkleTree, Commitment, Encoding) {
        assert!(
            poly.len().is_power_of_two(),
            "MLE's coefficients need to be a power of 2"
        );

        let buffer = Self::encode(poly, &roots);

        let merkle_tree = MerkleTree::new(&buffer).unwrap();
        let commitment = merkle_tree.root();

        (merkle_tree, commitment, buffer)
    }

    pub fn evaluate(
        poly: &MLE<Fp>,
        eval_point: &[Fp4],
        challenger: &mut Challenger,
        evaluation: Fp4,
        merkle_tree: MerkleTree,
        encoding: Encoding,
        roots: Vec<Vec<Fp>>,
    ) -> Result<EvalProof, anyhow::Error> {
        let eq = EqEvals::gen_from_point(&eval_point[1..]);
        let rounds = poly.n_vars();

        let mut current_claim = evaluation;

        let mut random_point = Vec::new();
        let mut round_proofs = Vec::new();

        let mut commitments = Vec::new();
        let mut merkle_trees = Vec::new();
        let mut encodings = Vec::<Vec<Fp4>>::new();


        let mut current_poly;
        for round in 0..rounds {
            let mut g_0 = Fp4::ZERO;

            for i in 0..1 << (rounds - round - 1) {
                g_0 += eq[i] * poly[i << 1]
            }

            let g1 = (current_claim - (Fp4::ONE - eval_point[round]) * g_0) / eval_point[0];

            let round_coeffs = vec![g_0, g1 - g_0];
            let round_proof = UnivariatePoly::new(round_coeffs).unwrap();

            challenger.observe_fp4_elems(&round_proof.coefficients());

            let r = challenger.get_challenge();

            // CODE FOLDING: if round is 0, fold the encoding of the original MLE's dense representation. Else fold the previous encoding.
            let current_encoding = match round {
                0 => fold(&encoding, r, &roots[round]),
                _ => fold(
                    encodings.last().expect("Will be non-empty"),
                    r,
                    &roots[round],
                ),
            };

            //If we are in the first round 
            current_poly = match round {
                0 => poly.fold_in_place(r),
                _ => current_poly.fold_in_place(r)
            };

            let current_merkle_tree = MerkleTree::new(&current_encoding)?;
            let current_commitment = current_merkle_tree.root();

            commitments.push(current_commitment);
            merkle_trees.push(current_merkle_tree);
            encodings.push(current_encoding);

            current_claim = round_proof.evaluate(r);

            round_proofs.push(round_proof);
            random_point.push(r);
        }

        Ok(EvalProof {
            sum_check_rounds: round_proofs,
            paths: todo!(),
            commitments: todo!(),
            codewords: todo!(),
        })
    }

    pub fn verify(
        proof: EvalProof,
        evaluation: Fp4,
        eval_point: &[Fp4],
        commitment: Commitment,
        challenger: &mut Challenger,
    ) {
        let mut current_claim = evaluation;

        let rounds = eval_point.len();

        let mut random_point = Vec::new();
        for round in 0..rounds {
            let round_poly = &proof.sum_check_rounds[round];

            assert_eq!(
                current_claim,
                (Fp4::ONE - eval_point[round]) * round_poly.evaluate(Fp4::ZERO)
                    + eval_point[round] * round_poly.evaluate(Fp4::ONE)
            );

            challenger.observe_fp4_elems(&round_poly.coefficients());

            let r = challenger.get_challenge();

            current_claim = round_poly.evaluate(r);
            random_point.push(r);
        }
    }
}

pub fn fold_pair<F>(codewords: (F, F), r: Fp4, twiddle: Fp) -> Fp4
where
    F: PrimeCharacteristicRing + Mul<Fp, Output = F> + Copy,
    Fp4: Mul<F, Output = Fp4> + Add<F, Output = Fp4>,
{
    let (a0, a1) = codewords;
    let (g0, g1) = ((a0 - a1) * HALF, (a0 + a1) * HALF * twiddle);
    r * (g0 + g1) + g0
}

pub fn fold<F>(mle: &[F], random_challenge: Fp4, roots: &[Fp]) -> Vec<Fp4>
where
    F: PrimeCharacteristicRing + Mul<Fp, Output = F> + Copy,
    Fp4: Mul<F, Output = Fp4> + Add<F, Output = Fp4>,
{
    let half_size = mle.len() >> 1;
    let mut folded = vec![Fp4::ZERO; half_size];
    for i in 0..half_size {
        folded[i] = fold_pair((mle[i], mle[i + half_size]), random_challenge, roots[i])
    }

    folded
}

struct RoundData{
        commitments: Vec<Commitment>,
        merkle_trees: Vec<MerkleTree>,
        encodings: Vec<Encoding>,
    };
    
