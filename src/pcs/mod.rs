use crate::{
    Fp,
    challenger::{self, Challenger},
    commitment::PolynomialCommitment,
    merkle_tree::{self, MerklePath, MerkleTree},
    polynomial::MLE,
    spartan::univariate::UnivariatePoly,
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
pub struct Basefold;

type Commitment = [u8; 32];
type Encoding = Vec<Fp>;

const RATE: usize = 2;

pub struct Proof {
    pub sum_check_rounds: Vec<UnivariatePoly>,
    pub paths: Vec<Vec<MerklePath>>,
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
        eval_point: &[Fp],
        challenger: &mut Challenger,
        evaluation: Fp,
        merkle_tree: &MerkleTree,
        encoding: &Encoding,
    ) {
        unimplemented!("Basefold::evaluate is not yet implemented");
    }
}
