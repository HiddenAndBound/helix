use crate::{Fp, commitment::PolynomialCommitment, polynomial::MLE};

pub struct Basefold{

};

impl PolynomialCommitment<Fp> for Basefold {
    type Commitment = [u8; 32];

    type Proof = ();

    type Error = ();

    fn commit(polynomial: &MLE<Fp>) -> Result<Self::Commitment, Self::Error> {}

    fn open(
        polynomial: &crate::polynomial::MLE<Fp>,
        point: &[crate::Fp4],
        commitment: &Self::Commitment,
    ) -> Result<Self::Proof, Self::Error> {
        todo!()
    }

    fn verify(
        commitment: &Self::Commitment,
        point: &[crate::Fp4],
        value: crate::Fp4,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        todo!()
    }
}
