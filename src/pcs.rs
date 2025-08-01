use ark_ff::Field;

pub trait PolynomialCommitmentScheme<F: Field> {
    type Polynomial;
    type Commitment;
    type Proof;
    type Prover;
    type Verifier;

    fn new() -> Self;

    fn commit(&self, poly: &Self::Polynomial) -> Self::Commitment;

    fn prove(&self, poly: &Self::Polynomial, point: F) -> Self::Proof;

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: F,
        value: F,
        proof: &Self::Proof,
    ) -> bool;
}