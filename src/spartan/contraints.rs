//Describes an R1CS instance.
pub struct ConstraintSystem {
    A: SparseMLE,
    B: SparseMLE,
    C: SparseMLE,
    io: MLE<Fp>,
    w: MLE<Fp>,
}

impl ConstraintSystem {
    pub fn new(A: SparseMLE, B: SparseMLE, C: SparseMLE, io: MLE<Fp>, w: MLE<Fp>) -> Self {
        Self { A, B, C, io, w }
    }
    
}
