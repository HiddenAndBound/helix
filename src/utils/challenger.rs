use blake3::{self, Hasher};
use p3_baby_bear::BabyBear;
use p3_field::RawDataSerializable;
pub struct Challenger {
    state: Hasher,
    round: usize,
}

impl Challenger {
    pub fn new() -> Self {
        Challenger {
            state: Hasher::new(),
            round: 0,
        }
    }

    pub fn observe_field_elem(&mut self, input: &BabyBear) {
        self.state.update(&input.into_bytes());
        self.round += 1;
        self.state.update(&self.round.to_le_bytes());
    }

    pub fn observe_field_elems(&mut self, input: &[BabyBear]) {
        self.state = Hasher::new();
        for element in input {
            self.state.update(&element.into_bytes());
        }
        self.round += 1;
        self.state.update(&self.round.to_le_bytes());
    }

    pub fn get_challenge(&mut self) {
        let challenge_bytes: [u8; 4] = self.state.finalize().as_bytes()[0..4]
            .try_into()
            .expect("Hash output is 32 bytes, should be able to get array of size 4");
        let challenge = BabyBear::new(u32::from_be_bytes(challenge_bytes));
        self.state.reset().update(&challenge.into_bytes());
        self.round += 1;
    }
}
