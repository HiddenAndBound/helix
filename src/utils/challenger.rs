use blake3::{self, Hasher};
use p3_baby_bear::BabyBear;
use p3_field::{RawDataSerializable, PrimeCharacteristicRing, PackedValue};
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
    
    pub fn observe_fp4_elems(&mut self, input: &[crate::utils::Fp4]) {
        self.state = Hasher::new();
        for element in input {
            // Hash each coefficient of the Fp4 element
            for coeff in element.as_slice() {
                let bytes: Vec<u8> = coeff.into_bytes().into_iter().collect();
                self.state.update(&bytes);
            }
        }
        self.round += 1;
        self.state.update(&self.round.to_le_bytes());
    }

    pub fn get_challenge(&mut self) -> crate::utils::Fp4 {
        let challenge_bytes: [u8; 16] = self.state.finalize().as_bytes()[0..16]
            .try_into()
            .expect("Hash output is 32 bytes, should be able to get array of size 16");
        
        // Create 4 BabyBear elements from the bytes
        let mut coeffs = [BabyBear::ZERO; 4];
        for i in 0..4 {
            let bytes = [
                challenge_bytes[i * 4],
                challenge_bytes[i * 4 + 1],
                challenge_bytes[i * 4 + 2],
                challenge_bytes[i * 4 + 3],
            ];
            coeffs[i] = BabyBear::new(u32::from_le_bytes(bytes));
        }
        
        // Convert base field elements to extension field
        let mut fp4_coeffs = [crate::utils::Fp4::ZERO; 1];
        fp4_coeffs[0] = crate::utils::Fp4::from_base_slice(&coeffs);
        let challenge_fp4 = fp4_coeffs[0];
        
        // Update state with the challenge for next round
        self.state.reset();
        for coeff in &coeffs {
            self.state.update(&coeff.into_bytes());
        }
        self.round += 1;
        
        challenge_fp4
    }
}
