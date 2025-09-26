use crate::spartan::error::{SparseError, SparseResult};
use crate::spartan::r1cs::{R1CS, Witness};
use crate::spartan::spark::sparse::SparseMLE;
use p3_baby_bear::{
    BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL, BabyBear,
    Poseidon2BabyBear, default_babybear_poseidon2_16,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_symmetric::Permutation;
use std::collections::HashMap;

const WIDTH: usize = 16;
const RATE: usize = 2;
const EXTERNAL_ROUNDS: usize = 4;
const INTERNAL_ROUNDS: usize = 13;

/// Layout metadata for the Poseidon2 R1CS instance.
#[derive(Debug, Clone)]
pub struct Poseidon2Layout {
    /// Indices of the public input lanes in the witness vector.
    pub public_input_positions: Vec<usize>,
    /// Variable index reserved for the constant-`1` wire.
    pub one_index: usize,
    /// Variable indices storing the final Poseidon2 state after all rounds.
    pub final_state_positions: [usize; WIDTH],
}

/// Convenience container bundling the R1CS witness with Poseidon2-specific metadata.
#[derive(Debug, Clone)]
pub struct Poseidon2Witness {
    /// Spartan witness split into public and private segments.
    pub witness: Witness,
    /// Full assignment vector assembled during witness generation.
    pub full_assignment: Vec<BabyBear>,
    /// Final Poseidon2 state values (all 16 lanes).
    pub final_state: [BabyBear; WIDTH],
    /// Digest extracted from the rate portion (first two lanes).
    pub digest: [BabyBear; RATE],
    /// Layout metadata describing index positions.
    pub layout: Poseidon2Layout,
}

/// A ready-to-use R1CS instance enforcing the Poseidon2 permutation over BabyBear.
#[derive(Debug, Clone)]
pub struct Poseidon2Instance {
    pub r1cs: R1CS,
    pub witness: Poseidon2Witness,
}

/// Construct the Poseidon2 R1CS instance together with a satisfying witness.
///
/// The returned instance wires a single Poseidon2 permutation with width 16, modelling the
/// round schedule used in Plonky3. The first 16 witness variables correspond to the initial
/// state and are exposed as public inputs so callers can decide which lanes to open.
///
/// * `rate` – up to two field elements absorbed into lanes 0 and 1.
/// * `capacity` – optional array supplying the remaining 14 lanes (defaults to zeroes).
/// * `poseidon` – reference permutation used for validation; call
///   [`default_babybear_poseidon2_16`] when unsure.
pub fn build_poseidon2_instance(
    rate: &[BabyBear],
    capacity: Option<&[BabyBear; WIDTH - RATE]>,
    poseidon: &Poseidon2BabyBear<WIDTH>,
) -> SparseResult<Poseidon2Instance> {
    if rate.len() > RATE {
        return Err(SparseError::ValidationError(
            "rate must contain at most two elements".to_string(),
        ));
    }

    let mut initial_state = [BabyBear::ZERO; WIDTH];
    initial_state[..rate.len()].copy_from_slice(rate);
    if let Some(capacity) = capacity {
        initial_state[RATE..].copy_from_slice(capacity);
    }

    let mut builder = ConstraintBuilder::new(initial_state.to_vec());
    let mut state_indices = core::array::from_fn(|i| i);
    let mut state_values = initial_state;

    let num_public_inputs = WIDTH;
    let one_index = builder.alloc(BabyBear::ONE);
    builder.enforce_constant_one(one_index);

    // Initial external linear layer applied before the first external round block.
    let (next_indices, next_values) =
        external_linear_layer(&mut builder, &state_indices, &state_values, one_index);
    state_indices = next_indices;
    state_values = next_values;

    for round in 0..EXTERNAL_ROUNDS {
        let constants = BABYBEAR_RC16_EXTERNAL_INITIAL
            .get(round)
            .expect("missing initial round constant");
        let (indices_after_const, values_after_const) = add_round_constants(
            &mut builder,
            &state_indices,
            &state_values,
            constants,
            one_index,
        );
        let (indices_after_sbox, values_after_sbox) =
            full_sbox_layer(&mut builder, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) = external_linear_layer(
            &mut builder,
            &indices_after_sbox,
            &values_after_sbox,
            one_index,
        );
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    for round in 0..INTERNAL_ROUNDS {
        let rc = BABYBEAR_RC16_INTERNAL
            .get(round)
            .expect("missing internal round constant");
        let (indices_after_const, values_after_const) =
            add_internal_constant(&mut builder, &state_indices, &state_values, *rc, one_index);
        let (indices_after_sbox, values_after_sbox) =
            single_sbox_layer(&mut builder, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) = internal_linear_layer(
            &mut builder,
            &indices_after_sbox,
            &values_after_sbox,
            one_index,
        );
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    for round in 0..EXTERNAL_ROUNDS {
        let constants = BABYBEAR_RC16_EXTERNAL_FINAL
            .get(round)
            .expect("missing terminal round constant");
        let (indices_after_const, values_after_const) = add_round_constants(
            &mut builder,
            &state_indices,
            &state_values,
            constants,
            one_index,
        );
        let (indices_after_sbox, values_after_sbox) =
            full_sbox_layer(&mut builder, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) = external_linear_layer(
            &mut builder,
            &indices_after_sbox,
            &values_after_sbox,
            one_index,
        );
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    // Sanity-check against the reference permutation to ensure we wired everything correctly.
    let mut expected_state = initial_state;
    poseidon.permute_mut(&mut expected_state);
    if expected_state != state_values {
        return Err(SparseError::ValidationError(
            "Poseidon2 wiring mismatch compared to reference permutation".to_string(),
        ));
    }

    let final_state_positions = state_indices;
    let digest = [state_values[0], state_values[1]];

    let padded_len = builder.assignment.len().next_power_of_two();
    if builder.assignment.len() < padded_len {
        builder.assignment.resize(padded_len, BabyBear::ZERO);
    }

    let full_assignment = builder.assignment.clone();
    let witness = Witness::from_vec(full_assignment.clone(), num_public_inputs);

    let layout = Poseidon2Layout {
        public_input_positions: (0..num_public_inputs).collect(),
        one_index,
        final_state_positions,
    };

    let poseidon_witness = Poseidon2Witness {
        witness,
        full_assignment,
        final_state: state_values,
        digest,
        layout,
    };

    let a = SparseMLE::new(builder.a)?;
    let b = SparseMLE::new(builder.b)?;
    let c = SparseMLE::new(builder.c)?;
    let r1cs = R1CS::new(a, b, c, num_public_inputs)?;

    Ok(Poseidon2Instance {
        r1cs,
        witness: poseidon_witness,
    })
}

/// Helper for consumers that do not need to pass an explicit Poseidon2 reference implementation.
pub fn build_default_poseidon2_instance(
    rate: &[BabyBear],
    capacity: Option<&[BabyBear; WIDTH - RATE]>,
) -> SparseResult<Poseidon2Instance> {
    let poseidon = default_babybear_poseidon2_16();
    build_poseidon2_instance(rate, capacity, &poseidon)
}

struct ConstraintBuilder {
    a: HashMap<(usize, usize), BabyBear>,
    b: HashMap<(usize, usize), BabyBear>,
    c: HashMap<(usize, usize), BabyBear>,
    next_row: usize,
    assignment: Vec<BabyBear>,
}

impl ConstraintBuilder {
    fn new(initial_assignment: Vec<BabyBear>) -> Self {
        Self {
            a: HashMap::new(),
            b: HashMap::new(),
            c: HashMap::new(),
            next_row: 0,
            assignment: initial_assignment,
        }
    }

    fn alloc(&mut self, value: BabyBear) -> usize {
        let idx = self.assignment.len();
        self.assignment.push(value);
        idx
    }

    fn enforce_constant_one(&mut self, idx: usize) {
        let row = self.next_row;
        self.next_row += 1;
        Self::accumulate(&mut self.a, row, idx, BabyBear::ONE);
        Self::accumulate(&mut self.b, row, idx, BabyBear::ONE);
        Self::accumulate(&mut self.c, row, idx, BabyBear::ONE);
    }

    fn add_mul_constraint(&mut self, left: usize, right: usize, out: usize) {
        let row = self.next_row;
        self.next_row += 1;
        Self::accumulate(&mut self.a, row, left, BabyBear::ONE);
        Self::accumulate(&mut self.b, row, right, BabyBear::ONE);
        Self::accumulate(&mut self.c, row, out, BabyBear::ONE);
    }

    fn enforce_linear_relation(
        &mut self,
        terms: &[(usize, BabyBear)],
        constant: BabyBear,
        one_index: usize,
    ) {
        let row = self.next_row;
        self.next_row += 1;
        for &(idx, coeff) in terms {
            if coeff != BabyBear::ZERO {
                Self::accumulate(&mut self.a, row, idx, coeff);
            }
        }
        if constant != BabyBear::ZERO {
            Self::accumulate(&mut self.a, row, one_index, constant);
        }
        Self::accumulate(&mut self.b, row, one_index, BabyBear::ONE);
    }

    fn accumulate(
        matrix: &mut HashMap<(usize, usize), BabyBear>,
        row: usize,
        col: usize,
        coeff: BabyBear,
    ) {
        matrix
            .entry((row, col))
            .and_modify(|existing| {
                *existing += coeff;
            })
            .or_insert(coeff);
    }
}

fn add_round_constants(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    constants: &[BabyBear; WIDTH],
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    for lane in 0..WIDTH {
        let updated = state_values[lane] + constants[lane];
        let idx = builder.alloc(updated);
        let terms = [
            (idx, BabyBear::NEG_ONE),
            (state_indices[lane], BabyBear::ONE),
        ];
        builder.enforce_linear_relation(&terms, constants[lane], one_index);
        next_indices[lane] = idx;
        next_values[lane] = updated;
    }

    (next_indices, next_values)
}

fn add_internal_constant(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    constant: BabyBear,
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    let updated = state_values[0] + constant;
    let idx = builder.alloc(updated);
    let terms = [(idx, BabyBear::NEG_ONE), (state_indices[0], BabyBear::ONE)];
    builder.enforce_linear_relation(&terms, constant, one_index);
    next_indices[0] = idx;
    next_values[0] = updated;

    (next_indices, next_values)
}

fn full_sbox_layer(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    for lane in 0..WIDTH {
        let (idx, value) = apply_sbox(builder, state_indices[lane], state_values[lane]);
        next_indices[lane] = idx;
        next_values[lane] = value;
    }

    (next_indices, next_values)
}

fn single_sbox_layer(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;
    let (idx, value) = apply_sbox(builder, state_indices[0], state_values[0]);
    next_indices[0] = idx;
    next_values[0] = value;
    (next_indices, next_values)
}

fn apply_sbox(
    builder: &mut ConstraintBuilder,
    input_idx: usize,
    input_value: BabyBear,
) -> (usize, BabyBear) {
    let x2 = input_value * input_value;
    let x2_idx = builder.alloc(x2);
    builder.add_mul_constraint(input_idx, input_idx, x2_idx);

    let x4 = x2 * x2;
    let x4_idx = builder.alloc(x4);
    builder.add_mul_constraint(x2_idx, x2_idx, x4_idx);

    let x6 = x4 * x2;
    let x6_idx = builder.alloc(x6);
    builder.add_mul_constraint(x4_idx, x2_idx, x6_idx);

    let x7 = x6 * input_value;
    let out_idx = builder.alloc(x7);
    builder.add_mul_constraint(x6_idx, input_idx, out_idx);

    (out_idx, x7)
}

fn external_linear_layer(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mat = external_mds_matrix();

    let mut block_indices = [[0usize; 4]; 4];
    let mut block_values = [[BabyBear::ZERO; 4]; 4];
    for block in 0..4 {
        for lane in 0..4 {
            let mut acc_value = BabyBear::ZERO;
            for k in 0..4 {
                acc_value += mat[lane][k] * state_values[4 * block + k];
            }

            let var_idx = builder.alloc(acc_value);
            let mut terms = Vec::with_capacity(5);
            terms.push((var_idx, BabyBear::NEG_ONE));
            for k in 0..4 {
                let coeff = mat[lane][k];
                if coeff != BabyBear::ZERO {
                    terms.push((state_indices[4 * block + k], coeff));
                }
            }
            builder.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

            block_indices[block][lane] = var_idx;
            block_values[block][lane] = acc_value;
        }
    }

    let mut column_sum_indices = [0usize; 4];
    let mut column_sum_values = [BabyBear::ZERO; 4];
    for lane in 0..4 {
        let mut sum_value = BabyBear::ZERO;
        for block in 0..4 {
            sum_value += block_values[block][lane];
        }

        let sum_idx = builder.alloc(sum_value);
        let mut terms = Vec::with_capacity(5);
        terms.push((sum_idx, BabyBear::NEG_ONE));
        for block in 0..4 {
            terms.push((block_indices[block][lane], BabyBear::ONE));
        }
        builder.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

        column_sum_indices[lane] = sum_idx;
        column_sum_values[lane] = sum_value;
    }

    let mut next_indices = *state_indices;
    let mut next_values = *state_values;
    for block in 0..4 {
        for lane in 0..4 {
            let idx = 4 * block + lane;
            let value = block_values[block][lane] + column_sum_values[lane];
            let var_idx = builder.alloc(value);
            let terms = [
                (var_idx, BabyBear::NEG_ONE),
                (block_indices[block][lane], BabyBear::ONE),
                (column_sum_indices[lane], BabyBear::ONE),
            ];
            builder.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

            next_indices[idx] = var_idx;
            next_values[idx] = value;
        }
    }

    (next_indices, next_values)
}

fn internal_linear_layer(
    builder: &mut ConstraintBuilder,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let diag = internal_diag();
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    let mut sum_value = BabyBear::ZERO;
    for value in state_values.iter() {
        sum_value += *value;
    }

    let sum_idx = builder.alloc(sum_value);
    let mut sum_terms = Vec::with_capacity(WIDTH + 1);
    sum_terms.push((sum_idx, BabyBear::NEG_ONE));
    for &idx in state_indices.iter() {
        sum_terms.push((idx, BabyBear::ONE));
    }
    builder.enforce_linear_relation(&sum_terms, BabyBear::ZERO, one_index);

    for lane in 0..WIDTH {
        let updated = sum_value + state_values[lane] * diag[lane];
        let var_idx = builder.alloc(updated);
        let state_idx = state_indices[lane];
        let terms = [
            (var_idx, BabyBear::ONE),
            (sum_idx, BabyBear::NEG_ONE),
            (state_idx, BabyBear::NEG_ONE * diag[lane]),
        ];
        builder.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

        next_indices[lane] = var_idx;
        next_values[lane] = updated;
    }

    (next_indices, next_values)
}

fn external_mds_matrix() -> [[BabyBear; 4]; 4] {
    let one = BabyBear::ONE;
    let two = BabyBear::TWO;
    let three = two + one;
    [
        [two, three, one, one],
        [one, two, three, one],
        [one, one, two, three],
        [three, one, one, two],
    ]
}

fn internal_diag() -> [BabyBear; WIDTH] {
    let one = BabyBear::ONE;
    let two = BabyBear::TWO;
    let neg_one = BabyBear::NEG_ONE;
    let three = two + one;
    let four = two + two;
    let half = one.div_2exp_u64(1);
    let quarter = half * half;
    let eighth = quarter * half;
    let sixteenth = eighth * half;
    let inv_2_pow_8 = one.div_2exp_u64(8);
    let inv_2_pow_27 = one.div_2exp_u64(27);

    [
        neg_one * two, // -2
        one,           // 1
        two,           // 2
        half,          // 1/2
        three,         // 3
        four,          // 4
        -half,         // -1/2
        -three,        // -3
        -four,         // -4
        inv_2_pow_8,   // 1/2^8
        quarter,       // 1/4
        eighth,        // 1/8
        inv_2_pow_27,  // 1/2^27
        -inv_2_pow_8,  // -1/2^8
        -sixteenth,    // -1/16
        -inv_2_pow_27, // -1/2^27
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::integers::QuotientMap;
    use proptest::prelude::*;

    #[test]
    fn poseidon2_roundtrip_matches_reference() {
        let poseidon = default_babybear_poseidon2_16();
        let rate = [BabyBear::ONE, BabyBear::TWO];
        let capacity = [BabyBear::ZERO; WIDTH - RATE];
        let instance = build_poseidon2_instance(&rate, Some(&capacity), &poseidon)
            .expect("poseidon2 instance construction should succeed");

        let mut expected = [BabyBear::ZERO; WIDTH];
        expected[..RATE].copy_from_slice(&rate);
        expected[RATE..].copy_from_slice(&capacity);
        let mut expected_perm = expected;
        poseidon.permute_mut(&mut expected_perm);

        assert_eq!(expected_perm[..], instance.witness.final_state[..]);
        assert_eq!(&expected_perm[..RATE], &instance.witness.digest);

        let verified = instance
            .r1cs
            .verify(&instance.witness.witness.to_mle())
            .expect("verification should succeed");
        assert!(verified);
    }

    #[test]
    fn poseidon2_random_input_is_consistent() {
        use rand::{Rng, SeedableRng};

        let poseidon = default_babybear_poseidon2_16();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xdeadbeef);
        let mut rate = [BabyBear::ZERO; RATE];
        for lane in rate.iter_mut() {
            *lane = BabyBear::from_int(rng.r#gen::<u32>());
        }
        let mut capacity = [BabyBear::ZERO; WIDTH - RATE];
        for lane in capacity.iter_mut() {
            *lane = BabyBear::from_int(rng.r#gen::<u32>());
        }

        let instance = build_poseidon2_instance(&rate, Some(&capacity), &poseidon)
            .expect("poseidon2 instance construction should succeed");
        let verified = instance
            .r1cs
            .verify(&instance.witness.witness.to_mle())
            .expect("verification should succeed");
        assert!(verified);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]
        #[test]
        fn poseidon2_random_state_matches_reference(state in proptest::array::uniform16(any::<u32>())) {
            let poseidon = default_babybear_poseidon2_16();

            let mut initial_state = [BabyBear::ZERO; WIDTH];
            for (dst, value) in initial_state.iter_mut().zip(state.into_iter()) {
                *dst = BabyBear::from_int(value);
            }

            let mut rate = [BabyBear::ZERO; RATE];
            rate.copy_from_slice(&initial_state[..RATE]);
            let mut capacity = [BabyBear::ZERO; WIDTH - RATE];
            capacity.copy_from_slice(&initial_state[RATE..]);

            let instance = build_poseidon2_instance(&rate, Some(&capacity), &poseidon)
                .expect("poseidon2 instance construction should succeed");

            let mut expected = initial_state;
            poseidon.permute_mut(&mut expected);

            prop_assert_eq!(expected, instance.witness.final_state);
            let verified = instance.r1cs
                .verify(&instance.witness.witness.to_mle())
                .expect("verification should succeed");
            prop_assert!(verified);
        }
    }
}
