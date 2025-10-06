use crate::spartan::error::{SparseError, SparseResult};
use crate::spartan::r1cs::{R1CS, R1CSInstance, Witness};
use crate::utils::sparse::SparseMLE;
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

/// Column seed describing the absorbed rate and optional capacity lanes for a single Poseidon2 hash.
#[derive(Debug, Clone)]
pub struct Poseidon2ColumnSeed {
    /// Elements absorbed into the rate portion (lanes 0 and 1). Length must be ≤ RATE.
    pub rate: Vec<BabyBear>,
    /// Optional capacity elements populating lanes 2..15. Defaults to zero when omitted.
    pub capacity: Option<[BabyBear; WIDTH - RATE]>,
}

/// Column-major matrix of Poseidon2 witnesses where each column represents an independent permutation trace.
#[derive(Debug, Clone)]
pub struct Poseidon2WitnessMatrix {
    /// Flattened column-major storage of witness assignments.
    pub assignments: Vec<BabyBear>,
    /// Number of witness variables per column after padding to a power of two.
    pub column_len: usize,
    /// Number of Poseidon2 witnesses stored in the matrix.
    pub num_columns: usize,
    /// Number of public inputs at the start of each column (always WIDTH).
    pub num_public_inputs: usize,
    /// Shared layout metadata describing variable positions.
    pub layout: Poseidon2Layout,
    /// Final Poseidon2 states (all 16 lanes) for every column.
    pub final_states: Vec<[BabyBear; WIDTH]>,
    /// Rate digests extracted from lanes 0 and 1 for every column.
    pub digests: Vec<[BabyBear; RATE]>,
}

impl Poseidon2WitnessMatrix {
    /// Returns true when no witness columns are stored.
    pub fn is_empty(&self) -> bool {
        self.num_columns == 0
    }

    /// Returns a slice view over the requested column, if it exists.
    pub fn column_slice(&self, column: usize) -> Option<&[BabyBear]> {
        if column >= self.num_columns {
            return None;
        }
        let start = column * self.column_len;
        let end = start + self.column_len;
        self.assignments.get(start..end)
    }

    /// Materialises a Spartan witness for the requested column.
    pub fn column_witness(&self, column: usize) -> Option<Witness> {
        let column_slice = self.column_slice(column)?;
        Some(Witness::from_vec(
            column_slice.to_vec(),
            self.num_public_inputs,
        ))
    }

    /// Returns a flattened column-major representation of the transpose of the
    /// witness matrix. If the matrix is empty, an empty vector is returned.
    pub fn flattened_transpose(&self) -> Vec<BabyBear> {
        if self.assignments.is_empty() {
            return Vec::new();
        }

        let rows = self.column_len;
        let cols = self.num_columns;
        let mut transposed = vec![BabyBear::ZERO; self.assignments.len()];

        for col in 0..cols {
            let start = col * rows;
            for row in 0..rows {
                let value = self.assignments[start + row];
                let index = row * cols + col;
                transposed[index] = value;
            }
        }

        transposed
    }
}

/// A ready-to-use R1CS instance enforcing the Poseidon2 permutation over BabyBear.
#[derive(Debug, Clone)]
pub struct Poseidon2Instance {
    pub r1cs: R1CS,
    pub witness: Poseidon2Witness,
    pub witness_matrix: Poseidon2WitnessMatrix,
}

impl Poseidon2Instance {
    /// Converts the Poseidon2 instance into a generic Spartan R1CS instance.
    pub fn to_r1cs_instance(&self) -> SparseResult<R1CSInstance> {
        R1CSInstance::new(self.r1cs.clone(), self.witness.witness.clone())
    }
}

trait Poseidon2Backend {
    fn alloc(&mut self, value: BabyBear) -> usize;
    fn enforce_constant_one(&mut self, idx: usize);
    fn add_mul_constraint(&mut self, left: usize, right: usize, out: usize);
    fn enforce_linear_relation(
        &mut self,
        terms: &[(usize, BabyBear)],
        constant: BabyBear,
        one_index: usize,
    );
}

struct AssignmentCollector {
    assignment: Vec<BabyBear>,
}

impl AssignmentCollector {
    fn new(initial_assignment: Vec<BabyBear>) -> Self {
        Self {
            assignment: initial_assignment,
        }
    }

    fn into_assignment(self) -> Vec<BabyBear> {
        self.assignment
    }

    fn len(&self) -> usize {
        self.assignment.len()
    }
}

impl Poseidon2Backend for AssignmentCollector {
    fn alloc(&mut self, value: BabyBear) -> usize {
        let idx = self.assignment.len();
        self.assignment.push(value);
        idx
    }

    fn enforce_constant_one(&mut self, _idx: usize) {}

    fn add_mul_constraint(&mut self, _left: usize, _right: usize, _out: usize) {}

    fn enforce_linear_relation(
        &mut self,
        _terms: &[(usize, BabyBear)],
        _constant: BabyBear,
        _one_index: usize,
    ) {
    }
}

struct Poseidon2WitnessArtifacts {
    layout: Poseidon2Layout,
    final_state: [BabyBear; WIDTH],
    digest: [BabyBear; RATE],
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
    let initial_state = assemble_initial_state(rate, capacity)?;
    let mut builder = ConstraintBuilder::new(initial_state.to_vec());
    let artifacts = generate_poseidon2_witness(&mut builder, &initial_state, poseidon)?;
    let num_public_inputs = WIDTH;

    let padded_len = builder.assignment.len().next_power_of_two();
    if builder.assignment.len() < padded_len {
        builder.assignment.resize(padded_len, BabyBear::ZERO);
    }

    let full_assignment = builder.assignment.clone();
    let witness = Witness::from_vec(full_assignment.clone(), num_public_inputs);

    let poseidon_witness = Poseidon2Witness {
        witness,
        full_assignment,
        final_state: artifacts.final_state,
        digest: artifacts.digest,
        layout: artifacts.layout.clone(),
    };

    let a = SparseMLE::new(builder.a)?;
    let b = SparseMLE::new(builder.b)?;
    let c = SparseMLE::new(builder.c)?;
    let r1cs = R1CS::new(a, b, c, num_public_inputs)?;

    let witness_matrix = Poseidon2WitnessMatrix {
        assignments: poseidon_witness.full_assignment.clone(),
        column_len: padded_len,
        num_columns: 1,
        num_public_inputs,
        layout: artifacts.layout,
        final_states: vec![poseidon_witness.final_state],
        digests: vec![poseidon_witness.digest],
    };

    Ok(Poseidon2Instance {
        r1cs,
        witness: poseidon_witness,
        witness_matrix,
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

fn assemble_initial_state(
    rate: &[BabyBear],
    capacity: Option<&[BabyBear; WIDTH - RATE]>,
) -> SparseResult<[BabyBear; WIDTH]> {
    if rate.len() > RATE {
        return Err(SparseError::ValidationError(
            "rate must contain at most two elements".to_string(),
        ));
    }

    let mut state = [BabyBear::ZERO; WIDTH];
    state[..rate.len()].copy_from_slice(rate);
    if let Some(capacity) = capacity {
        state[RATE..].copy_from_slice(capacity);
    }
    Ok(state)
}

fn generate_poseidon2_witness<B: Poseidon2Backend>(
    backend: &mut B,
    initial_state: &[BabyBear; WIDTH],
    poseidon: &Poseidon2BabyBear<WIDTH>,
) -> SparseResult<Poseidon2WitnessArtifacts> {
    let mut state_indices = core::array::from_fn(|i| i);
    let mut state_values = *initial_state;

    let one_index = backend.alloc(BabyBear::ONE);
    backend.enforce_constant_one(one_index);

    let (state_indices_tmp, state_values_tmp) =
        external_linear_layer(backend, &state_indices, &state_values, one_index);
    state_indices = state_indices_tmp;
    state_values = state_values_tmp;

    for round in 0..EXTERNAL_ROUNDS {
        let constants = BABYBEAR_RC16_EXTERNAL_INITIAL
            .get(round)
            .expect("missing initial round constant");
        let (indices_after_const, values_after_const) =
            add_round_constants(backend, &state_indices, &state_values, constants, one_index);
        let (indices_after_sbox, values_after_sbox) =
            full_sbox_layer(backend, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) =
            external_linear_layer(backend, &indices_after_sbox, &values_after_sbox, one_index);
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    for round in 0..INTERNAL_ROUNDS {
        let rc = BABYBEAR_RC16_INTERNAL
            .get(round)
            .expect("missing internal round constant");
        let (indices_after_const, values_after_const) =
            add_internal_constant(backend, &state_indices, &state_values, *rc, one_index);
        let (indices_after_sbox, values_after_sbox) =
            single_sbox_layer(backend, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) =
            internal_linear_layer(backend, &indices_after_sbox, &values_after_sbox, one_index);
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    for round in 0..EXTERNAL_ROUNDS {
        let constants = BABYBEAR_RC16_EXTERNAL_FINAL
            .get(round)
            .expect("missing terminal round constant");
        let (indices_after_const, values_after_const) =
            add_round_constants(backend, &state_indices, &state_values, constants, one_index);
        let (indices_after_sbox, values_after_sbox) =
            full_sbox_layer(backend, &indices_after_const, &values_after_const);
        let (indices_after_linear, values_after_linear) =
            external_linear_layer(backend, &indices_after_sbox, &values_after_sbox, one_index);
        state_indices = indices_after_linear;
        state_values = values_after_linear;
    }

    let mut expected_state = *initial_state;
    poseidon.permute_mut(&mut expected_state);
    if expected_state != state_values {
        return Err(SparseError::ValidationError(
            "Poseidon2 wiring mismatch compared to reference permutation".to_string(),
        ));
    }

    let layout = Poseidon2Layout {
        public_input_positions: (0..WIDTH).collect(),
        one_index,
        final_state_positions: state_indices,
    };

    Ok(Poseidon2WitnessArtifacts {
        layout,
        final_state: state_values,
        digest: [state_values[0], state_values[1]],
    })
}

/// Builds a column-major matrix where each column encodes the witness for an independent Poseidon2 hash.
pub fn build_poseidon2_witness_matrix(
    seeds: &[Poseidon2ColumnSeed],
    poseidon: &Poseidon2BabyBear<WIDTH>,
) -> SparseResult<Poseidon2WitnessMatrix> {
    if seeds.is_empty() {
        return Err(SparseError::ValidationError(
            "at least one Poseidon2 column seed is required".to_string(),
        ));
    }

    let mut initial_states = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let state = assemble_initial_state(seed.rate.as_slice(), seed.capacity.as_ref())?;
        initial_states.push(state);
    }

    build_poseidon2_witness_matrix_from_states(&initial_states, poseidon)
}

/// Builds a Poseidon2 witness matrix from full 16-lane initial states.
///
/// Each entry in `initial_states` is treated as the public input state for an independent
/// permutation. All columns share an identical constraint layout; the builder validates this
/// invariant, pads every column to the same power-of-two `column_len`, and stores the witness
/// assignments in column-major order. Returns an error when `initial_states` is empty or when a
/// column produces a layout/length that differs from the first column.
pub fn build_poseidon2_witness_matrix_from_states(
    initial_states: &[[BabyBear; WIDTH]],
    poseidon: &Poseidon2BabyBear<WIDTH>,
) -> SparseResult<Poseidon2WitnessMatrix> {
    if initial_states.is_empty() {
        return Err(SparseError::ValidationError(
            "at least one Poseidon2 initial state is required".to_string(),
        ));
    }

    let mut builder = ConstraintBuilder::new(initial_states[0].to_vec());
    let base_artifacts = generate_poseidon2_witness(&mut builder, &initial_states[0], poseidon)?;
    let Poseidon2WitnessArtifacts {
        layout: base_layout,
        final_state: first_final_state,
        digest: first_digest,
    } = base_artifacts;
    let raw_column_len = builder.assignment.len();
    let column_len = raw_column_len.next_power_of_two();
    if builder.assignment.len() < column_len {
        builder.assignment.resize(column_len, BabyBear::ZERO);
    }

    let mut assignments = Vec::with_capacity(column_len * initial_states.len());
    assignments.extend_from_slice(&builder.assignment);

    let mut final_states = Vec::with_capacity(initial_states.len());
    final_states.push(first_final_state);

    let mut digests = Vec::with_capacity(initial_states.len());
    digests.push(first_digest);

    for initial_state in initial_states.iter().skip(1) {
        let mut collector = AssignmentCollector::new(initial_state.to_vec());
        let artifacts = generate_poseidon2_witness(&mut collector, initial_state, poseidon)?;

        if collector.len() != raw_column_len {
            return Err(SparseError::ValidationError(
                "Poseidon2 witness column length mismatch".to_string(),
            ));
        }

        let mut column = collector.into_assignment();
        column.resize(column_len, BabyBear::ZERO);
        assignments.extend_from_slice(&column);

        if artifacts.layout.final_state_positions != base_layout.final_state_positions
            || artifacts.layout.one_index != base_layout.one_index
            || artifacts.layout.public_input_positions != base_layout.public_input_positions
        {
            return Err(SparseError::ValidationError(
                "Poseidon2 layout mismatch across witness columns".to_string(),
            ));
        }

        final_states.push(artifacts.final_state);
        digests.push(artifacts.digest);
    }

    Ok(Poseidon2WitnessMatrix {
        assignments,
        column_len,
        num_columns: initial_states.len(),
        num_public_inputs: WIDTH,
        layout: base_layout,
        final_states,
        digests,
    })
}

/// Convenience wrapper that wires the default BabyBear Poseidon2 permutation into
/// [`build_poseidon2_witness_matrix_from_states`].
pub fn build_default_poseidon2_witness_matrix_from_states(
    initial_states: &[[BabyBear; WIDTH]],
) -> SparseResult<Poseidon2WitnessMatrix> {
    let poseidon = default_babybear_poseidon2_16();
    build_poseidon2_witness_matrix_from_states(initial_states, &poseidon)
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

impl Poseidon2Backend for ConstraintBuilder {
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
}

fn add_round_constants(
    backend: &mut impl Poseidon2Backend,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    constants: &[BabyBear; WIDTH],
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    for lane in 0..WIDTH {
        let updated = state_values[lane] + constants[lane];
        let idx = backend.alloc(updated);
        let terms = [
            (idx, BabyBear::NEG_ONE),
            (state_indices[lane], BabyBear::ONE),
        ];
        backend.enforce_linear_relation(&terms, constants[lane], one_index);
        next_indices[lane] = idx;
        next_values[lane] = updated;
    }

    (next_indices, next_values)
}

fn add_internal_constant(
    backend: &mut impl Poseidon2Backend,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
    constant: BabyBear,
    one_index: usize,
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    let updated = state_values[0] + constant;
    let idx = backend.alloc(updated);
    let terms = [(idx, BabyBear::NEG_ONE), (state_indices[0], BabyBear::ONE)];
    backend.enforce_linear_relation(&terms, constant, one_index);
    next_indices[0] = idx;
    next_values[0] = updated;

    (next_indices, next_values)
}

fn full_sbox_layer(
    backend: &mut impl Poseidon2Backend,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;

    for lane in 0..WIDTH {
        let (idx, value) = apply_sbox(backend, state_indices[lane], state_values[lane]);
        next_indices[lane] = idx;
        next_values[lane] = value;
    }

    (next_indices, next_values)
}

fn single_sbox_layer(
    backend: &mut impl Poseidon2Backend,
    state_indices: &[usize; WIDTH],
    state_values: &[BabyBear; WIDTH],
) -> ([usize; WIDTH], [BabyBear; WIDTH]) {
    let mut next_indices = *state_indices;
    let mut next_values = *state_values;
    let (idx, value) = apply_sbox(backend, state_indices[0], state_values[0]);
    next_indices[0] = idx;
    next_values[0] = value;
    (next_indices, next_values)
}

fn apply_sbox(
    backend: &mut impl Poseidon2Backend,
    input_idx: usize,
    input_value: BabyBear,
) -> (usize, BabyBear) {
    let x2 = input_value * input_value;
    let x2_idx = backend.alloc(x2);
    backend.add_mul_constraint(input_idx, input_idx, x2_idx);

    let x4 = x2 * x2;
    let x4_idx = backend.alloc(x4);
    backend.add_mul_constraint(x2_idx, x2_idx, x4_idx);

    let x6 = x4 * x2;
    let x6_idx = backend.alloc(x6);
    backend.add_mul_constraint(x4_idx, x2_idx, x6_idx);

    let x7 = x6 * input_value;
    let out_idx = backend.alloc(x7);
    backend.add_mul_constraint(x6_idx, input_idx, out_idx);

    (out_idx, x7)
}

fn external_linear_layer(
    backend: &mut impl Poseidon2Backend,
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

            let var_idx = backend.alloc(acc_value);
            let mut terms = Vec::with_capacity(5);
            terms.push((var_idx, BabyBear::NEG_ONE));
            for k in 0..4 {
                let coeff = mat[lane][k];
                if coeff != BabyBear::ZERO {
                    terms.push((state_indices[4 * block + k], coeff));
                }
            }
            backend.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

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

        let sum_idx = backend.alloc(sum_value);
        let mut terms = Vec::with_capacity(5);
        terms.push((sum_idx, BabyBear::NEG_ONE));
        for block in 0..4 {
            terms.push((block_indices[block][lane], BabyBear::ONE));
        }
        backend.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

        column_sum_indices[lane] = sum_idx;
        column_sum_values[lane] = sum_value;
    }

    let mut next_indices = *state_indices;
    let mut next_values = *state_values;
    for block in 0..4 {
        for lane in 0..4 {
            let idx = 4 * block + lane;
            let value = block_values[block][lane] + column_sum_values[lane];
            let var_idx = backend.alloc(value);
            let terms = [
                (var_idx, BabyBear::NEG_ONE),
                (block_indices[block][lane], BabyBear::ONE),
                (column_sum_indices[lane], BabyBear::ONE),
            ];
            backend.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

            next_indices[idx] = var_idx;
            next_values[idx] = value;
        }
    }

    (next_indices, next_values)
}

fn internal_linear_layer(
    backend: &mut impl Poseidon2Backend,
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

    let sum_idx = backend.alloc(sum_value);
    let mut sum_terms = Vec::with_capacity(WIDTH + 1);
    sum_terms.push((sum_idx, BabyBear::NEG_ONE));
    for &idx in state_indices.iter() {
        sum_terms.push((idx, BabyBear::ONE));
    }
    backend.enforce_linear_relation(&sum_terms, BabyBear::ZERO, one_index);

    for lane in 0..WIDTH {
        let updated = sum_value + state_values[lane] * diag[lane];
        let var_idx = backend.alloc(updated);
        let state_idx = state_indices[lane];
        let terms = [
            (var_idx, BabyBear::ONE),
            (sum_idx, BabyBear::NEG_ONE),
            (state_idx, BabyBear::NEG_ONE * diag[lane]),
        ];
        backend.enforce_linear_relation(&terms, BabyBear::ZERO, one_index);

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

    #[test]
    fn poseidon2_witness_matrix_generates_independent_columns() {
        let poseidon = default_babybear_poseidon2_16();

        let mut capacity0 = [BabyBear::ZERO; WIDTH - RATE];
        capacity0[0] = BabyBear::from_int(17);
        capacity0[3] = BabyBear::from_int(5);

        let seeds = vec![
            Poseidon2ColumnSeed {
                rate: vec![BabyBear::ONE, BabyBear::TWO],
                capacity: Some(capacity0),
            },
            Poseidon2ColumnSeed {
                rate: vec![BabyBear::TWO],
                capacity: None,
            },
        ];

        let matrix = build_poseidon2_witness_matrix(&seeds, &poseidon)
            .expect("matrix generation should succeed");

        assert_eq!(matrix.num_columns, seeds.len());
        assert_eq!(matrix.num_public_inputs, WIDTH);
        assert_eq!(
            matrix.assignments.len(),
            matrix.column_len * matrix.num_columns
        );

        let instance0 = build_poseidon2_instance(
            seeds[0].rate.as_slice(),
            seeds[0].capacity.as_ref(),
            &poseidon,
        )
        .expect("first column witness generation should succeed");
        let instance1 = build_poseidon2_instance(
            seeds[1].rate.as_slice(),
            seeds[1].capacity.as_ref(),
            &poseidon,
        )
        .expect("second column witness generation should succeed");

        let column0 = matrix.column_slice(0).expect("column 0 should exist");
        let column1 = matrix.column_slice(1).expect("column 1 should exist");

        assert_eq!(column0, instance0.witness.full_assignment.as_slice());
        assert_eq!(column1, instance1.witness.full_assignment.as_slice());

        assert_eq!(matrix.final_states[0], instance0.witness.final_state);
        assert_eq!(matrix.final_states[1], instance1.witness.final_state);
        assert_eq!(matrix.digests[0], instance0.witness.digest);
        assert_eq!(matrix.digests[1], instance1.witness.digest);

        let witness0 = matrix
            .column_witness(0)
            .expect("should construct witness for first column");
        assert_eq!(witness0, instance0.witness.witness);

        let row = 3;
        assert_eq!(matrix.assignments[matrix.column_len + row], column1[row]);

        assert!(matrix.column_slice(2).is_none());
    }

    #[test]
    fn poseidon2_witness_matrix_flattened_transpose_matches_manual() {
        let poseidon = default_babybear_poseidon2_16();
        let seeds = vec![Poseidon2ColumnSeed {
            rate: vec![BabyBear::ONE, BabyBear::from_u32(2)],
            capacity: None,
        }];

        let matrix = build_poseidon2_witness_matrix(&seeds, &poseidon)
            .expect("matrix generation should succeed");

        let rows = matrix.column_len;
        let cols = matrix.num_columns;

        let mut expected = vec![BabyBear::ZERO; matrix.assignments.len()];
        for col in 0..cols {
            for row in 0..rows {
                let value = matrix.assignments[col * rows + row];
                expected[row * cols + col] = value;
            }
        }

        assert_eq!(matrix.flattened_transpose(), expected);
    }

    #[test]
    fn poseidon2_witness_matrix_from_states_multi_state_smoke() {
        let mut state0 = [BabyBear::ZERO; WIDTH];
        state0[0] = BabyBear::ONE;
        state0[1] = BabyBear::TWO;

        let mut state1 = [BabyBear::ZERO; WIDTH];
        state1[0] = BabyBear::from_int(3);
        state1[5] = BabyBear::from_int(7);

        let states = vec![state0, state1];
        let matrix = build_default_poseidon2_witness_matrix_from_states(&states)
            .expect("matrix generation should succeed");

        assert_eq!(matrix.num_columns, states.len());
        assert_eq!(matrix.num_public_inputs, WIDTH);
        assert_eq!(
            matrix.assignments.len(),
            matrix.column_len * matrix.num_columns
        );

        for (column, state) in states.iter().enumerate() {
            let slice = matrix
                .column_slice(column)
                .unwrap_or_else(|| panic!("column {column} should exist"));
            assert_eq!(&slice[..WIDTH], &state[..]);

            let witness = matrix
                .column_witness(column)
                .unwrap_or_else(|| panic!("column witness {column} should exist"));
            assert_eq!(witness.public_inputs.len(), WIDTH);
        }
    }

    #[test]
    fn poseidon2_witness_matrix_from_states_matches_single_column() {
        let poseidon = default_babybear_poseidon2_16();

        let mut state0 = [BabyBear::ZERO; WIDTH];
        state0[0] = BabyBear::ONE;
        state0[2] = BabyBear::from_int(9);

        let mut state1 = [BabyBear::ZERO; WIDTH];
        state1[0] = BabyBear::from_int(5);
        state1[1] = BabyBear::from_int(11);
        state1[4] = BabyBear::from_int(13);

        let states = vec![state0, state1];
        let matrix = build_poseidon2_witness_matrix_from_states(&states, &poseidon)
            .expect("matrix generation should succeed");

        let expected_public_inputs: Vec<usize> = (0..WIDTH).collect();
        assert_eq!(matrix.layout.public_input_positions, expected_public_inputs);

        for (column, state) in states.iter().enumerate() {
            let mut rate = [BabyBear::ZERO; RATE];
            rate.copy_from_slice(&state[..RATE]);

            let mut capacity = [BabyBear::ZERO; WIDTH - RATE];
            capacity.copy_from_slice(&state[RATE..]);

            let instance = build_poseidon2_instance(&rate, Some(&capacity), &poseidon)
                .expect("single-column instance should build");

            assert_eq!(matrix.final_states[column], instance.witness.final_state);
            assert_eq!(matrix.digests[column], instance.witness.digest);
            assert_eq!(
                matrix.layout.final_state_positions,
                instance.witness.layout.final_state_positions
            );
            assert_eq!(matrix.layout.one_index, instance.witness.layout.one_index);
            assert_eq!(
                matrix.layout.public_input_positions,
                instance.witness.layout.public_input_positions
            );
        }
    }

    #[test]
    fn poseidon2_witness_matrix_from_states_rejects_empty_input() {
        let poseidon = default_babybear_poseidon2_16();
        let result = build_poseidon2_witness_matrix_from_states(&[], &poseidon);

        assert!(matches!(
            result,
            Err(SparseError::ValidationError(ref msg))
                if msg == "at least one Poseidon2 initial state is required"
        ));
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
