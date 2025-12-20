# Poseidon2 R1CS Plan Summary

## Scope & Objectives
- Target permutation: Poseidon2 over the BabyBear field with width 16, mirroring Plonky3's configuration.
- Deliver an explicit Spartan-compatible R1CS plus helper APIs for witness assembly and public digest extraction.
- Keep documentation and tests aligned with the Plonky3 reference to guarantee interoperability.

## Permutation Parameters
- Round schedule: initial external linear layer → 4 external rounds → 13 internal rounds → 4 final external rounds.
- S-box: degree-7 power map `x -> x^7` realized via a four-step multiplication chain (`x2`, `x4`, `x6`, `x7`).
- Constants: use Horizon Labs vectors exposed as `BABYBEAR_RC16_EXTERNAL_INITIAL`, `BABYBEAR_RC16_EXTERNAL_FINAL`, and `BABYBEAR_RC16_INTERNAL` from `p3_baby_bear`.
- Linear layers:
  - External: apply the fixed 4×4 MDS matrix within each 4-lane block, then add column sums across blocks (the "mds_light" permutation).
  - Internal: compute the total state sum and add diagonal-scaled contributions using the BabyBear-specific vector `[-2, 1, 2, 1/2, …, -1/2^27]`.

## Constraint System Blueprint
- Public inputs: full 16-lane initial state, plus a dedicated constant-one wire enforced via a single multiplication constraint.
- S-box constraints: four multiplicative rows per invocation to build `x^7`; external rounds wire all 16 lanes, internal rounds wire lane 0 only.
- Round constants: linear constraints asserting `state_after = state_before + rc` using the constant-one slot.
- External linear layers: three-stage gadget—block MDS outputs, column-sum accumulators, and final additions producing the next state snapshot.
- Internal linear layers: allocate a sum accumulator, then enforce `state'[i] = sum + diag[i] * state[i]` with linear equations.
- Matrices (current impl): non-zero counts `A=3165`, `B=1251`, `C=565`; dimensions padded to the next power of two by `SparseMLE`.

## Witness Generation Workflow
- Assemble the initial state from rate (≤2 elements) and optional capacity (14 elements, default zero).
- Apply each permutation step, pushing intermediate variables (S-box chain values, block outputs, column sums, internal sums) into the witness vector in lockstep with constraint indices.
- Maintain layout metadata (`Poseidon2Layout`) so callers can locate public inputs, final state lanes, and the constant wire.
- Final witness exposes the complete assignment, the digest (lanes 0–1), and the final 16-lane state for chaining.

## Validation Strategy
- Deterministic test: feed `[1, 2]` + zero capacity, compare the R1CS witness output against `Poseidon2BabyBear<16>` and re-run `R1CS::verify`.
- Randomized checks: seeded RNG unit test plus a `proptest` fuzz harness sampling 16-lane states to ensure every witness matches the reference permutation and satisfies the constraint system.
- Debug guidance: dump intermediate round states and inspect `(A·z, B·z, C·z)` when mismatches arise; keep documentation in sync with any parameter updates from Plonky3.

## Current Status (2025-09-23)
- External linear layer gadget rewritten to match the documented block-MDS and column-sum structure without double counting.
- Internal linear layer now uses an explicit sum accumulator and correct diagonal coefficients.
- Unit tests (`poseidon2_roundtrip_matches_reference`, `poseidon2_random_input_is_consistent`) and property test (`poseidon2_random_state_matches_reference`) pass under `cargo test poseidon2`.
- Witness vector padded to the next power-of-two length to respect `SparseMLE` expectations.

## Follow-Up Items
- Publish offset metadata for linear-layer scratch variables and S-box auxiliary slots to aid downstream gadgets.
- Decide whether to explicitly pad constraint matrices (rows/columns) beyond `SparseMLE`'s inferred dimensions for clearer size contracts.
- Integrate the Poseidon2 instance into higher-level Spartan flows (e.g., prover harnesses) once the wiring stabilizes.
