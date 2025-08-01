Implementation plan for converting the sumcheck spec into a concrete engineering plan has been produced, aligned to the repository and the confirmed normative guidance.

Scope and sources
- Authoritative protocol details: Sumcheck_spec.md
- Naming/scope alignment: Plan.md and README.md
- In-repo backends only: src/utils/polynomial.rs and existing field types
- Transcript: src/utils/challenger.rs
- No special perf or environment constraints beyond std

Protocol outline (normative)
- Goal: Prover convinces Verifier that sum_{x in {0,1}^n} C(P(x)) equals claimed T, using round-by-round univariate checks with Fiat–Shamir challenges r_i.
- Primitives: Field F from repo; base polynomials P_i via in-repo Polynomial; composition C via CompositionPolynomial; transcript via Challenger.
- Round i (1..n): Prover sends univariate g_i(t) = sum over remaining variables of C at prefix fixed, Verifier checks degree(g_i) ≤ bound and g_i(0)+g_i(1) equals previous value; then samples r_i from transcript.
- Final: Verify g_n(r_n) equals C evaluated at r = (r_1..r_n).

Repository mapping and required components
- src/sumcheck.rs
  - Public API for prover/ verifier, proof struct, univariate helper, transcript binding.
  - New types:
    - Univariate<F> { coefficients: Vec<F> } with degree(), eval(t: F)->F, sum_ends()->F.
    - SumcheckProof<F> { rounds: Vec<Univariate<F>>, final_eval: F }.
  - New trait:
    - CompositionPolynomial<F> {
        fn degree_bound(&self) -> usize; // or fn round_degree_bound(&self, round: usize) -> usize
        fn evaluate(&self, bases: &[&dyn Polynomial<F>], point: &[F]) -> F;
      }
  - Helpers:
    - absorb_univariate(challenger, round_index, &Univariate<F>) to bind in transcript.
    - construct_round_univariate(...) for building g_i from C and bases.
- src/utils/polynomial.rs
  - Ensure Polynomial<F> supports num_vars() and evaluate(point: &[F]).
  - If not present, add or extend to support generic evaluation at arbitrary F^n points.
- src/utils/challenger.rs
  - Must support absorbing field elements or bytes deterministically and sampling field elements: sample_field() -> F; absorb_field(&F) or equivalent.
- src/prover.rs
  - Integrate sumcheck proof generation as needed; keep protocol logic in sumcheck.rs.
- src/lib.rs
  - Re-exports for SumcheckProof, Univariate, CompositionPolynomial, prove_sumcheck, verify_sumcheck.

Proposed public APIs
- In src/sumcheck.rs
  - sumcheck.rs:pub struct Univariate<F> { pub coefficients: Vec<F> }
  - sumcheck.rs:impl Univariate<F> { pub fn degree(&self)->usize; pub fn eval(&self, t: F)->F; pub fn sum_ends(&self)->F }
  - sumcheck.rs:pub struct SumcheckProof<F> { pub rounds: Vec<Univariate<F>>, pub final_eval: F }
  - sumcheck.rs:pub trait CompositionPolynomial<F> {
      fn degree_bound(&self) -> usize;
      fn evaluate(&self, bases: &[&dyn Polynomial<F>], point: &[F]) -> F;
    }
  - sumcheck.rs:pub fn prove_sumcheck<F: Field>(
        composition: &impl CompositionPolynomial<F>,
        bases: &[&dyn Polynomial<F>],
        n_vars: usize,
        claimed_sum: F,
        challenger: &mut Challenger<F>,
     ) -> SumcheckProof<F>
  - sumcheck.rs:pub fn verify_sumcheck<F: Field>(
        composition: &impl CompositionPolynomial<F>,
        bases: &[&dyn Polynomial<F>],
        n_vars: usize,
        claimed_sum: F,
        proof: &SumcheckProof<F>,
        challenger: &mut Challenger<F>,
     ) -> bool
- Notes:
  - Consider Result-returning variants for better error diagnostics:
    - verify_sumcheck(...) -> Result<(), VerifyError>
    - prove_sumcheck(...) -> Result<SumcheckProof<F>, ProverError>

Prover algorithm (degree_bound 1 baseline, extensible)
- Inputs: C, bases, n, claimed_sum T, challenger
- State: prev_value = T; r_prefix = []
- For i in 1..=n:
  1) Compute evaluations g_i(0) and g_i(1) by summing over all assignments of remaining n-i Boolean variables and calling C.evaluate(bases, point).
  2) If degree_bound == 1, construct linear Univariate from these two evaluations; else evaluate at degree_bound+1 distinct points and interpolate to coefficients.
  3) Assert (debug) deg ≤ degree_bound; absorb_univariate(challenger, i, &g_i).
  4) r_i = challenger.sample_field()
  5) Check internally g_i(0)+g_i(1) == prev_value; update prev_value = g_i(r_i); push r_i.
- After loop: final_eval = C.evaluate(bases, r_prefix); return SumcheckProof { rounds, final_eval }.

Verifier algorithm
- Inputs: C, bases, n, claimed_sum T, proof, challenger
- State: prev_value = T; r = []
- For i in 1..=n:
  1) Check degree(g_i) ≤ bound; check g_i(0)+g_i(1) == prev_value
  2) absorb_univariate(challenger, i, &g_i); r_i = challenger.sample_field(); prev_value = g_i(r_i); push r_i
- After loop: check prev_value == proof.final_eval; compute c_eval = C.evaluate(bases, r); accept if equal.

Interpolation for general degree
- Use degree_bound()+1 distinct points (e.g., 0,1,2,... or field-specific) to evaluate g_i(t) and interpolate to coefficients.
- Provide an internal helper interpolate_univariate(points: &[F], values: &[F]) -> Univariate<F>; add unit tests.

Transcript binding details
- For each round, absorb:
  - domain separator bytes for “sumcheck_round”
  - round index as u32
  - degree as u32
  - coefficients as field elements via absorb_field
- Derive challenge with sample_field thereafter.

Testing strategy
- Unit tests:
  - Univariate construction, eval, sum_ends
  - Interpolation round-trip for random polynomials up to small degree
  - CompositionPolynomial mock implementations: sum, product, affine comb; degree_bound 1 and 2
  - Polynomial MLEs with small n
- Integration tests:
  - End-to-end prove/verify for n=1..5, degree_bound 1 compositions
  - Tamper tests: degree violation; sum consistency failure; final_eval mismatch
  - Transcript determinism: same inputs produce same r_i; any coefficient change triggers rejection
- Optional property tests with seeded challenger for reproducibility

Milestones and acceptance criteria
- M1: Types and trait scaffolding compile (Univariate, SumcheckProof, CompositionPolynomial); re-export in lib
- M2: Verifier with checks and transcript binding; unit tests for Univariate and basic checks
- M3: Prover for degree_bound 1; end-to-end tests n≤5 pass
- M4: General degree support via interpolation; tests with degree 2+ compositions
- M5: Negative tests and robustness; all tamper cases rejected
- M6: Documentation: rustdoc on public APIs and a short README section describing usage

Deferrals and non-normative parts
- prove_targeted/verify_targeted: deferred until fully specified
- Small-field micro-optimizations and sparse MLE variant: future optimization phases
- PCS commitments integration: out of scope for initial plan; keep clean extension hooks

File to write
- Recommend committing this plan as IMPLEMENTATION_PLAN_SUMCHECK.md at repository root or updating Plan.md with a dedicated Sumcheck section.

This plan is complete and ready for implementation sequencing within the repository.