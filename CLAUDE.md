# Claude Context: Deep-FRI Project

## Project Overview
Deep-FRI is a Rust cryptographic library implementing zero-knowledge proof primitives. It focuses on multilinear extensions (MLEs), polynomial commitments, and Merkle tree constructions for efficient zero-knowledge proof systems.

## Architecture & Structure

### Module Organization
```
src/
├── lib.rs                 # Root module (pub mod prover; pub mod utils;)
├── prover.rs             # WHIR commitment scheme
└── utils/                # Core cryptographic primitives
    ├── mod.rs           # Type aliases: Fp=BabyBear, Fp4=BinomialExtensionField<BabyBear,4>
    ├── challenger.rs    # Fiat-Shamir challenger with BLAKE3
    ├── eq.rs           # Equality polynomial computations
    ├── merkle_tree.rs  # BLAKE3-based Merkle trees
    └── polynomial.rs   # Multilinear extension evaluations
```

### Key Types & Aliases
- `Fp` = `BabyBear` (base field)
- `Fp4` = `BinomialExtensionField<BabyBear, 4>` (4-degree extension)

## Core Components

### 1. WHIR Commitment (`prover.rs`)
- **Struct**: `WHIRCommitment { rate: usize, merkle_tree: MerkleTree, code: Vec<BabyBear> }`
- **Key Method**: `commit(&mut self, domain_evals: &[BabyBear], root_table: Option<&[Vec<BabyBear>]>)`
- Uses FFT operations: `BabyBear::forward_fft()` and `BabyBear::roots_of_unity_table()`

### 2. Equality Polynomials (`utils/eq.rs`)
- **Struct**: `EqEvals<'a> { point: &'a [Fp4], coeffs: Vec<Fp4>, n_vars: usize }`
- **Key Method**: `gen_from_point(point: &'a [Fp4])` - generates coefficients via tensor product expansion
- **Algorithm**: Iterative tensor product construction for equality check polynomials over hypercubes
- **Test Coverage**: Single variable and empty point cases

### 3. Multilinear Extensions (`utils/polynomial.rs`)
- **Struct**: `MLE { coeffs: Vec<BabyBear> }`
- **Constraint**: Coefficient length must be power of two
- **Key Method**: `evaluate(&self, point: &[Fp4]) -> Fp4` - uses EqEvals for efficient evaluation
- **Integration**: Relies on equality polynomials for evaluation

### 4. Merkle Trees (`utils/merkle_tree.rs`)
- **Struct**: `MerkleTree { root: [u8; 32], nodes: Vec<[u8; 32]>, depth: u32 }`
- **Hash Function**: BLAKE3 exclusively
- **Constraint**: Leaf count must be power of two
- **Methods**: `new()`, `get_path()`, `verify_path()`
- **Storage**: Flat vector with leaves first, then layers up to root

### 5. Challenger (`utils/challenger.rs`)
- **Struct**: `Challenger { state: Hasher, round: usize }`
- **Purpose**: Fiat-Shamir transformations for interactive proof protocols
- **Hash Function**: BLAKE3 Hasher
- **Methods**: `observe_field_elem()`, `observe_field_elems()`, `get_challenge()`

## Dependencies & Build
- **p3-baby-bear**: BabyBear finite field implementation
- **p3-field**: Generic field arithmetic traits
- **blake3**: Fast cryptographic hash function (used throughout)
- **anyhow**: Error handling
- **serde**: Serialization framework

**Build Commands**:
- `cargo build --release`
- `cargo test`

## Security & Design Principles
- Power-of-two requirements for cryptographic security
- BLAKE3 hash function used consistently
- Constant-time field operations where supported
- Extension field arithmetic over BabyBear base field
- Memory-efficient polynomial evaluation algorithms

## Current State
- Modified file: `src/utils/eq.rs` (git status shows modifications)
- Recent commits focus on polynomial evaluation and equality polynomial generation
- Comprehensive test coverage for core components
- Well-documented API with usage examples in README

## Development Notes
- All cryptographic operations use BLAKE3
- Power-of-two constraints are enforced throughout
- Extension field support is central to the design
- Test suite includes correctness verification for all major components
- Code follows Rust 2024 edition conventions


# Spartan + WHIR + Memory-Checking – Refined Implementation Plan

*Target implementer: Claude (Rust developer familiar with ZK tooling)*
*Output: Production-ready Rust library & CLI demo*

---

## 0 High-Level Goal

Create a Rust implementation of the Spartan transparent SNARK that:

1. Uses **WHIR** 🌪️ as the multilinear Polynomial Commitment Scheme (PCS).
2. Uses **offline memory-checking arguments** for sparse polynomial openings during Spartan's witness processing.
3. Instantiates all arithmetic over Plonky3's **BabyBear** field  \(\mathbb F_{\text{BB}}\;{=}\;2^{31}-2^{27}+1\).  
4. Runs on **Rust's std::thread** (and `crossbeam` scoped threads) – **NO Rayon** – while retaining deterministic parallelism.
5. Ships as a workspace containing core crates plus examples:
   ```text
   spartan-whir/         # Workspace root
   ├── spartan-core/     # Spartan protocol + sum-check  
   ├── whir-pcs/         # WHIR polynomial commitment scheme
   ├── memory-check/     # Offline memory-checking arguments
   ├── field-utils/      # BabyBear field operations & helpers
   └── examples/         # CLI demo + benchmarks
   ```

---

## 1 Dependencies & External Code

| Area | Crate / Repo | Notes | Priority |
|------|--------------|-------|----------|
|Field | `p3-baby-bear` (Plonky3) | Already integrated. SIMD optional via `avx2`.|⭐⭐⭐|
|Hash  | `blake3` | Already integrated. Fast & post-quantum secure.|⭐⭐⭐|
|FFT   | Built-in to `p3-baby-bear` | Forward/inverse FFT via `BabyBear::forward_fft()`.|⭐⭐⭐|
|Threads | `crossbeam` | For scoped threads; fallback to `std::thread`.|⭐⭐|
|WHIR Base | Fork `github.com/WizardOfMenlo/whir` | Academic prototype - needs production hardening.|⭐⭐⭐|
|Spartan Base | Fork `github.com/microsoft/Spartan` | 8K lines Rust, curve25519 → BabyBear adaptation.|⭐⭐⭐|
|Testing | `proptest`, `criterion` | Property-based testing & benchmarking.|⭐|

---

## 2 Refined Architecture Overview

```
    ┌─────────────────────────────────────────┐
    │  Spartan Prover (spartan-core)          │
    ├─────────────────────────────────────────┤
    │  • R1CS → Constraint matrix encoding    │
    │  • Multilinear extension (MLE) setup    │
    │  • Sum-check protocol execution         │
    │  • Sparse polynomial evaluation         │
    └─────────────┬───────────────────────────┘
                  │ uses PCS trait
    ┌─────────────▼───────────────────────────┐
    │  WHIR Polynomial Commitment (whir-pcs)  │
    ├─────────────────────────────────────────┤
    │  • Reed-Solomon proximity testing       │
    │  • BLAKE3-based Merkle commitment       │
    │  • Batch opening proofs                 │
    │  • BabyBear field optimization          │
    └─────────────┬───────────────────────────┘
                  │ memory operations
    ┌─────────────▼───────────────────────────┐
    │  Memory Checking (memory-check)         │
    ├─────────────────────────────────────────┤
    │  • Offline memory consistency proofs    │
    │  • Read/write operation validation      │
    │  • Address collision handling           │
    │  • Sparse access pattern optimization   │
    └─────────────────────────────────────────┘

    Cross-cutting: Field operations, hashing, threading
```

---

## 3 Revised Implementation Steps

### Phase 1: Foundation Layer (`field-utils`) - Week 1

1. **Leverage existing Deep-FRI infrastructure**:
   * ✅ `BabyBear` + `BinomialExtensionField<BabyBear, 4>` already integrated
   * ✅ BLAKE3 hashing already available  
   * ✅ Merkle tree implementation exists

2. **Extend field utilities**:
   ```rust
   // Build on existing src/utils/mod.rs
   pub trait SpartanField: Field {
       fn batch_invert(elements: &mut [Self]);
       fn random_elements<R: Rng>(n: usize, rng: &mut R) -> Vec<Self>;
   }
   impl SpartanField for BabyBear { /* use Montgomery trick */ }
   ```

3. **Threading abstraction**:
   ```rust
   pub struct DeterministicThreadPool {
       worker_count: usize,
   }
   impl DeterministicThreadPool {
       pub fn scoped<F, R>(&self, f: F) -> R 
       where F: FnOnce(&crossbeam::Scope) -> R;
   }
   ```

### Phase 2: WHIR Integration (`whir-pcs`) - Weeks 2-3

1. **Fork and adapt WHIR implementation**:
   * ⚠️  **Critical**: WHIR is academic prototype, needs production hardening
   * Replace arkworks fields with BabyBear from existing codebase
   * Integrate with existing BLAKE3 Merkle tree implementation
   * **Domain size constraint**: BabyBear supports up to 2²⁷ elements efficiently

2. **Production-ready API design**:
   ```rust
   pub struct WhirCommitment {
       root: [u8; 32],            // BLAKE3 hash root
       evaluation_domain_size: usize,
       rate_parameter: f64,       // Configurable 1/2 to 1/16
   }
   
   pub trait PolynomialCommitmentScheme<F: SpartanField> {
       type Commitment;
       type Opening;
       type BatchOpening;
       
       fn commit(&self, poly: &[F]) -> Result<(Self::Commitment, ProverData), Error>;
       fn open(&self, poly: &[F], point: &[F], data: &ProverData) -> Result<Self::Opening, Error>;
       fn batch_open(&self, queries: &[(usize, &[F])], data: &ProverData) -> Result<Self::BatchOpening, Error>;
       fn verify(&self, commitment: &Self::Commitment, opening: &Self::Opening) -> bool;
   }
   ```

3. **Performance targets** (based on research):
   * Commitment time: < 10ms for 2²⁰ coefficients  
   * Opening proof size: < 200KB
   * Verification time: < 5ms
   * **Key insight**: WHIR achieves 5700×-12000× speedup over alternatives

### Phase 3: Memory Checking (`memory-check`) - Week 4

**⚠️ Major Plan Revision**: Based on research, specific "Twist & Shout" implementation unavailable. 
Alternative approach using established offline memory-checking:

1. **Classical offline memory checking**:
   * Use timestamped read/write logs
   * Prove consistency via sorting argument
   * Leverage existing sum-check infrastructure from Deep-FRI

2. **Integration with Spartan**:
   ```rust
   pub struct MemoryChecker<F: SpartanField> {
       memory_size: usize,
       access_log: Vec<MemoryAccess<F>>,
   }
   
   pub struct MemoryAccess<F> {
       timestamp: u64,
       address: usize, 
       value: F,
       is_write: bool,
   }
   
   impl<F: SpartanField> MemoryChecker<F> {
       pub fn read(&mut self, addr: usize) -> F;
       pub fn write(&mut self, addr: usize, value: F);
       pub fn prove_consistency(&self) -> MemoryProof<F>;
   }
   ```

### Phase 4: Spartan Core (`spartan-core`) - Weeks 5-6

1. **Port Microsoft Spartan to BabyBear**:
   * Replace `curve25519-dalek` → `BabyBear` field operations
   * Adapt `ristretto255` group operations → BabyBear extension field
   * Maintain R1CS → constraint matrix encoding logic
   * **Key challenge**: Discrete log assumption needs validation in BabyBear context

2. **PCS trait integration**:
   ```rust
   pub struct SpartanProver<PCS: PolynomialCommitmentScheme<BabyBear>> {
       pcs: PCS,
       memory_checker: MemoryChecker<BabyBear>,
       thread_pool: DeterministicThreadPool,
   }
   ```

3. **Sum-check protocol optimization**:
   * Leverage existing `EqEvals` implementation from `src/utils/eq.rs`
   * Integrate with existing MLE evaluation from `src/utils/polynomial.rs`
   * **Performance target**: 36-152× speedup maintained from original Spartan

### Phase 5: Integration & CLI (`examples/`) - Weeks 7-8

1. **CLI implementation leveraging existing patterns**:
   ```bash
   # Build on existing Deep-FRI structure
   cargo run --bin spartan-prove --release -- circuit.r1cs witness.json
   cargo run --bin spartan-verify --release -- proof.bin public.json
   ```

2. **Benchmark suite**:
   * Reuse existing `criterion` setup if available
   * Target circuits: Fibonacci, SHA-256, merkle proof validation
   * Performance comparison vs original Spartan

---

## 4 Refined Project Timeline & Risk Assessment

| Week | Deliverable | Risk Level | Mitigation |
|-----:|-------------|------------|------------|
|1|✅ Foundation layer on existing Deep-FRI|🟢 Low|Leverage existing BabyBear/BLAKE3 integration|
|2-3|🔄 WHIR-PCS adaptation|🟡 Medium|WHIR is prototype - allocate extra time for hardening|
|4|🔄 Memory-checking (revised approach)|🟡 Medium|Fallback to simpler offline checking if needed|
|5-6|🔄 Spartan core integration|🔴 High|Complex cryptographic adaptation - security critical|
|7|🔄 Threading & optimization|🟡 Medium|Use existing patterns from Deep-FRI|
|8|🔄 CLI + benchmarks|🟢 Low|Build on existing infrastructure|

**Critical Dependencies Identified**:
1. ⚠️  **WHIR production readiness**: Academic prototype status
2. ⚠️  **BabyBear security**: Discrete log assumptions need validation  
3. ⚠️  **Memory-checking**: "Twist & Shout" unavailable, need alternative

---

## 5 Enhanced Testing & Validation Strategy

### Testing Pyramid
* **Unit tests** (build on existing Deep-FRI patterns):
  * Field arithmetic: BabyBear operations, extension field computations
  * Cryptographic primitives: BLAKE3 hashing, Merkle tree operations  
  * WHIR components: Reed-Solomon proximity, commitment/opening
  * Memory checking: consistency proofs, access pattern validation

* **Integration tests**:
  * End-to-end Spartan proving: R1CS → proof → verification
  * PCS trait compatibility: WHIR integration with Spartan core
  * Cross-component: Memory checker with sum-check protocol

* **Property-based testing** (`proptest`):
  * Random R1CS circuits (2⁸ to 2¹⁶ constraints)
  * Fuzzing polynomial evaluations and commitments  
  * Memory access pattern randomization

* **Performance benchmarks** (`criterion`):
  * Proof generation scaling: 2⁸, 2¹², 2¹⁶, 2²⁰ constraints
  * Verification latency vs proof size
  * Memory usage profiling during proving
  * **Baseline**: Compare against original Spartan performance claims

### Security Validation
* **Cryptographic soundness**:
  * WHIR security parameter validation (128-bit security)
  * BabyBear discrete log assumption verification
  * Memory-checking argument soundness proofs

* **Implementation security**:
  * `#![forbid(unsafe_code)]` except necessary SIMD
  * Constant-time field operations via `p3-baby-bear`
  * Side-channel resistance analysis

---

## 6 Production Readiness & Security Considerations

### Code Quality Standards
1. **Memory safety**: Leverage Rust's ownership system, minimize `unsafe`
2. **Deterministic execution**: Reproducible proofs across platforms
3. **Error handling**: Comprehensive `Result` types, no panics in production paths
4. **Documentation**: Rustdoc for all public APIs, implementation notes for complex algorithms

### Security Review Checklist
- [ ] **Field operations**: Validate BabyBear arithmetic correctness
- [ ] **WHIR adaptation**: Security parameter preservation during field change
- [ ] **Memory checking**: Consistency proof soundness validation
- [ ] **Threading**: Race condition analysis, deterministic parallel execution
- [ ] **Fiat-Shamir**: Challenge generation security (existing BLAKE3 challenger)

### Deployment Considerations
- **Feature flags**: `default = ["std"], no-std = [], simd = ["p3-baby-bear/avx2"]`
- **Audit trail**: `cargo auditable`, dependency scanning
- **Performance monitoring**: Built-in timing/memory instrumentation
- **Backward compatibility**: Stable proof format, versioned serialization

---

## 7 Build & Usage (Updated)

### Development Workflow
```bash
# Build entire workspace (leveraging existing Deep-FRI base)
cargo build --release --workspace

# Run comprehensive test suite
cargo test --workspace --release

# Benchmark suite  
cargo bench --workspace

# Example: Prove Fibonacci circuit
cargo run --bin spartan-prove --release -- \
  --circuit examples/circuits/fibonacci.r1cs \
  --witness examples/witnesses/fibonacci.json \
  --output proof.bin

# Verify proof
cargo run --bin spartan-verify --release -- \
  --proof proof.bin \
  --public examples/public/fibonacci.json
```

### Workspace Structure (Revised)
```text
spartan-whir/              # Workspace root (rename from deep-fri)
├── Cargo.toml            # Workspace manifest
├── field-utils/          # BabyBear utilities & traits  
├── whir-pcs/             # WHIR polynomial commitments
├── memory-check/         # Offline memory-checking  
├── spartan-core/         # Spartan protocol implementation
├── examples/             # CLI binaries & test circuits
└── benches/              # Cross-crate benchmarks
```

### Feature Configuration
```toml
[features]
default = ["std"]
std = []                           # Standard library support
no-std = []                        # Embedded/WASM compatibility  
simd = ["p3-baby-bear/avx2"]      # Vectorized field operations
parallel = ["crossbeam"]           # Multi-threading support
```

### Integration with Existing Deep-FRI
* ✅ Reuse `src/utils/` modules (merkle_tree, challenger, eq, polynomial)
* ✅ Extend `BabyBear` + `BinomialExtensionField` usage
* ✅ Leverage existing BLAKE3 integration
* 🔄 Refactor `src/prover.rs` → `whir-pcs/` crate
* 🔄 Create workspace structure around existing components

---

[^1]: https://github.com/microsoft/Spartan

[^2]: https://github.com/microsoft/Spartan2

[^3]: https://www.microsoft.com/en-us/research/publication/spartan-efficient-and-general-purpose-zksnarks-without-trusted-setup/

[^4]: https://perfilesycapacidades.javeriana.edu.co/files/26740951/Douchet_Antoine_DOUA72320201_EVALUATION_1.pdf

[^5]: https://crypto.ku.edu.tr/spartan-efficient-and-general-purpose-zksnarks-without-trusted-setup/

[^6]: https://encrypt.a41.io/zk/snark/spartan

[^7]: https://identity.foundation/spartan_zkSNARK_signatures/draft-setty-cfrg-spartan-incubation.html

[^8]: https://hackmd.io/@HBBHZjW4TU-_rDF7GQh3lg/Hk3G6kcPi

[^9]: https://personaelabs.org/posts/spartan-ecdsa/

[^10]: https://people.csail.mit.edu/devadas/pubs/micro24_nocap.pdf

[^11]: https://zkplabs.network/blog/explore-zksnarks-an-introduction-and-diverse-applications-in-web3-space

[^12]: https://blog.lambdaclass.com/our-highly-subjective-view-on-the-history-of-zero-knowledge-proofs/

[^13]: https://dl.acm.org/doi/10.1007/978-3-030-56877-1_25

[^14]: https://www.youtube.com/watch?v=FPQs7T7f_AU

[^15]: https://arxiv.org/html/2502.07063v1

[^16]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-20.pdf

[^17]: https://2π.com/24/whir/

[^18]: https://en.wikipedia.org/wiki/Whirlpool_(hash_function)

[^19]: https://people.seas.harvard.edu/~salil/research/ZKcommit-tcc.pdf

[^20]: https://ethresear.ch/t/on-the-gas-efficiency-of-the-whir-polynomial-commitment-scheme/21301

[^21]: https://mojoauth.com/compare-hashing-algorithms/whirlpool-vs-bcrypt/

[^22]: https://arxiv.org/pdf/2310.14848.pdf

[^23]: https://wizardofmenlo.github.io/papers/whir/

[^24]: https://vssut.ac.in/lecture_notes/lecture1428550736.pdf

[^25]: https://blog.cryptographyengineering.com/2014/11/27/zero-knowledge-proofs-illustrated-primer/

[^26]: https://www.youtube.com/watch?v=iPKzmxLDdII

[^27]: https://www.europeanpaymentscouncil.eu/sites/default/files/kb/file/2025-03/EPC342-08 v15.0 Approved - Guidelines on Cryptographic Algorithms Usage and Key Management 1.pdf

[^28]: https://wizardofmenlo.github.io/tags/theory/

[^29]: https://berry.win.tue.nl/CryptographicProtocols/LectureNotes.pdf

[^30]: https://www.sciencedirect.com/science/article/pii/S0304397521004242

[^31]: https://dl.acm.org/doi/10.1007/978-3-031-91134-7_8

[^32]: https://www.splunk.com/en_us/blog/learn/cryptography.html

[^33]: https://blog.cloudflare.com/introducing-zero-knowledge-proofs-for-private-web-attestation-with-cross-multi-vendor-hardware/

[^34]: https://github.com/WizardOfMenlo/WizardOfMenlo

[^35]: https://twitter.com/GiacomoFenzi/status/1843554151080370643

[^36]: https://www.youtube.com/watch?v=q6V7z7_y9hk

[^37]: https://en.wikipedia.org/wiki/Zero-knowledge_proof

[^38]: https://georgwiese.github.io/crypto-summaries/Concepts/Protocols/Offline-Memory-Checking

[^39]: https://a16zcrypto.com/posts/article/introducing-twist-and-shout/

[^40]: https://pages.cs.wisc.edu/~shrey/2020/02/26/zero-knowledge-proofs.html

[^41]: https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=1aee9b1b256e5b70aeb37f655d91a492a1783a83

[^42]: https://www.youtube.com/watch?v=nEEFjyTK8OI

[^43]: https://www.nttdata.com/global/en/insights/focus/2024/what-is-zero-knowledge-proof

[^44]: https://www.wisdom.weizmann.ac.il/~naor/PAPERS/omc.pdf

[^45]: https://x.com/SuccinctJT/status/1882435762190504143

[^46]: https://www.tdx.cat/bitstream/10803/671222/1/tjsv.pdf

[^47]: https://microarch.org/micro36/html/pdf/suh-EfficMemory.pdf

[^48]: https://x.com/Lhree/status/1895232455998873998

[^49]: https://101blockchains.com/zero-knowledge-proof/

[^50]: https://dl.acm.org/doi/abs/10.1145/3707202

[^51]: https://web-cdn.bsky.app/profile/zkhack.dev/post/3lgiomp3p6k2a

[^52]: https://www.youtube.com/watch?v=saVD9qo3aJ0

[^53]: https://dspace.mit.edu/bitstream/handle/1721.1/158076/3707202.pdf?sequence=1\&isAllowed=y

[^54]: https://chalkdustmagazine.com/features/twist-and-shout/

[^55]: https://www.boazbarak.org/cs127spring16/chap14_zero_knowledge.html

[^56]: https://hackmd.io/@Voidkai/BkNX3xUZA

[^57]: https://vitalik.eth.limo/general/2024/04/29/binius.html

[^58]: https://docs.rs/risc0-zkp/latest/risc0_zkp/field/baby_bear/struct.BabyBear.html

[^59]: https://crypto.stackexchange.com/questions/111785/discrete-log-of-goldilocks-babybear-and-mersenne31-fields

[^60]: https://www.binance.com/en/square/post/10883006009353

[^61]: https://docs.rs/p3-baby-bear

[^62]: https://docs.polygon.technology/learn/plonky3/examples/fibonacci/

[^63]: https://github.com/BitVM/rust-bitcoin-m31-or-babybear

[^64]: https://github.com/Plonky3/Plonky3

[^65]: https://lib.rs/crates/p3-baby-bear

[^66]: https://docs.powdr.org/backends/plonky3.html

[^67]: https://crates.io/crates/p3-baby-bear/0.1.3-succinct

[^68]: https://polygon.technology/blog/polygon-plonky3-the-next-generation-of-zk-proving-systems-is-production-ready

[^69]: https://crates.io/crates/p3-baby-bear/0.2.3-succinct/dependencies

[^70]: https://www.lita.foundation/blog/plonky-3-valida-october-review

[^71]: https://github.com/telosnetwork/plonky2_goldibear/

[^72]: https://lita.gitbook.io/lita-documentation/architecture/proving-system-plonky3

[^73]: https://blog.icme.io/small-fields-for-zero-knowledge/

[^74]: https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/baby_bear.rs

[^75]: https://www.fortsmithchamber.org/spartan-logistics/

[^76]: https://en.wikipedia.org/wiki/Sparse_polynomial

[^77]: https://www.reddit.com/r/rust/comments/zsm56p/rayonjoin_vs_stdthread_what_should_i_use/

[^78]: https://hackmd.io/@clientsideproving/spartan-whir-proposal

[^79]: https://drops.dagstuhl.de/storage/00lipics/lipics-vol250-fsttcs2022/LIPIcs.FSTTCS.2022.10/LIPIcs.FSTTCS.2022.10.pdf

[^80]: https://www.linkedin.com/pulse/data-parallelism-rust-rayoncrate-luis-soares-m-sc--csvsf

[^81]: https://www.xilinx.com/publications/prod_mktg/pn2027.pdf

[^82]: https://www.csd.uwo.ca/~mmorenom/Publications/Sparse_polynomial_arithmetic.pdf

[^83]: https://www.youtube.com/watch?v=XtU1s8yoSpI

[^84]: https://www.spartancontrols.com/services/measurement-automation-services/remote-connected-services/

[^85]: https://arxiv.org/pdf/1807.08289.pdf

[^86]: https://users.rust-lang.org/t/rayon-thread-pool-vs-thread-pool/12358

[^87]: https://www.spartansolutions.com/benefits/synchronise-operations-with-enterprise-applications/

[^88]: https://www.cse.iitk.ac.in/users/nitin/papers/symmetricSparse.pdf

[^89]: https://docs.rs/rayon

[^90]: https://spartans.tech/services/augmented-reality/

[^91]: https://simons.berkeley.edu/sites/default/files/docs/11063/deterministicfactorizationofsparsepolynomialswithboundedindividualdegreefocs9.pdf

[^92]: https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html

[^93]: https://aws.amazon.com/marketplace/pp/prodview-45a57y3jj4tlw

[^94]: https://www.sciencedirect.com/science/article/pii/S0022247X2100528X

[^95]: https://github.com/arkworks-rs/poly-commit

[^96]: https://hackmd.io/@sin7y/Bkun7AgFA

[^97]: https://docs.rs/zksnark

[^98]: https://arxiv.org/html/2405.12115v1

[^99]: https://github.com/SohamJog/reed_solomon_rs

[^100]: https://github.com/EspressoSystems/jellyfish

[^101]: https://docs.rs/reed-solomon/latest/reed_solomon/

[^102]: https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/

[^103]: https://www.mdpi.com/2078-2489/15/8/463

[^104]: https://github.com/WizardOfMenlo/whir

[^106]: https://en.wikiversity.org/wiki/Reed–Solomon_codes_for_coders

[^107]: https://lib.rs/algorithms

[^108]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/a45bd27be6368057e59dea9518141fb9/a2bb2f54-9fe6-4980-b2e9-f95b40acd6af/9f0c4b19.md


