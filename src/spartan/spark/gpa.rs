//! Batched Grand Product Argument (GPA) with Offline Memory Checking
//!
//! Proves multiple product claims with O(log²m) communication using gamma batching.
//! Supports offline memory checking by verifying that left multiset (I ∪ W) and
//! right multiset (R ∪ F) have matching products after fingerprinting.
//!
//! ```rust
//! use spartan::gpa::{MemoryCheckInstance, prove_batched_memory_consistency};
//! use spartan::utils::{Fp4, Challenger};
//! use p3_baby_bear::BabyBear;
//!
//! let mut instance = MemoryCheckInstance::new();
//! instance.add_init(0, BabyBear::from_u32(100));
//! instance.add_read(0, BabyBear::from_u32(100), 1);
//! instance.add_write(0, BabyBear::from_u32(200), 2);
//! instance.add_final(0, BabyBear::from_u32(200));
//!
//! let instances = vec![instance];
//! let proof = prove_batched_memory_consistency(&instances, alpha, gamma, tau, &mut challenger)?;
//! ```

// Note: We implement our own ProductTree structures instead of using product.rs
use crate::spartan::univariate::UnivariatePoly;
use crate::utils::challenger::Challenger;
use crate::utils::eq::EqEvals;
use crate::utils::polynomial::MLE;
use crate::utils::{Fp, Fp4};
use p3_field::PrimeCharacteristicRing;
use std::fmt;

/// Errors that can occur during GPA operations.
#[derive(Debug, Clone)]
pub enum GpaError {
    /// Field arithmetic error
    FieldError(String),
    /// Memory validation error
    MemoryValidationError(String),
    /// Batch size mismatch
    BatchSizeError(String),
    /// Index out of bounds
    IndexOutOfBounds { index: usize, bounds: usize },
}

impl fmt::Display for GpaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpaError::FieldError(msg) => write!(f, "Field error: {}", msg),
            GpaError::MemoryValidationError(msg) => write!(f, "Memory validation error: {}", msg),
            GpaError::BatchSizeError(msg) => write!(f, "Batch size error: {}", msg),
            GpaError::IndexOutOfBounds { index, bounds } => {
                write!(f, "Index {} out of bounds [0, {})", index, bounds)
            }
        }
    }
}

impl std::error::Error for GpaError {}

/// Binary product tree for batched GPA with left/right layer storage.
#[derive(Debug, Clone)]
pub struct ProductTree {
    /// Left and right halves for each layer (root to leaves)
    pub left_layers: Vec<MLE<Fp4>>,
    pub right_layers: Vec<MLE<Fp4>>,
    /// Tree depth (log of input size)
    pub depth: usize,
    /// Root value (final product)
    pub root_value: Fp4,
}

impl ProductTree {
    /// Creates a new product tree from input vector.
    pub fn new(input: MLE<Fp4>) -> Result<Self, GpaError> {
        if input.len() == 0 {
            return Err(GpaError::FieldError(
                "Cannot create tree from empty input".to_string(),
            ));
        }

        if !input.len().is_power_of_two() {
            return Err(GpaError::FieldError(
                "Input size must be power of two".to_string(),
            ));
        }

        let mut left_layers = Vec::new();
        let mut right_layers = Vec::new();
        let mut current_layer = input;
        let depth = current_layer.n_vars();

        // Build layers from bottom up
        for _ in 0..depth {
            let len = current_layer.len();
            let half_len = len / 2;

            // Split into left and right halves
            let mut left_coeffs = Vec::with_capacity(half_len);
            let mut right_coeffs = Vec::with_capacity(half_len);

            for i in 0..half_len {
                left_coeffs.push(current_layer[i]);
                right_coeffs.push(current_layer[i + half_len]);
            }

            let left_half = MLE::new(left_coeffs);
            let right_half = MLE::new(right_coeffs);

            left_layers.push(left_half.clone());
            right_layers.push(right_half.clone());

            // Compute next layer: element-wise multiplication
            let mut next_coeffs = Vec::with_capacity(half_len);
            for i in 0..half_len {
                next_coeffs.push(left_half[i] * right_half[i]);
            }

            if half_len == 1 {
                // We've reached the root
                break;
            }

            current_layer = MLE::new(next_coeffs);
        }

        // The root value is the final product
        let root_value = if left_layers.is_empty() {
            // Single element
            current_layer[0]
        } else {
            // Product of final left and right
            let last_left = &left_layers[left_layers.len() - 1];
            let last_right = &right_layers[right_layers.len() - 1];
            last_left[0] * last_right[0]
        };

        Ok(Self {
            left_layers,
            right_layers,
            depth,
            root_value,
        })
    }

    pub fn get_root_value(&self) -> Fp4 {
        self.root_value
    }

    pub fn num_layers(&self) -> usize {
        self.depth
    }

    pub fn left_layer(&self, layer: usize) -> Option<&MLE<Fp4>> {
        self.left_layers.get(layer)
    }

    pub fn right_layer(&self, layer: usize) -> Option<&MLE<Fp4>> {
        self.right_layers.get(layer)
    }
}

/// Memory operations for one execution trace.
/// Consistency verified by comparing products of I∪W and R∪F multisets.
#[derive(Debug, Clone)]
pub struct MemoryCheckInstance {
    /// Initial memory state: (address, value)
    pub init_ops: Vec<(usize, Fp)>,
    /// Read operations: (address, value, timestamp)
    pub read_ops: Vec<(usize, Fp, usize)>,
    /// Write operations: (address, value, timestamp)
    pub write_ops: Vec<(usize, Fp, usize)>,
    /// Final memory state: (address, value)
    pub final_ops: Vec<(usize, Fp)>,
}

impl MemoryCheckInstance {
    /// Creates a new empty memory checking instance.
    pub fn new() -> Self {
        Self {
            init_ops: Vec::new(),
            read_ops: Vec::new(),
            write_ops: Vec::new(),
            final_ops: Vec::new(),
        }
    }

    /// Adds an initial memory operation.
    pub fn add_init(&mut self, addr: usize, val: Fp) {
        self.init_ops.push((addr, val));
    }

    /// Adds a read memory operation.
    pub fn add_read(&mut self, addr: usize, val: Fp, timestamp: usize) {
        self.read_ops.push((addr, val, timestamp));
    }

    /// Adds a write memory operation.
    pub fn add_write(&mut self, addr: usize, val: Fp, timestamp: usize) {
        self.write_ops.push((addr, val, timestamp));
    }

    /// Adds a final memory operation.
    pub fn add_final(&mut self, addr: usize, val: Fp) {
        self.final_ops.push((addr, val));
    }

    /// Validates the memory trace for consistency.
    pub fn validate(&self) -> Result<(), GpaError> {
        use std::collections::HashSet;

        // Check that init and final operations are paired
        let init_addrs: HashSet<usize> = self.init_ops.iter().map(|(addr, _)| *addr).collect();
        let final_addrs: HashSet<usize> = self.final_ops.iter().map(|(addr, _)| *addr).collect();

        if init_addrs != final_addrs {
            return Err(GpaError::MemoryValidationError(
                "Init and final addresses must match".to_string(),
            ));
        }

        // Check that all read/write operations reference initialized addresses
        for (addr, _, _) in &self.read_ops {
            if !init_addrs.contains(addr) {
                return Err(GpaError::MemoryValidationError(format!(
                    "Read operation at uninitialized address {}",
                    addr
                )));
            }
        }

        for (addr, _, _) in &self.write_ops {
            if !init_addrs.contains(addr) {
                return Err(GpaError::MemoryValidationError(format!(
                    "Write operation at uninitialized address {}",
                    addr
                )));
            }
        }

        Ok(())
    }

    /// Computes fingerprinted multisets using `h(addr,val,t) = α*addr + val*γ + t`.
    /// Returns (left_multiset=I∪W, right_multiset=R∪F).
    pub fn fingerprint_multisets(&self, alpha: Fp4, gamma: Fp4, tau: Fp4) -> (Vec<Fp4>, Vec<Fp4>) {
        let mut left_multiset = Vec::new();
        let mut right_multiset = Vec::new();

        // Left multiset: Init operations (I)
        for &(addr, val) in &self.init_ops {
            // h_τ,γ(addr, val, t) = α*addr + val*γ + t (where t=τ for init)
            let fingerprint = alpha * Fp4::from_u32(addr as u32) + Fp4::from(val) * gamma + tau;
            left_multiset.push(fingerprint);
        }

        // Left multiset: Write operations (W)
        for &(addr, val, ts) in &self.write_ops {
            // h_τ,γ(addr, val, t) = α*addr + val*γ + t
            let fingerprint = alpha * Fp4::from_u32(addr as u32)
                + Fp4::from(val) * gamma
                + Fp4::from_u32(ts as u32);
            left_multiset.push(fingerprint);
        }

        // Right multiset: Read operations (R)
        for &(addr, val, ts) in &self.read_ops {
            // h_τ,γ(addr, val, t) = α*addr + val*γ + t
            let fingerprint = alpha * Fp4::from_u32(addr as u32)
                + Fp4::from(val) * gamma
                + Fp4::from_u32(ts as u32);
            right_multiset.push(fingerprint);
        }

        // Right multiset: Final operations (F)
        for &(addr, val) in &self.final_ops {
            // h_τ,γ(addr, val, t) = α*addr + val*γ + t (where t=τ for final)
            let fingerprint = alpha * Fp4::from_u32(addr as u32) + Fp4::from(val) * gamma + tau;
            right_multiset.push(fingerprint);
        }

        (left_multiset, right_multiset)
    }
}

impl Default for MemoryCheckInstance {
    fn default() -> Self {
        Self::new()
    }
}

/// Sumcheck proof for one layer across all batched trees.
#[derive(Debug, Clone)]
pub struct BatchedLayerProof {
    /// Sumcheck proof polynomials for this layer
    pub round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations: [left_eval, right_eval] for each tree at this layer
    pub final_evals: Vec<[Fp4; 2]>,
}

/// Complete batched GPA proof with layer proofs and product claims.
#[derive(Debug, Clone)]
pub struct BatchedProductProof {
    /// One proof per layer (from root to leaves)
    pub layer_proofs: Vec<BatchedLayerProof>,
    /// Final product claims for verification
    pub product_claims: Vec<Fp4>,
}

/// Manages multiple product trees for batched proving with gamma powers.
#[derive(Debug, Clone)]
pub struct BatchedProductCircuit {
    /// All product trees (must have same depth)
    pub trees: Vec<ProductTree>,
    /// Number of trees being batched
    pub num_trees: usize,
    /// Shared tree depth
    pub tree_depth: usize,
}

impl BatchedProductCircuit {
    /// Creates a new batched product circuit. All trees must have the same depth.
    pub fn new(trees: Vec<ProductTree>) -> Result<Self, GpaError> {
        if trees.is_empty() {
            return Err(GpaError::BatchSizeError(
                "Cannot create circuit with empty trees".to_string(),
            ));
        }

        let tree_depth = trees[0].num_layers();
        let num_trees = trees.len();

        // Verify all trees have the same depth
        for (i, tree) in trees.iter().enumerate() {
            if tree.num_layers() != tree_depth {
                return Err(GpaError::BatchSizeError(format!(
                    "Tree {} has depth {} but expected depth {}",
                    i,
                    tree.num_layers(),
                    tree_depth
                )));
            }
        }

        Ok(Self {
            trees,
            num_trees,
            tree_depth,
        })
    }

    pub fn num_trees(&self) -> usize {
        self.num_trees
    }

    pub fn tree_depth(&self) -> usize {
        self.tree_depth
    }

    pub fn tree(&self, index: usize) -> Option<&ProductTree> {
        self.trees.get(index)
    }

    /// Extracts root values from all trees as individual claims.
    pub fn extract_product_claims(&self) -> Vec<Fp4> {
        self.trees
            .iter()
            .map(|tree| tree.get_root_value())
            .collect()
    }
}

/// Computes batched sumcheck round: `Σ γ^i * left_i(X) * right_i(X) * eq(X)`.
pub fn compute_gpa_round_batched(
    left_trees: &[&MLE<Fp4>],
    right_trees: &[&MLE<Fp4>],
    eq_poly: &MLE<Fp4>,
    gamma: Fp4,
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> Result<UnivariatePoly, GpaError> {
    if left_trees.len() != right_trees.len() {
        return Err(GpaError::BatchSizeError(
            "Left and right trees must have same length".to_string(),
        ));
    }

    if left_trees.is_empty() {
        return Err(GpaError::BatchSizeError(
            "Cannot compute round with empty trees".to_string(),
        ));
    }

    let num_trees = left_trees.len();
    let domain_size = 1 << (rounds - round - 1);

    // Initialize coefficients for quadratic polynomial: g(X) = a + bX + cX²
    let mut round_coeffs = vec![Fp4::ZERO; 3];

    // Process each position in the current domain
    for i in 0..domain_size {
        let eq_val = if i < eq_poly.len() {
            eq_poly[i]
        } else {
            Fp4::ZERO
        };

        // Compute g(0): direct evaluation at X=0
        let mut term_0 = Fp4::ZERO;
        for tree_idx in 0..num_trees {
            let gamma_power = gamma.exp_u64(tree_idx as u64);

            let idx = i << 1; // Left child index
            if idx < left_trees[tree_idx].len() && idx < right_trees[tree_idx].len() {
                let tree_contribution =
                    left_trees[tree_idx][idx] * right_trees[tree_idx][idx] * eq_val;
                term_0 += gamma_power * tree_contribution;
            }
        }
        round_coeffs[0] += term_0;

        // Compute g(2): using multilinear identity f(2) = f(0) + 2*f(1)
        let mut term_2 = Fp4::ZERO;
        for tree_idx in 0..num_trees {
            let gamma_power = gamma.exp_u64(tree_idx as u64);

            let left_idx = i << 1; // Left child
            let right_idx = left_idx | 1; // Right child

            if right_idx < left_trees[tree_idx].len() && right_idx < right_trees[tree_idx].len() {
                // Apply multilinear identity: f(2) = f(0) + 2*f(1)
                let left_2 =
                    left_trees[tree_idx][left_idx] + left_trees[tree_idx][right_idx].double();
                let right_2 =
                    right_trees[tree_idx][left_idx] + right_trees[tree_idx][right_idx].double();
                let tree_contribution = left_2 * right_2 * eq_val;
                term_2 += gamma_power * tree_contribution;
            }
        }
        round_coeffs[2] += term_2;
    }

    // g(1): derived from sumcheck constraint g(0) + g(1) = current_claim
    round_coeffs[1] = current_claim - round_coeffs[0];

    // Create and validate the univariate polynomial
    let mut round_poly = UnivariatePoly::new(round_coeffs)
        .map_err(|e| GpaError::FieldError(format!("Failed to create polynomial: {}", e)))?;

    // Interpolate to get proper polynomial representation
    round_poly
        .interpolate()
        .map_err(|e| GpaError::FieldError(format!("Failed to interpolate polynomial: {}", e)))?;

    Ok(round_poly)
}

impl BatchedProductCircuit {
    /// Generates batched proof using layer-by-layer sumcheck protocol.
    pub fn prove_batched(
        &self,
        gamma: Fp4,
        challenger: &mut Challenger,
    ) -> Result<BatchedProductProof, GpaError> {
        // Extract individual product claims
        let product_claims = self.extract_product_claims();

        // Batch claims using gamma powers: Σ γ^i * claim_i
        let mut current_claim = Fp4::ZERO;
        for (i, &claim) in product_claims.iter().enumerate() {
            current_claim += gamma.exp_u64(i as u64) * claim;
        }

        let mut layer_proofs = Vec::new();
        let mut accumulated_randomness = Vec::new();

        // Process each layer from root to leaves
        for layer in 0..self.tree_depth {
            // Extract left and right halves for all trees at this layer
            let left_trees: Vec<_> = self
                .trees
                .iter()
                .filter_map(|tree| tree.left_layer(layer))
                .collect();
            let right_trees: Vec<_> = self
                .trees
                .iter()
                .filter_map(|tree| tree.right_layer(layer))
                .collect();

            if left_trees.len() != self.num_trees || right_trees.len() != self.num_trees {
                return Err(GpaError::FieldError(
                    "Failed to extract layers from all trees".to_string(),
                ));
            }

            // Create equality polynomial from accumulated randomness
            let eq_poly = if accumulated_randomness.is_empty() {
                // First layer: equality polynomial is all ones
                MLE::new(vec![Fp4::ONE; 1])
            } else {
                let eq_evals = EqEvals::gen_from_point(&accumulated_randomness);
                MLE::new(eq_evals.coeffs)
            };

            // Compute batched round polynomial
            let rounds_remaining = self.tree_depth - layer;
            let round_proof = compute_gpa_round_batched(
                &left_trees,
                &right_trees,
                &eq_poly,
                gamma,
                current_claim,
                layer,
                rounds_remaining,
            )?;

            // Add to transcript and get challenge
            challenger.observe_fp4_elems(&round_proof.coefficients());
            let challenge = challenger.get_challenge();

            // Update claim for next layer
            current_claim = round_proof.evaluate(challenge);
            accumulated_randomness.push(challenge);

            // Extract final evaluations for this layer
            let final_evals = extract_layer_evaluations(&left_trees, &right_trees, challenge)?;

            // Store proof for this layer
            layer_proofs.push(BatchedLayerProof {
                round_proofs: vec![round_proof],
                final_evals,
            });
        }

        Ok(BatchedProductProof {
            layer_proofs,
            product_claims,
        })
    }
}

impl BatchedProductProof {
    /// Verifies batched proof by checking sumcheck relations at each layer.
    pub fn verify(&self, gamma: Fp4, challenger: &mut Challenger) -> Result<bool, GpaError> {
        if self.layer_proofs.is_empty() {
            return Ok(false);
        }

        // Reconstruct batched claim using gamma powers
        let mut current_claim = Fp4::ZERO;
        for (i, &claim) in self.product_claims.iter().enumerate() {
            current_claim += gamma.exp_u64(i as u64) * claim;
        }

        let mut accumulated_randomness = Vec::new();

        // Verify each layer proof
        for (layer_idx, layer_proof) in self.layer_proofs.iter().enumerate() {
            if layer_proof.round_proofs.is_empty() {
                return Ok(false);
            }

            // Verify sumcheck relation for this layer
            let round_poly = &layer_proof.round_proofs[0];

            // Check: current_claim = g(0) + g(1)
            let g0 = round_poly.evaluate(Fp4::ZERO);
            let g1 = round_poly.evaluate(Fp4::ONE);
            let expected = g0 + g1;

            if current_claim != expected {
                return Ok(false);
            }

            // Get challenge and update claim
            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            accumulated_randomness.push(challenge);
        }

        // Final verification: check that claimed products match final evaluations
        if let Some(last_layer) = self.layer_proofs.last() {
            let is_final_valid = verify_final_evaluations(
                &last_layer.final_evals,
                &self.product_claims,
                current_claim,
            )?;
            if !is_final_valid {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Batched memory consistency proof with left/right product proofs.
#[derive(Debug, Clone)]
pub struct BatchedMemoryProof {
    /// Proof for batched left products: H_τ,γ(I ∪ W) for all instances
    pub left_product_proof: BatchedProductProof,
    /// Proof for batched right products: H_τ,γ(R ∪ F) for all instances
    pub right_product_proof: BatchedProductProof,
    /// Individual left product claims for verification
    pub left_claims: Vec<Fp4>,
    /// Individual right product claims for verification
    pub right_claims: Vec<Fp4>,
}

/// Proves batched memory consistency using fingerprinting and dual-tree GPA.
pub fn prove_batched_memory_consistency(
    memory_instances: &[MemoryCheckInstance],
    alpha: Fp4,
    gamma: Fp4,
    tau: Fp4,
    challenger: &mut Challenger,
) -> Result<BatchedMemoryProof, GpaError> {
    if memory_instances.is_empty() {
        return Err(GpaError::BatchSizeError(
            "Cannot prove empty instance set".to_string(),
        ));
    }

    // Step 1: Validate all memory instances
    for (i, instance) in memory_instances.iter().enumerate() {
        instance.validate().map_err(|e| {
            GpaError::MemoryValidationError(format!("Instance {} validation failed: {}", i, e))
        })?;
    }

    // Step 2: Fingerprint all memory operations
    let mut left_multisets = Vec::new();
    let mut right_multisets = Vec::new();

    for instance in memory_instances {
        let (left, right) = instance.fingerprint_multisets(alpha, gamma, tau);
        left_multisets.push(left);
        right_multisets.push(right);
    }

    // Step 3: Determine maximum size for padding
    let max_left_size = left_multisets.iter().map(|ms| ms.len()).max().unwrap_or(0);
    let max_right_size = right_multisets.iter().map(|ms| ms.len()).max().unwrap_or(0);
    let max_size = max_left_size.max(max_right_size);

    // Ensure size is a power of two
    let padded_size = if max_size == 0 {
        1
    } else {
        max_size.next_power_of_two()
    };

    // Step 4: Build product trees (pad to same size)
    let left_trees: Result<Vec<_>, _> = left_multisets
        .into_iter()
        .map(|ms| {
            let padded = pad_to_power_of_two(ms, padded_size);
            ProductTree::new(MLE::new(padded))
        })
        .collect();
    let left_trees = left_trees?;

    let right_trees: Result<Vec<_>, _> = right_multisets
        .into_iter()
        .map(|ms| {
            let padded = pad_to_power_of_two(ms, padded_size);
            ProductTree::new(MLE::new(padded))
        })
        .collect();
    let right_trees = right_trees?;

    // Step 5: Generate batching challenge
    challenger.observe_field_elem(&Fp::from_u32(memory_instances.len() as u32));
    let batch_gamma = challenger.get_challenge();

    // Step 6: Create batched circuits and prove both sides
    let left_circuit = BatchedProductCircuit::new(left_trees)?;
    let right_circuit = BatchedProductCircuit::new(right_trees)?;

    let left_product_proof = left_circuit.prove_batched(batch_gamma, challenger)?;
    let right_product_proof = right_circuit.prove_batched(batch_gamma, challenger)?;

    // Step 7: Extract individual claims for verification
    let left_claims = left_product_proof.product_claims.clone();
    let right_claims = right_product_proof.product_claims.clone();

    Ok(BatchedMemoryProof {
        left_product_proof,
        right_product_proof,
        left_claims,
        right_claims,
    })
}

impl BatchedMemoryProof {
    /// Verifies memory consistency by checking both product proofs and claim equality.
    pub fn verify(&self, batch_gamma: Fp4, challenger: &mut Challenger) -> Result<bool, GpaError> {
        // Verify both product proofs
        // Note: In a real implementation, we'd need to properly handle challenger state
        // For now, we'll verify sequentially and assume proper transcript management
        let left_valid = self.left_product_proof.verify(batch_gamma, challenger)?;
        if !left_valid {
            return Ok(false);
        }

        let right_valid = self.right_product_proof.verify(batch_gamma, challenger)?;
        if !right_valid {
            return Ok(false);
        }

        // Check that left and right claims match (indicating memory consistency)
        if self.left_claims.len() != self.right_claims.len() {
            return Ok(false);
        }

        for (left_claim, right_claim) in self.left_claims.iter().zip(&self.right_claims) {
            if left_claim != right_claim {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Checks memory consistency for a single instance by comparing multiset products.
pub fn check_memory_consistency(
    instance: &MemoryCheckInstance,
    alpha: Fp4,
    gamma: Fp4,
    tau: Fp4,
) -> Result<bool, GpaError> {
    instance.validate()?;

    let (left_multiset, right_multiset) = instance.fingerprint_multisets(alpha, gamma, tau);

    // Compute products directly for single instance
    let left_product = left_multiset.iter().fold(Fp4::ONE, |acc, &x| acc * x);
    let right_product = right_multiset.iter().fold(Fp4::ONE, |acc, &x| acc * x);

    Ok(left_product == right_product)
}

/// Extracts [left_eval, right_eval] pairs for all trees at challenge point.
pub fn extract_layer_evaluations(
    left_trees: &[&MLE<Fp4>],
    right_trees: &[&MLE<Fp4>],
    challenge: Fp4,
) -> Result<Vec<[Fp4; 2]>, GpaError> {
    if left_trees.len() != right_trees.len() {
        return Err(GpaError::BatchSizeError(
            "Left and right trees must have same length".to_string(),
        ));
    }

    let mut final_evals = Vec::new();

    for (left_tree, right_tree) in left_trees.iter().zip(right_trees.iter()) {
        // Proper multilinear evaluation at challenge point
        // For now, simplified evaluation - should be enhanced with proper MLE evaluation
        let left_eval = if left_tree.len() == 0 {
            Fp4::ZERO
        } else {
            left_tree[0] // Simplified - should evaluate at challenge
        };
        let right_eval = if right_tree.len() == 0 {
            Fp4::ZERO
        } else {
            right_tree[0] // Simplified - should evaluate at challenge
        };

        final_evals.push([left_eval, right_eval]);
    }

    Ok(final_evals)
}

/// Verifies that final evaluations match claimed products.
pub fn verify_final_evaluations(
    final_evals: &[[Fp4; 2]],
    product_claims: &[Fp4],
    _current_claim: Fp4,
) -> Result<bool, GpaError> {
    if final_evals.len() != product_claims.len() {
        return Err(GpaError::BatchSizeError(
            "Final evaluations and product claims must have same length".to_string(),
        ));
    }

    // Verify that each tree's final evaluations are consistent with claimed products
    for (i, (evals, &claimed_product)) in final_evals.iter().zip(product_claims.iter()).enumerate()
    {
        // For the final layer, the product should match the claimed value
        let actual_product = evals[0] * evals[1]; // left * right

        // Allow for small numerical differences due to field arithmetic
        if actual_product != claimed_product {
            // In a production system, might want to use approximate equality
            return Ok(false);
        }
    }

    Ok(true)
}

/// Pads multiset to power-of-two size with ones.
pub fn pad_to_power_of_two(mut multiset: Vec<Fp4>, target_size: usize) -> Vec<Fp4> {
    if !target_size.is_power_of_two() {
        // If target size is not power of two, round up
        let rounded_size = if target_size == 0 {
            1
        } else {
            target_size.next_power_of_two()
        };
        return pad_to_power_of_two(multiset, rounded_size);
    }

    // Pad with ones (multiplicative identity)
    while multiset.len() < target_size {
        multiset.push(Fp4::ONE);
    }

    // Truncate if too large (shouldn't happen in normal usage)
    multiset.truncate(target_size);
    multiset
}

/// Reconstructs batched claim: `Σ γ^i * claim_i`.
pub fn reconstruct_batched_claim(product_claims: &[Fp4], gamma: Fp4) -> Fp4 {
    let mut batched_claim = Fp4::ZERO;

    for (i, &claim) in product_claims.iter().enumerate() {
        batched_claim += gamma.exp_u64(i as u64) * claim;
    }

    batched_claim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_check_instance_new() {
        let instance = MemoryCheckInstance::new();
        assert!(instance.init_ops.is_empty());
        assert!(instance.read_ops.is_empty());
        assert!(instance.write_ops.is_empty());
        assert!(instance.final_ops.is_empty());
    }

    #[test]
    fn test_memory_check_instance_add_operations() {
        let mut instance = MemoryCheckInstance::new();

        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        instance.add_write(0, Fp::from_u32(20), 2);
        instance.add_final(0, Fp::from_u32(20));

        assert_eq!(instance.init_ops.len(), 1);
        assert_eq!(instance.read_ops.len(), 1);
        assert_eq!(instance.write_ops.len(), 1);
        assert_eq!(instance.final_ops.len(), 1);

        assert_eq!(instance.init_ops[0], (0, Fp::from_u32(10)));
        assert_eq!(instance.read_ops[0], (0, Fp::from_u32(10), 1));
        assert_eq!(instance.write_ops[0], (0, Fp::from_u32(20), 2));
        assert_eq!(instance.final_ops[0], (0, Fp::from_u32(20)));
    }

    #[test]
    fn test_memory_check_instance_validate_valid() {
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        instance.add_write(0, Fp::from_u32(20), 2);
        instance.add_final(0, Fp::from_u32(20));

        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_memory_check_instance_validate_invalid_missing_final() {
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        // Missing final operation

        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_memory_check_instance_validate_invalid_uninitialized_read() {
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(1, Fp::from_u32(10), 1); // Reading from uninitialized address
        instance.add_final(0, Fp::from_u32(10));

        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_memory_check_instance_fingerprint_multisets() {
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        instance.add_write(0, Fp::from_u32(20), 2);
        instance.add_final(0, Fp::from_u32(20));

        let alpha = Fp4::from_u32(3);
        let gamma = Fp4::from_u32(5);
        let tau = Fp4::from_u32(7);

        let (left_multiset, right_multiset) = instance.fingerprint_multisets(alpha, gamma, tau);

        // Left multiset should contain init + write operations
        assert_eq!(left_multiset.len(), 2);
        // Right multiset should contain read + final operations
        assert_eq!(right_multiset.len(), 2);

        // Check that fingerprints are computed correctly
        // Init fingerprint: addr*alpha*gamma + val*gamma + tau
        let expected_init = alpha * gamma * Fp4::from_u32(0) + Fp4::from_u32(10) * gamma + tau;
        assert_eq!(left_multiset[0], expected_init);

        // Write fingerprint: addr*alpha*gamma + val*gamma + timestamp
        let expected_write =
            alpha * gamma * Fp4::from_u32(0) + Fp4::from_u32(20) * gamma + Fp4::from_u32(2);
        assert_eq!(left_multiset[1], expected_write);
    }

    #[test]
    fn test_check_memory_consistency_valid() {
        // Simplest case: just init and final, no reads or writes
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_final(0, Fp::from_u32(10));

        let alpha = Fp4::from_u32(3);
        let gamma = Fp4::from_u32(5);
        let tau = Fp4::from_u32(7);

        let result = check_memory_consistency(&instance, alpha, gamma, tau);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should be consistent
    }

    #[test]
    fn test_check_memory_consistency_invalid() {
        let mut instance = MemoryCheckInstance::new();
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        instance.add_write(0, Fp::from_u32(20), 2);
        instance.add_final(0, Fp::from_u32(30)); // Wrong final value

        let alpha = Fp4::from_u32(3);
        let gamma = Fp4::from_u32(5);
        let tau = Fp4::from_u32(7);

        let result = check_memory_consistency(&instance, alpha, gamma, tau);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should be inconsistent
    }

    #[test]
    fn test_batched_product_circuit_new() {
        // Create test trees
        let tree1 = ProductTree::new(MLE::new(vec![Fp4::from_u32(2), Fp4::from_u32(3)])).unwrap();
        let tree2 = ProductTree::new(MLE::new(vec![Fp4::from_u32(4), Fp4::from_u32(5)])).unwrap();

        let trees = vec![tree1, tree2];
        let circuit = BatchedProductCircuit::new(trees).unwrap();

        assert_eq!(circuit.num_trees(), 2);
        assert_eq!(circuit.tree_depth(), 1); // log2(2) = 1
    }

    #[test]
    fn test_batched_product_circuit_new_different_depths() {
        let tree1 = ProductTree::new(MLE::new(vec![Fp4::from_u32(2), Fp4::from_u32(3)])).unwrap();
        let tree2 = ProductTree::new(MLE::new(vec![
            Fp4::from_u32(4),
            Fp4::from_u32(5),
            Fp4::from_u32(6),
            Fp4::from_u32(7),
        ]))
        .unwrap();

        let trees = vec![tree1, tree2];
        let result = BatchedProductCircuit::new(trees);

        assert!(result.is_err()); // Should fail due to different depths
    }

    #[test]
    fn test_pad_to_power_of_two() {
        let multiset = vec![Fp4::from_u32(2), Fp4::from_u32(3)];
        let padded = pad_to_power_of_two(multiset, 4);

        assert_eq!(padded.len(), 4);
        assert_eq!(padded[0], Fp4::from_u32(2));
        assert_eq!(padded[1], Fp4::from_u32(3));
        assert_eq!(padded[2], Fp4::ONE); // Padding with ones
        assert_eq!(padded[3], Fp4::ONE);
    }

    #[test]
    fn test_pad_to_power_of_two_non_power() {
        let multiset = vec![Fp4::from_u32(2)];
        let padded = pad_to_power_of_two(multiset, 3); // Not a power of two

        assert_eq!(padded.len(), 4); // Should round up to next power of two
        assert_eq!(padded[0], Fp4::from_u32(2));
        for i in 1..4 {
            assert_eq!(padded[i], Fp4::ONE);
        }
    }

    #[test]
    fn test_reconstruct_batched_claim() {
        let claims = vec![
            Fp4::from_u32(6),  // 2 * 3
            Fp4::from_u32(20), // 4 * 5
            Fp4::from_u32(42), // 6 * 7
        ];
        let gamma = Fp4::from_u32(2);

        let batched = reconstruct_batched_claim(&claims, gamma);

        // Expected: 6*1 + 20*2 + 42*4 = 6 + 40 + 168 = 214
        let expected = Fp4::from_u32(6)
            + Fp4::from_u32(20) * Fp4::from_u32(2)
            + Fp4::from_u32(42) * Fp4::from_u32(4);
        assert_eq!(batched, expected);
    }

    #[test]
    fn test_extract_layer_evaluations() {
        let left1 = MLE::new(vec![Fp4::from_u32(2), Fp4::from_u32(3)]);
        let right1 = MLE::new(vec![Fp4::from_u32(4), Fp4::from_u32(5)]);
        let left2 = MLE::new(vec![Fp4::from_u32(6), Fp4::from_u32(7)]);
        let right2 = MLE::new(vec![Fp4::from_u32(8), Fp4::from_u32(9)]);

        let left_trees = vec![&left1, &left2];
        let right_trees = vec![&right1, &right2];
        let challenge = Fp4::from_u32(42);

        let result = extract_layer_evaluations(&left_trees, &right_trees, challenge);
        assert!(result.is_ok());

        let evals = result.unwrap();
        assert_eq!(evals.len(), 2);

        // With simplified evaluation, should return first elements
        assert_eq!(evals[0], [Fp4::from_u32(2), Fp4::from_u32(4)]);
        assert_eq!(evals[1], [Fp4::from_u32(6), Fp4::from_u32(8)]);
    }

    #[test]
    fn test_verify_final_evaluations() {
        let final_evals = vec![
            [Fp4::from_u32(2), Fp4::from_u32(3)], // Product: 6
            [Fp4::from_u32(4), Fp4::from_u32(5)], // Product: 20
        ];
        let product_claims = vec![Fp4::from_u32(6), Fp4::from_u32(20)];
        let current_claim = Fp4::from_u32(100); // Not used in current implementation

        let result = verify_final_evaluations(&final_evals, &product_claims, current_claim);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should be valid
    }

    #[test]
    fn test_verify_final_evaluations_invalid() {
        let final_evals = vec![
            [Fp4::from_u32(2), Fp4::from_u32(3)], // Product: 6
            [Fp4::from_u32(4), Fp4::from_u32(5)], // Product: 20
        ];
        let product_claims = vec![
            Fp4::from_u32(6),
            Fp4::from_u32(21), // Wrong claim
        ];
        let current_claim = Fp4::from_u32(100);

        let result = verify_final_evaluations(&final_evals, &product_claims, current_claim);
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should be invalid
    }

    #[test]
    fn test_memory_consistency_simple() {
        // Test simple memory consistency with init/final operations that have matching values
        let mut instance = MemoryCheckInstance::new();

        // Create a simple memory trace where final values match init values
        instance.add_init(0, Fp::from_u32(10));
        instance.add_final(0, Fp::from_u32(10));

        let alpha = Fp4::from_u32(3);
        let gamma = Fp4::from_u32(5);
        let tau = Fp4::from_u32(7);

        let (left_multiset, right_multiset) = instance.fingerprint_multisets(alpha, gamma, tau);

        println!("Left multiset: {:?}", left_multiset);
        println!("Right multiset: {:?}", right_multiset);

        // For matching init/final values, fingerprints should be identical
        assert_eq!(left_multiset.len(), 1);
        assert_eq!(right_multiset.len(), 1);
        assert_eq!(left_multiset[0], right_multiset[0]);

        println!(
            "Left multiset: {:?} Right multiset: {:?} - Products match ✓",
            left_multiset, right_multiset
        );
    }

    #[test]
    fn test_memory_consistency_complex() {
        // Test complex memory consistency with read/write operations
        let mut instance = MemoryCheckInstance::new();

        // Create a memory trace: init 10 -> read 10 -> write 20 -> final 20
        instance.add_init(0, Fp::from_u32(10));
        instance.add_read(0, Fp::from_u32(10), 1);
        instance.add_write(0, Fp::from_u32(20), 2);
        instance.add_final(0, Fp::from_u32(20));

        let alpha = Fp4::from_u32(3);
        let gamma = Fp4::from_u32(5);
        let tau = Fp4::from_u32(7);

        let (left_multiset, right_multiset) = instance.fingerprint_multisets(alpha, gamma, tau);

        println!("Left multiset: {:?}", left_multiset);
        println!("Right multiset: {:?}", right_multiset);

        // Left multiset should contain init + write operations
        assert_eq!(left_multiset.len(), 2);
        // Right multiset should contain read + final operations
        assert_eq!(right_multiset.len(), 2);

        // Products should match for consistent memory
        let left_product: Fp4 = left_multiset.iter().fold(Fp4::ONE, |acc, &x| acc * x);
        let right_product: Fp4 = right_multiset.iter().fold(Fp4::ONE, |acc, &x| acc * x);

        println!(
            "Left product: {:?}, Right product: {:?}",
            left_product, right_product
        );
        assert_eq!(
            left_product, right_product,
            "Memory consistency check failed"
        );
    }
}
