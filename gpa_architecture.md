# Batched Grand Product Argument Architecture

## Overview

This document describes the architecture for implementing a **Batched Grand Product Argument (GPA)** system that extends the existing gamma batching patterns in the Helix codebase. The GPA enables efficient proof of multiple grand product claims simultaneously, with applications to offline memory checking in zero-knowledge proofs.

## Context and Motivation

### Memory Checking Protocol
The GPA is designed for **offline memory checking** where:
- Memory operations are fingerprinted: `h_τ,γ(addr, val, t) = α·γ + val·γ + t`
- Two multisets are formed: **I∪W** (init + writes) and **R∪F** (reads + final)
- **Memory consistency** is proven by showing: `H_τ,γ(I ∪ W) = H_τ,γ(R ∪ F)`
- Products are computed via **binary tree circuits** for logarithmic complexity

### Existing Batching Patterns
The codebase already implements gamma batching in:
- **InnerSumCheck**: 3 claims with `gamma`, `gamma²`, `gamma³`
- **SparkSumCheck**: 3 matrix evaluations with same gamma powers
- **Layer-by-layer processing**: Single sumcheck per round handles all batched claims

## Core Architecture

### 1. Gamma Powers Extension

#### Current Pattern (3 instances):
```rust
// InnerSumCheck and SparkSumCheck
current_claim = gamma * claim_0 + gamma.square() * claim_1 + gamma.cube() * claim_2;
```

#### Extended Pattern (n instances):
```rust
// Batched GPA for n product claims
let mut current_claim = Fp4::ZERO;
for (i, claim) in product_claims.iter().enumerate() {
    current_claim += gamma.pow(&[i as u64]) * claim;
}
```

#### Mathematical Foundation:
- **Linear Combination**: Multiple product claims combined using random coefficients
- **Soundness**: If any individual claim is false, batched claim fails with high probability
- **Efficiency**: Single proof for n claims vs n individual proofs

### 2. Data Structures

#### Core Components:
```rust
/// Single binary product tree
pub struct ProductTree {
    /// Left and right halves for each layer (root to leaves)
    left_layers: Vec<MLE<Fp4>>,
    right_layers: Vec<MLE<Fp4>>,
    /// Tree depth (log of input size)
    depth: usize,
}

/// Multiple trees of the same size for batching
pub struct BatchedProductCircuit {
    /// All product trees (must have same size)
    trees: Vec<ProductTree>,
    /// Number of trees being batched
    num_trees: usize,
    /// Shared tree depth
    tree_depth: usize,
}

/// Proof for one layer across all batched trees
pub struct BatchedLayerProof {
    /// Sumcheck proof for this layer
    round_proofs: Vec<UnivariatePoly>,
    /// Final evaluations: [left_eval, right_eval] per tree
    final_evals: Vec<[Fp4; 2]>,
}

/// Complete batched product proof
pub struct BatchedProductProof {
    /// One proof per layer (from root to leaves)  
    layer_proofs: Vec<BatchedLayerProof>,
    /// Final product claims for verification
    product_claims: Vec<Fp4>,
}
```

### 3. Layer-by-Layer Processing

#### Batched Layer Computation:
```rust
pub fn compute_gpa_round_batched(
    // Left halves for all trees at current layer
    left_trees: &[MLE<Fp4>], 
    // Right halves for all trees at current layer
    right_trees: &[MLE<Fp4>],
    // Equality polynomial for accumulated randomness
    eq_poly: &MLE<Fp4>,
    // Batching parameter
    gamma: Fp4,
    // Current batched claim
    current_claim: Fp4,
    round: usize,
    rounds: usize,
) -> UnivariatePoly {
    let mut round_coeffs = vec![Fp4::ZERO; 3];
    let num_trees = left_trees.len();
    
    // Process all trees for this layer simultaneously
    for i in 0..1 << (rounds - round - 1) {
        // g(0): evaluate at X=0, batch with gamma powers
        let mut term_0 = Fp4::ZERO;
        for tree_idx in 0..num_trees {
            let gamma_power = gamma.pow(&[tree_idx as u64]);
            let tree_contribution = left_trees[tree_idx][i << 1] 
                                  * right_trees[tree_idx][i << 1] 
                                  * eq_poly[i];
            term_0 += gamma_power * tree_contribution;
        }
        round_coeffs[0] += term_0;
        
        // g(2): evaluate at X=2 using multilinear identity
        let mut term_2 = Fp4::ZERO;
        for tree_idx in 0..num_trees {
            let gamma_power = gamma.pow(&[tree_idx as u64]);
            let left_2 = left_trees[tree_idx][i << 1] 
                       + left_trees[tree_idx][i << 1 | 1].double();
            let right_2 = right_trees[tree_idx][i << 1] 
                        + right_trees[tree_idx][i << 1 | 1].double();
            let tree_contribution = left_2 * right_2 * eq_poly[i];
            term_2 += gamma_power * tree_contribution;
        }
        round_coeffs[2] += term_2;
    }
    
    // g(1): derived from sumcheck constraint
    round_coeffs[1] = current_claim - round_coeffs[0];
    
    UnivariatePoly::new(round_coeffs).unwrap()
}
```

### 4. Protocol Flow

#### Phase 1: Tree Construction
```rust
impl ProductTree {
    pub fn new(input: MLE<Fp4>) -> Self {
        let mut left_layers = Vec::new();
        let mut right_layers = Vec::new();
        
        // Start with input vector
        let mut current_layer = input;
        let depth = current_layer.n_vars();
        
        // Build layers bottom-up
        for layer in 0..depth {
            // Split current layer in half
            let (left_half, right_half) = split_layer(&current_layer);
            left_layers.push(left_half.clone());
            right_layers.push(right_half.clone());
            
            // Compute next layer: element-wise multiplication
            current_layer = multiply_layer_halves(&left_half, &right_half);
        }
        
        Self { left_layers, right_layers, depth }
    }
}
```

#### Phase 2: Batched Proving
```rust
impl BatchedProductCircuit {
    pub fn prove_batched(
        trees: Vec<ProductTree>,
        gamma: Fp4,
        challenger: &mut Challenger,
    ) -> BatchedProductProof {
        let num_trees = trees.len();
        let tree_depth = trees[0].depth;
        
        // Extract individual product claims
        let product_claims: Vec<Fp4> = trees.iter()
            .map(|tree| tree.get_root_value())
            .collect();
        
        // Batch claims using gamma powers
        let mut current_claim = Fp4::ZERO;
        for (i, &claim) in product_claims.iter().enumerate() {
            current_claim += gamma.pow(&[i as u64]) * claim;
        }
        
        let mut layer_proofs = Vec::new();
        let mut accumulated_randomness = Vec::new();
        
        // Process each layer from root to leaves
        for layer in 0..tree_depth {
            // Extract left and right halves for all trees at this layer
            let left_trees: Vec<_> = trees.iter()
                .map(|tree| &tree.left_layers[layer])
                .collect();
            let right_trees: Vec<_> = trees.iter()
                .map(|tree| &tree.right_layers[layer])
                .collect();
                
            // Create equality polynomial from accumulated randomness
            let eq_poly = MLE::new(EqPolynomial::new(accumulated_randomness.clone()).evals());
            
            // Compute batched round polynomial
            let round_proof = compute_gpa_round_batched(
                &left_trees,
                &right_trees, 
                &eq_poly,
                gamma,
                current_claim,
                layer,
                tree_depth - layer,
            );
            
            // Add to transcript and get challenge
            challenger.observe_fp4_elems(&round_proof.coefficients());
            let challenge = challenger.get_challenge();
            
            // Update claim for next layer
            current_claim = round_proof.evaluate(challenge);
            accumulated_randomness.push(challenge);
            
            // Store proof for this layer
            layer_proofs.push(BatchedLayerProof {
                round_proofs: vec![round_proof],
                final_evals: extract_layer_evaluations(&left_trees, &right_trees, &challenge),
            });
        }
        
        BatchedProductProof { layer_proofs, product_claims }
    }
}
```

#### Phase 3: Verification
```rust
impl BatchedProductProof {
    pub fn verify(
        &self,
        gamma: Fp4,
        challenger: &mut Challenger,
    ) -> bool {
        // Reconstruct batched claim
        let mut current_claim = Fp4::ZERO;
        for (i, &claim) in self.product_claims.iter().enumerate() {
            current_claim += gamma.pow(&[i as u64]) * claim;
        }
        
        let mut accumulated_randomness = Vec::new();
        
        // Verify each layer proof
        for layer_proof in &self.layer_proofs {
            // Verify sumcheck relation for this layer
            let round_poly = &layer_proof.round_proofs[0];
            
            // Check: current_claim = g(0) + g(1)
            let expected = round_poly.evaluate(Fp4::ZERO) + round_poly.evaluate(Fp4::ONE);
            if current_claim != expected {
                return false;
            }
            
            // Get challenge and update claim
            challenger.observe_fp4_elems(&round_poly.coefficients());
            let challenge = challenger.get_challenge();
            current_claim = round_poly.evaluate(challenge);
            accumulated_randomness.push(challenge);
        }
        
        // Final verification: check claimed products match final evaluations
        verify_final_evaluations(&self.layer_proofs.last().unwrap().final_evals, 
                                &self.product_claims, gamma, current_claim)
    }
}
```

## Memory Checking Integration

### 1. Dual-Tree Memory Consistency

#### Memory Operation Types:
- **Init (I)**: Initial memory state `(addr, val)`  
- **Reads (R)**: Read operations `(addr, val, timestamp)`
- **Writes (W)**: Write operations `(addr, val, timestamp)`
- **Final (F)**: Final memory state `(addr, val)`

#### Multiset Construction:
```rust
pub struct MemoryCheckInstance {
    pub init_ops: Vec<(usize, Fp)>,
    pub read_ops: Vec<(usize, Fp, usize)>,
    pub write_ops: Vec<(usize, Fp, usize)>,
    pub final_ops: Vec<(usize, Fp)>,
}

impl MemoryCheckInstance {
    pub fn fingerprint_multisets(
        &self,
        alpha: Fp4,
        gamma: Fp4, 
        tau: Fp4,
    ) -> (Vec<Fp4>, Vec<Fp4>) {
        // Left multiset: I ∪ W
        let mut left_multiset = Vec::new();
        
        // Add init operations  
        for &(addr, val) in &self.init_ops {
            let fingerprint = alpha * gamma + Fp4::from(val) * gamma + tau;
            left_multiset.push(fingerprint);
        }
        
        // Add write operations
        for &(addr, val, ts) in &self.write_ops {
            let fingerprint = alpha * gamma + Fp4::from(val) * gamma + Fp4::from(ts as u32);
            left_multiset.push(fingerprint);
        }
        
        // Right multiset: R ∪ F  
        let mut right_multiset = Vec::new();
        
        // Add read operations
        for &(addr, val, ts) in &self.read_ops {
            let fingerprint = alpha * gamma + Fp4::from(val) * gamma + Fp4::from(ts as u32);
            right_multiset.push(fingerprint);
        }
        
        // Add final operations
        for &(addr, val) in &self.final_ops {
            let fingerprint = alpha * gamma + Fp4::from(val) * gamma + tau;
            right_multiset.push(fingerprint);
        }
        
        (left_multiset, right_multiset)
    }
}
```

### 2. Batched Memory Consistency Proof

#### Multi-Instance Memory Checking:
```rust
pub struct BatchedMemoryProof {
    /// Proof for batched left products: H_τ,γ(I ∪ W) for all instances
    left_product_proof: BatchedProductProof,
    /// Proof for batched right products: H_τ,γ(R ∪ F) for all instances  
    right_product_proof: BatchedProductProof,
    /// Individual product claims for verification
    left_claims: Vec<Fp4>,
    right_claims: Vec<Fp4>,
}

pub fn prove_batched_memory_consistency(
    memory_instances: &[MemoryCheckInstance],
    alpha: Fp4,
    gamma: Fp4,
    tau: Fp4,
    challenger: &mut Challenger,
) -> BatchedMemoryProof {
    // Step 1: Fingerprint all memory operations
    let mut left_multisets = Vec::new();
    let mut right_multisets = Vec::new();
    
    for instance in memory_instances {
        let (left, right) = instance.fingerprint_multisets(alpha, gamma, tau);
        left_multisets.push(left);
        right_multisets.push(right);
    }
    
    // Step 2: Build product trees (pad to same size if needed)
    let max_size = left_multisets.iter().map(|ms| ms.len()).max().unwrap();
    let left_trees: Vec<_> = left_multisets.into_iter()
        .map(|ms| ProductTree::new(pad_to_power_of_two(ms, max_size)))
        .collect();
    let right_trees: Vec<_> = right_multisets.into_iter()
        .map(|ms| ProductTree::new(pad_to_power_of_two(ms, max_size)))
        .collect();
    
    // Step 3: Generate batching challenge
    challenger.observe_u32(memory_instances.len() as u32);
    let batch_gamma = challenger.get_challenge();
    
    // Step 4: Prove both sides  
    let left_product_proof = BatchedProductCircuit::prove_batched(left_trees, batch_gamma, challenger);
    let right_product_proof = BatchedProductCircuit::prove_batched(right_trees, batch_gamma, challenger);
    
    // Step 5: Extract individual claims for verification
    let left_claims = left_product_proof.product_claims.clone();
    let right_claims = right_product_proof.product_claims.clone();
    
    BatchedMemoryProof {
        left_product_proof,
        right_product_proof,
        left_claims,
        right_claims,
    }
}
```

## Performance Analysis

### 1. Complexity Comparison

| Operation | Naive Approach | Binary Tree GPA | Batched GPA (n instances) |
|-----------|---------------|-----------------|---------------------------|
| **Product Computation** | O(m) per instance | O(log m) per instance | O(n · log m) total |
| **Proof Size** | O(m) per instance | O(log²m) per instance | O(log²m) total |
| **Verification** | O(m) per instance | O(log²m) per instance | O(log²m) total |
| **Communication** | O(n · m) | O(n · log²m) | O(log²m) |

Where:
- **m**: Size of each multiset
- **n**: Number of instances being batched

### 2. Batching Benefits

#### Communication Complexity:
- **Individual proofs**: n × O(log²m) field elements
- **Batched proof**: O(log²m) field elements  
- **Savings**: Factor of n reduction in communication

#### Verification Complexity:
- **Individual verification**: n × O(log²m) operations
- **Batched verification**: O(log²m) operations
- **Savings**: Factor of n reduction in verification time

#### Prover Complexity:
- **Individual proving**: n × O(m) operations  
- **Batched proving**: O(n · m) operations
- **Overhead**: Constant factor increase (acceptable)

### 3. Memory Checking Scalability

#### Large Memory Traces:
- **Memory operations**: Can handle thousands of read/write operations
- **Batching**: Prove consistency for multiple execution traces simultaneously
- **Sparse access**: Efficient handling of sparse memory access patterns

#### Practical Considerations:
- **Tree size uniformity**: All instances must have same multiset size (pad with dummy operations)
- **Field arithmetic**: All computations in Fp4 for consistency with fingerprinting
- **Challenge generation**: Proper Fiat-Shamir integration for soundness

## Implementation Checklist

### Core Components:
- [ ] `ProductTree` implementation with layer construction
- [ ] `BatchedProductCircuit` for multiple tree handling  
- [ ] `compute_gpa_round_batched()` following SparkSumCheck patterns
- [ ] `BatchedProductProof` verification logic

### Memory Integration:
- [ ] Memory operation fingerprinting functions
- [ ] Multiset construction (I∪W, R∪F)  
- [ ] Padding utilities for uniform tree sizes
- [ ] `BatchedMemoryProof` for dual-tree consistency

### Testing and Validation:
- [ ] Unit tests for single tree product computation
- [ ] Batching tests with varying numbers of instances
- [ ] Memory consistency tests with valid/invalid traces
- [ ] Performance benchmarks vs individual proofs
- [ ] Integration tests with existing Spartan infrastructure

### Documentation:
- [ ] API documentation for all public interfaces
- [ ] Usage examples for memory checking integration
- [ ] Performance characteristics and scaling guidelines
- [ ] Security considerations and assumptions

This architecture provides a complete blueprint for implementing batched grand product arguments that integrate seamlessly with the existing Helix codebase while enabling efficient proof of memory consistency at scale.