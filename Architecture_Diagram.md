# Sum-Check Module Architecture Diagram

## Current vs. Proposed Architecture

### Current Architecture
```mermaid
graph TD
    A[SumcheckProver] --> B[MLE<BabyBear>]
    A --> C[Challenger]
    A --> D[compute_round_polynomial]
    D --> E[fold_in_place]
    
    F[SumcheckVerifier] --> G[SumcheckProof]
    F --> C
    F --> H[verify]
    
    B --> I[utils/polynomial.rs]
    C --> J[utils/challenger.rs]
    
    style A fill:#ffcccc
    style F fill:#ccffcc
    style B fill:#ccccff
```

### Proposed Hybrid Architecture
```mermaid
graph TD
    subgraph "Public API Layer"
        A[SumcheckProver<F,C,L>] 
        B[SumcheckVerifier<F,C,L>]
        C[LegacySumcheckProver]
        D[LegacySumcheckVerifier]
    end
    
    subgraph "Configuration Layer"
        E[SumcheckConfig]
        E --> E1[max_degree]
        E --> E2[use_small_opt]
        E --> E3[use_karatsuba]
        E --> E4[skip_threshold]
    end
    
    subgraph "Trait Abstractions"
        F[ChallengeGenerator<F>]
        G[LowDegreeComposer<F>]
        H[FieldExt<F>]
    end
    
    subgraph "Optimization Suite"
        I[SmallValueOptimizer]
        J[KaratsubaOptimizer]
        K[SkipRoundOptimizer]
    end
    
    subgraph "Composer Implementations"
        L[SchoolbookComposer]
        M[ToomCookComposer]
    end
    
    subgraph "Core Utilities"
        N[MLE<F>]
        O[Challenger]
        P[EqEvals]
    end
    
    A --> E
    A --> F
    A --> G
    A --> I
    A --> J
    A --> K
    
    B --> E
    B --> F
    B --> G
    
    C --> A
    D --> B
    
    G --> L
    G --> M
    
    F --> O
    
    I --> H
    J --> H
    K --> H
    
    A --> N
    B --> N
    N --> P
    
    style A fill:#ff9999
    style B fill:#99ff99
    style C fill:#ffcccc
    style D fill:#ccffcc
    style E fill:#ffff99
    style I fill:#99ccff
    style J fill:#99ccff
    style K fill:#99ccff
```

## Optimization Integration Flow

### Small-Value Field Splits (ePrint 2025/1117)
```mermaid
graph LR
    A[MLE Input] --> B{Check Coefficients}
    B -->|Small Values| C[Base Field Operations]
    B -->|Large Values| D[Extension Field Operations]
    C --> E[2-4x Speedup]
    D --> F[Standard Performance]
    
    style C fill:#90EE90
    style E fill:#90EE90
```

### Karatsuba-Style Reduction (ePrint 2024/1046)
```mermaid
graph TD
    A[Degree-d Composition] --> B[Evaluate at d+1 Points]
    B --> C[Multiply Base Products]
    C --> D[Interpolate Univariate Poly]
    D --> E[Reduced from 2^d to d+1 Operations]
    
    F[Memory Usage] --> G[O(d·t) vs O(2^(d·t))]
    
    style E fill:#87CEEB
    style G fill:#87CEEB
```

### Unequal-Degree Skip (ePrint 2024/108)
```mermaid
graph TD
    A[Round i] --> B{Degree Analysis}
    B -->|Dominant Degree| C[Skip Full Expansion]
    B -->|Balanced Degrees| D[Standard Sum-Check]
    C --> E[Sample One Challenge]
    D --> F[Compute Full Polynomial]
    
    G[round >= skip_threshold] --> B
    
    style C fill:#DDA0DD
    style E fill:#DDA0DD
```

## Data Flow Architecture

### Prover Flow with Optimizations
```mermaid
sequenceDiagram
    participant P as Prover
    participant C as Config
    participant O as Optimizers
    participant M as MLE
    participant Ch as Challenger
    
    P->>C: Check optimization flags
    P->>M: Analyze MLE characteristics
    
    loop For each round
        P->>O: Select optimization strategy
        alt Small values detected
            O->>M: Apply small-value optimization
        else High degree detected
            O->>M: Apply Karatsuba reduction
        else Unequal degrees
            O->>M: Apply skip-round optimization
        else Standard case
            P->>M: Standard round computation
        end
        
        P->>Ch: Observe round polynomial
        Ch->>P: Return challenge
        P->>M: Fold with challenge
    end
    
    P->>P: Generate proof
```

### Verifier Flow with Optimization Awareness
```mermaid
sequenceDiagram
    participant V as Verifier
    participant C as Config
    participant P as Proof
    participant Ch as Challenger
    participant M as MLE Commitment
    
    V->>C: Load same configuration as prover
    V->>P: Parse proof structure
    
    loop For each round
        V->>C: Check which optimization was used
        V->>P: Verify round polynomial consistency
        V->>Ch: Observe round polynomial
        Ch->>V: Return challenge
        V->>V: Update running sum
    end
    
    V->>M: Final evaluation check
    V->>V: Return verification result
```

## Module Dependencies

### Dependency Graph
```mermaid
graph TD
    subgraph "External Dependencies"
        A[p3-baby-bear]
        B[p3-field]
        C[blake3]
        D[rayon]
    end
    
    subgraph "Core Modules"
        E[utils/mod.rs]
        F[utils/polynomial.rs]
        G[utils/challenger.rs]
        H[utils/eq.rs]
    end
    
    subgraph "Sum-Check Modules"
        I[sumcheck/mod.rs]
        J[sumcheck/config.rs]
        K[sumcheck/traits/]
        L[sumcheck/optimizations/]
        M[sumcheck/composers/]
        N[sumcheck/prover.rs]
        O[sumcheck/verifier.rs]
        P[sumcheck/legacy.rs]
    end
    
    A --> E
    B --> E
    C --> G
    D --> L
    
    E --> F
    E --> G
    E --> H
    
    F --> I
    G --> I
    H --> I
    
    J --> I
    K --> I
    L --> I
    M --> I
    
    K --> N
    K --> O
    L --> N
    M --> N
    
    I --> P
    N --> P
    O --> P
    
    style A fill:#FFE4B5
    style B fill:#FFE4B5
    style C fill:#FFE4B5
    style D fill:#FFE4B5
    style L fill:#98FB98
    style M fill:#98FB98
    style N fill:#87CEFA
    style O fill:#87CEFA
```

## Performance Optimization Decision Tree

```mermaid
graph TD
    A[Input MLE] --> B{Check MLE Size}
    B -->|Small < 1K| C[Use Schoolbook Composer]
    B -->|Large >= 1K| D{Check Coefficient Values}
    
    D -->|Small Values| E[Enable Small-Value Optimization]
    D -->|Mixed/Large| F{Check Degree}
    
    F -->|Degree <= 2| G[Use Standard Algorithm]
    F -->|Degree > 2| H[Enable Karatsuba Optimization]
    
    E --> I{Check Round}
    G --> I
    H --> I
    
    I -->|Round >= Threshold| J{Check Degree Balance}
    I -->|Round < Threshold| K[Standard Round]
    
    J -->|Unbalanced| L[Enable Skip-Round Optimization]
    J -->|Balanced| K
    
    C --> M[Execute with Schoolbook]
    E --> N[Execute with Small-Value]
    H --> O[Execute with Karatsuba]
    L --> P[Execute with Skip-Round]
    K --> Q[Execute Standard]
    
    style E fill:#90EE90
    style H fill:#87CEEB
    style L fill:#DDA0DD
    style N fill:#90EE90
    style O fill:#87CEEB
    style P fill:#DDA0DD
```

## Memory Layout Optimization

### Standard vs. Optimized Memory Usage
```mermaid
graph LR
    subgraph "Standard Implementation"
        A[MLE Coefficients: 2^n]
        B[Round Polynomials: n × degree]
        C[Intermediate Results: 2^d]
        D[Total: O(2^n + n×d + 2^d)]
    end
    
    subgraph "Optimized Implementation"
        E[MLE Coefficients: 2^n]
        F[Round Polynomials: n × degree]
        G[Toom-Cook Cache: d+1]
        H[Small-Value Cache: O(1)]
        I[Total: O(2^n + n×d + d)]
    end
    
    A --> E
    B --> F
    C --> G
    C --> H
    
    style G fill:#87CEEB
    style H fill:#90EE90
    style I fill:#FFD700
```

This architecture diagram illustrates the comprehensive hybrid approach that preserves existing functionality while adding advanced optimizations through a clean, modular design.