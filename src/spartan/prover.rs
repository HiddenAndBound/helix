//! Spartan zkSNARK prover implementation.
//!
//! Spartan provides zero-knowledge proofs for R1CS instances without trusted setup.
//! Uses sum-check protocols for efficient proving with logarithmic verification time.

use p3_baby_bear::BabyBear;

use crate::{
    challenger::Challenger,
    pcs::{ BaseFoldConfig, Basefold, BasefoldCommitment, EvalProof },
    spartan::{ Poseidon2Instance, R1CSInstance},
};

#[cfg(test)]
mod tests {
    use core::time;
    use std::{ collections::HashMap, time::Instant };

    use crate::{
        challenger::Challenger,
        pcs::{ BaseFoldConfig, Basefold },
        polynomial::MLE,
        sparse::SparseMLE,
        spartan::{
            Poseidon2ColumnSeed,
            build_default_poseidon2_instance,
            build_poseidon2_witness_matrix_from_states,
            r1cs::{ R1CS, R1CSInstance, Witness },
            sumcheck::batch_sumcheck::BatchSumCheckProof,
        },
        *,
    };
    use anyhow::bail;
    use itertools::multizip;
    use p3_baby_bear::default_babybear_poseidon2_16;
    use p3_dft::*;
    use p3_field::{ Field, PrimeCharacteristicRing };
    use p3_monty_31::dft;
    use rand::{ Rng, SeedableRng, rngs::StdRng, thread_rng };
    use rayon::iter::{ IntoParallelIterator, ParallelIterator };
    use serde::Serialize;
    use serde_json::Serializer;
    use tracing::{ span, subscriber::set_global_default, Level };
    use tracing_subscriber::{ fmt::format::FmtSpan, util::SubscriberInitExt };

    #[test]
    fn helix_round_trip() -> anyhow::Result<()> {
        let mut rng = StdRng::seed_from_u64(0);

        const COLS: usize = 1 << 10;
        let poseidon = default_babybear_poseidon2_16();
        let initial_states = (0..COLS).map(|_| Fp::new_array(rng.r#gen())).collect::<Vec<_>>();
        let witness_matrix = build_poseidon2_witness_matrix_from_states(
            &initial_states,
            &poseidon
        )?;

        let instance = build_default_poseidon2_instance(&[Fp::ZERO, Fp::ONE], None)?;

        let r1cs = instance.r1cs;
        let z_transposed = MLE::new(witness_matrix.flattened_transpose());

        let config = BaseFoldConfig::new();
        let roots = Fp::roots_of_unity_table(1 << (z_transposed.n_vars() + 1));
        let (commitment, prover_data) = Basefold::commit(&z_transposed, &roots, &config)?;
        let mut prover_challenger = Challenger::new();
        let (proof, round_challenges) = BatchSumCheckProof::prove(
            &r1cs.a,
            &r1cs.b,
            &r1cs.c,
            &z_transposed,
            &commitment,
            &prover_data,
            &roots,
            &config,
            &mut prover_challenger
        )?;

        assert_eq!(round_challenges.len(), z_transposed.n_vars());

        let mut verifier_challenger = Challenger::new();
        let (verified_challenges, final_evals) = proof.verify(
            &r1cs.a,
            &r1cs.b,
            &r1cs.c,
            commitment,
            &roots,
            &mut verifier_challenger,
            &config
        )?;

        assert_eq!(verified_challenges, round_challenges);
        Ok(())
    }

    #[test]
    fn helix_timing() {
        use tracing::{ Subscriber, span, Level };
        use tracing::{ debug, info };
        let rate = [Fp::ONE, Fp::TWO];
        let instance = build_default_poseidon2_instance(&rate, None).expect(
            "Poseidon2 instance construction should succeed"
        );
        let poseidon = default_babybear_poseidon2_16();

        let r1cs = instance.r1cs;

        let sub = tracing_subscriber
            ::fmt()
            // enable everything
            .with_max_level(tracing::Level::TRACE)
            // sets this to be the default, global subscriber for this application.
            .finish();

        set_global_default(sub).expect("Setting subscriber failed");
        for vars in 5..20 {
            tracing::info!("Vars 2^{vars}");
            let initial_states = (0..1 << vars)
                .into_par_iter()
                .map(|_| Fp::new_array(thread_rng().r#gen()))
                .collect::<Vec<_>>();
            let witness_matrix = build_poseidon2_witness_matrix_from_states(
                &initial_states,
                &poseidon
            ).unwrap();

            let z_transposed = MLE::new(witness_matrix.flattened_transpose());

            let config = BaseFoldConfig::new().with_early_stopping(11);
            let roots = Fp::roots_of_unity_table(1 << (z_transposed.n_vars() + 1));

            let (commitment, prover_data) = Basefold::commit(
                &z_transposed,
                &roots,
                &config
            ).unwrap();
            let mut prover_challenger = Challenger::new();
            let (proof, round_challenges) = BatchSumCheckProof::prove(
                &r1cs.a,
                &r1cs.b,
                &r1cs.c,
                &z_transposed,
                &commitment,
                &prover_data,
                &roots,
                &config,
                &mut prover_challenger
            ).unwrap();

            assert_eq!(round_challenges.len(), z_transposed.n_vars());

            let proof_bytes = serde_json::to_vec(&proof).unwrap();
            println!("Proof size {:?} bytes", proof_bytes.len());
            let mut verifier_challenger = Challenger::new();
            let (verified_challenges, final_evals) = proof
                .verify(
                    &r1cs.a,
                    &r1cs.b,
                    &r1cs.c,
                    commitment,
                    &roots,
                    &mut verifier_challenger,
                    &config
                )
                .unwrap();

            assert_eq!(verified_challenges, round_challenges);
        }
    }
}
