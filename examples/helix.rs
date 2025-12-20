use helix::{
    BaseFoldConfig, BatchSumCheckProof, Challenger, Fp, MLE, build_default_poseidon2_instance,
    build_poseidon2_witness_matrix_from_states,
};
use p3_baby_bear::default_babybear_poseidon2_16;
use p3_field::PrimeCharacteristicRing;
use p3_monty_31::dft::RecursiveDft;
use rand::{Rng, thread_rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn main() -> anyhow::Result<()> {
    let _guard = tracing_profile::init_tracing().expect("Tracing failed");
    let rate = [Fp::ONE, Fp::TWO];
    let instance = build_default_poseidon2_instance(&rate, None)
        .expect("Poseidon2 instance construction should succeed");
    let poseidon = default_babybear_poseidon2_16();

    let r1cs = instance.r1cs;
    let dft = RecursiveDft::new(1 << 25);
    for vars in 7..15 {
        tracing::info!("Vars 2^{vars}");
        let num_states = 1usize << vars;
        let initial_states = (0..num_states)
            .into_par_iter()
            .map(|_| Fp::new_array(thread_rng().r#gen()))
            .collect::<Vec<_>>();
        let witness_matrix =
            build_poseidon2_witness_matrix_from_states(&initial_states, &poseidon).unwrap();

        let z_transposed = MLE::new(witness_matrix.flattened_transpose());

        let config = BaseFoldConfig::new()
            .with_early_stopping(11)
            .with_round_skip(6);
        let roots = Fp::roots_of_unity_table(1 << (z_transposed.n_vars() + 1));

        let (commitment, prover_data) =
            BatchSumCheckProof::commit_skip(&z_transposed, &dft, &config).unwrap();
        let mut prover_challenger = Challenger::new();
        let _prove_span = tracing::info_span!(
            "batch_sumcheck_prove",
            vars,
            num_states,
            witness_height = z_transposed.n_vars()
        );
        let (proof, round_challenges) = BatchSumCheckProof::prove(
            &r1cs.a,
            &r1cs.b,
            &r1cs.c,
            &z_transposed,
            &commitment,
            &prover_data,
            &roots,
            &config,
            &mut prover_challenger,
        )?;

        let proof_bytes = serde_json::to_vec(&proof)?;

        println!("Proof size = {:?} bytes", proof_bytes.len());
        assert_eq!(round_challenges.len(), z_transposed.n_vars());

        let mut verifier_challenger = Challenger::new();
        let verify_span = tracing::info_span!(
            "batch_sumcheck_verify",
            vars,
            num_states,
            witness_height = z_transposed.n_vars()
        );
        let _verify_guard = verify_span.enter();
        let (verified_challenges, _final_evals) = proof.verify(
            &r1cs.a,
            &r1cs.b,
            &r1cs.c,
            commitment,
            &roots,
            &mut verifier_challenger,
            &config,
        )?;
        drop(_verify_guard);

        assert_eq!(verified_challenges, round_challenges);
    }
    Ok(())
}
