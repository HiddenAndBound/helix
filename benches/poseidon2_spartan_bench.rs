use criterion::{ Criterion, black_box, criterion_group, criterion_main };
use helix::{
    Fp,
    challenger::Challenger,
    pcs::{ BaseFoldConfig, Basefold },
    polynomial::MLE,
    helix::{
        build_default_poseidon2_instance,
        build_poseidon2_witness_matrix_from_states,
        sumcheck::batch_sumcheck::BatchSumCheckProof,
    },
};
use p3_baby_bear::{ BabyBear, default_babybear_poseidon2_16 };
use p3_field::PrimeCharacteristicRing;
use rand::{ Rng, SeedableRng, rngs::StdRng };

fn poseidon2_spartan_bench(c: &mut Criterion) {
    let rate = [BabyBear::ONE, BabyBear::TWO];
    let instance = build_default_poseidon2_instance(&rate, None).expect(
        "Poseidon2 instance construction should succeed"
    );
    let poseidon = default_babybear_poseidon2_16();
    let mut rng = StdRng::seed_from_u64(0);
    let r1cs = instance.r1cs;
    for vars in 5..20 {
        let initial_states = (0..1 << vars).map(|_| Fp::new_array(rng.r#gen())).collect::<Vec<_>>();
        let witness_matrix = build_poseidon2_witness_matrix_from_states(
            &initial_states,
            &poseidon
        ).unwrap();

        let z_transposed = MLE::new(witness_matrix.flattened_transpose());

        let config = BaseFoldConfig::new();
        let roots = Fp::roots_of_unity_table(1 << (z_transposed.n_vars() + 1));

        let (commitment, prover_data) = Basefold::commit(&z_transposed, &roots, &config).unwrap();
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

criterion_group!(poseidon2, poseidon2_spartan_bench);
criterion_main!(poseidon2);
