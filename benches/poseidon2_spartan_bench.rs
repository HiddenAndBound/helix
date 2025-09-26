use criterion::{ Criterion, black_box, criterion_group, criterion_main };
use helix::{ challenger::Challenger, spartan::{ SpartanProof, build_default_poseidon2_instance } };
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;

fn poseidon2_spartan_bench(c: &mut Criterion) {
    let rate = [BabyBear::ONE, BabyBear::TWO];
    let instance = build_default_poseidon2_instance(&rate, None).expect(
        "Poseidon2 instance construction should succeed"
    );

    c.bench_function("poseidon2_prover", |b| {
        b.iter(|| {
            let mut challenger = Challenger::new();
            let output = SpartanProof::prove_poseidon2(&instance, &mut challenger).expect(
                "Poseidon2 proof generation should succeed"
            );
            black_box(output);
        });
    });

    let mut prover_challenger = Challenger::new();
    let (proof, commitment) = SpartanProof::prove_poseidon2(
        &instance,
        &mut prover_challenger
    ).expect("Poseidon2 proof generation should succeed");

    let proof_size_bytes = std::mem::size_of_val(&proof);
    println!("SpartanProof size: {} bytes", proof_size_bytes);

    c.bench_function("poseidon2_verifier", |b| {
        b.iter(|| {
            let mut verifier_challenger = Challenger::new();
            proof
                .verify(commitment.clone(), &mut verifier_challenger)
                .expect("Poseidon2 proof verification should succeed");
        });
    });
}

criterion_group!(poseidon2, poseidon2_spartan_bench);
criterion_main!(poseidon2);
