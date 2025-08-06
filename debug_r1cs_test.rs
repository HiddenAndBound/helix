// Debug script to check the witness creation
use deep_fri::spartan::r1cs::R1CS;

fn main() {
    let (r1cs, witness) = R1CS::multi_constraint_test_instance().unwrap();
    
    println!("R1CS constraints: {}", r1cs.num_constraints);
    println!("R1CS variables: {}", r1cs.num_variables);
    println!("R1CS public inputs: {}", r1cs.num_public_inputs);
    println!("Witness public inputs: {}", witness.public_inputs.len());
    println!("Witness private variables: {}", witness.private_variables.len());
    println!("Total witness length: {}", witness.len());
    
    // Check if the witness is valid
    let z = witness.to_mle();
    println!("Witness MLE length: {}", z.len());
    
    // Verify constraints
    let result = r1cs.verify(&z);
    println!("Verification result: {:?}", result);
}