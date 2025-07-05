pub struct MerkleTree {
    // The root of the Merkle tree
    pub root: String,
    // The leaves of the Merkle tree
    pub leaves: Vec<[u8; 32]>,
}
