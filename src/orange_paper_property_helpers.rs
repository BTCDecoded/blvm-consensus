//! Property test helpers — re-export from blvm-primitives + VerifyConsensusCommitment stubs
//!
//! Formula helpers (TotalSupply, GetBlockSubsidy, etc.) live in blvm-primitives.
//! VerifyConsensusCommitment stubs stay here (protocol-specific; uses UtxoCommitment).
//!
//! **Not a supported production API:** functions that `panic!` or `unimplemented!` here are for
//! spec-lock / test wiring only. Production verification lives under
//! `blvm_protocol::utxo_commitments::verification`.

pub use blvm_primitives::orange_paper_helpers::*;

/// VerifyConsensusCommitment(uc, hs) — stub; use blvm_protocol::utxo_commitments::verification
#[cfg(not(feature = "utxo-commitments"))]
pub fn expected_verifyconsensuscommitment_from_orange_paper(_params: u64) -> i64 {
    panic!(
        "VerifyConsensusCommitment requires (UtxoCommitment, headers). \
         Use expected_verifyconsensuscommitment_from_orange_paper_impl with utxo-commitments feature."
    )
}

#[cfg(feature = "utxo-commitments")]
pub fn expected_verifyconsensuscommitment_from_orange_paper(_params: u64) -> i64 {
    panic!(
        "VerifyConsensusCommitment requires (UtxoCommitment, headers). \
         Use blvm_protocol::utxo_commitments::verification or blvm-spec-lock for property tests."
    )
}

/// VerifyConsensusCommitment(uc, hs) — intentionally not implemented here.
///
/// Consensus stays free of the full UTXO-commitment verification graph; call
/// `blvm_protocol::utxo_commitments::verification` (`verify_header_chain`, `verify_supply`,
/// `verify_commitment_block_hash`) from tests or tooling that already depend on protocol.
#[cfg(feature = "utxo-commitments")]
pub fn expected_verifyconsensuscommitment_from_orange_paper_impl<C>(
    _commitment: &C,
    _headers: &[crate::types::BlockHeader],
) -> i64 {
    unimplemented!(
        "Use blvm_protocol::utxo_commitments::verification::{{verify_header_chain, verify_supply, verify_commitment_block_hash}}"
    )
}
