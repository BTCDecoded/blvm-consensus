//! Script execution context for signature verification.
//!
//! Bundles transaction, prevouts, and sighash-related parameters to reduce
//! argument count in execute_opcode_with_context_full.

use crate::types::*;

use super::SigVersion;

/// Context for script execution with transaction/signature verification.
/// Passed to execute_opcode_with_context_full instead of 15+ separate arguments.
#[derive(Clone, Copy)]
pub struct ScriptContext<'a> {
    pub tx: &'a Transaction,
    pub input_index: usize,
    pub prevout_values: &'a [i64],
    pub prevout_script_pubkeys: &'a [&'a [u8]],
    pub block_height: Option<u64>,
    pub median_time_past: Option<u64>,
    pub network: crate::types::Network,
    pub sigversion: SigVersion,
    pub redeem_script_for_sighash: Option<&'a [u8]>,
    pub script_sig_for_sighash: Option<&'a ByteString>,
    pub tapscript_for_sighash: Option<&'a [u8]>,
    pub tapscript_codesep_pos: Option<u32>,
    pub taproot_annex_hash: Option<&'a Hash>,
    #[cfg(feature = "production")]
    pub schnorr_collector: Option<&'a crate::bip348::SchnorrSignatureCollector>,
    #[cfg(feature = "production")]
    pub precomputed_bip143: Option<&'a crate::transaction_hash::Bip143PrecomputedHashes>,
    #[cfg(feature = "production")]
    pub sighash_cache: Option<&'a crate::transaction_hash::SighashMidstateCache>,
}
