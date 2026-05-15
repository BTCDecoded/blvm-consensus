//! Script verification flags and script-exec cache helpers for block validation.
//!
//! Groups base/per-tx script flags, cache insertion/merge, and BIP143 precompute
//! so block connect logic stays in the parent module.

use crate::activation::{ForkActivationTable, IsForkActive};
use crate::constants::*;
use crate::opcodes::*;
use crate::segwit::{is_segwit_transaction, Witness};
use crate::transaction::is_coinbase;
use crate::types::*;
use blvm_spec_lock::spec_locked;
#[cfg(feature = "production")]
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::LazyLock;

use super::BlockValidationContext;

// ---------------------------------------------------------------------------
// Script flags (base, per-tx, and combined)
// ---------------------------------------------------------------------------

/// Base script flags for a block from activation context.
/// Call once per block, then use `calculate_script_flags_for_block` or `add_per_tx_script_flags`.
#[spec_locked("5.2.5", "CalculateScriptFlags")]
#[inline]
pub(crate) fn calculate_base_script_flags_for_block(
    height: u64,
    activation: &impl IsForkActive,
) -> u32 {
    let mut flags: u32 = 0;

    if activation.is_fork_active(ForkId::Bip16, height) {
        flags |= 0x01; // SCRIPT_VERIFY_P2SH
    }
    // BIP66: strict DER for ECDSA signatures only. Bitcoin Core `GetBlockScriptFlags`
    // adds SCRIPT_VERIFY_DERSIG here — not SCRIPT_VERIFY_STRICTENC or SCRIPT_VERIFY_LOW_S
    // (those are standardness / mempool policy; legacy blocks may contain high-S sigs).
    if activation.is_fork_active(ForkId::Bip66, height) {
        flags |= 0x04; // SCRIPT_VERIFY_DERSIG
    }
    if activation.is_fork_active(ForkId::Bip65, height) {
        flags |= 0x200; // CHECKLOCKTIMEVERIFY
    }
    // BIP112 (CSV): CHECKSEQUENCEVERIFY at CSV deployment height (mainnet 419328).
    // Bitcoin Core `GetBlockScriptFlags`: DEPLOYMENT_CSV, not SegWit.
    if activation.is_fork_active(ForkId::Bip112, height) {
        flags |= 0x400; // SCRIPT_VERIFY_CHECKSEQUENCEVERIFY
    }
    // BIP147 NULLDUMMY: Bitcoin Core enables at DEPLOYMENT_SEGWIT (same height as BIP147 on mainnet).
    if activation.is_fork_active(ForkId::Bip147, height) {
        flags |= 0x10; // SCRIPT_VERIFY_NULLDUMMY
    }
    #[cfg(feature = "ctv")]
    if activation.is_fork_active(ForkId::Ctv, height) {
        flags |= 0x80000000; // CHECK_TEMPLATE_VERIFY_HASH
    }

    flags
}

/// Convenience: base script flags from (height, network) when no context is available (e.g. mempool).
#[inline]
pub fn calculate_base_script_flags_for_block_network(
    height: u64,
    network: crate::types::Network,
) -> u32 {
    let table = ForkActivationTable::from_network(network);
    calculate_base_script_flags_for_block(height, &table)
}

/// Per-tx script flags (SegWit + Taproot). Add to base flags from `calculate_base_script_flags_for_block`.
#[spec_locked("5.2.5", "CalculateScriptFlags")]
#[inline]
fn add_per_tx_script_flags(
    base_flags: u32,
    tx: &Transaction,
    has_witness: bool,
    height: u64,
    activation: &impl IsForkActive,
) -> u32 {
    let mut flags = base_flags;
    if activation.is_fork_active(ForkId::SegWit, height)
        && (has_witness || is_segwit_transaction(tx))
    {
        flags |= 0x800;
    }
    if activation.is_fork_active(ForkId::Taproot, height) {
        for output in &tx.outputs {
            let script = &output.script_pubkey;
            if script.len() == TAPROOT_SCRIPT_LENGTH
                && script[0] == OP_1
                && script[1] == PUSH_32_BYTES
            {
                flags |= 0x8000;
                break;
            }
        }
    }
    flags
}

/// Calculate script verification flags for a transaction in a block (with activation context).
#[spec_locked("5.2.5", "CalculateScriptFlags")]
pub(crate) fn calculate_script_flags_for_block(
    tx: &Transaction,
    has_witness: bool,
    height: u64,
    activation: &impl IsForkActive,
) -> u32 {
    let base = calculate_base_script_flags_for_block(height, activation);
    add_per_tx_script_flags(base, tx, has_witness, height, activation)
}

/// Convenience: script flags from (height, network) when no context is available (e.g. mempool, bench tools).
#[spec_locked("5.2.5", "CalculateScriptFlags")]
pub fn calculate_script_flags_for_block_network(
    tx: &Transaction,
    has_witness: bool,
    height: u64,
    network: crate::types::Network,
) -> u32 {
    let table = ForkActivationTable::from_network(network);
    calculate_script_flags_for_block(tx, has_witness, height, &table)
}

/// Calculate script verification flags for a transaction in a block (with precomputed base flags).
#[spec_locked("5.2.5", "CalculateScriptFlags")]
#[inline]
pub(crate) fn calculate_script_flags_for_block_with_base(
    tx: &Transaction,
    has_witness: bool,
    base_flags: u32,
    height: u64,
    activation: &impl IsForkActive,
) -> u32 {
    add_per_tx_script_flags(base_flags, tx, has_witness, height, activation)
}

// ---------------------------------------------------------------------------
// §5.2.6 Script flag exceptions (Orange Paper; table from Bitcoin Core chainparams)
// ---------------------------------------------------------------------------

/// RPC / explorer 64-hex block id → canonical digest order used by `hash256(serialize(header))` here.
fn block_hash_from_rpc_hex(hex_str: &str) -> Hash {
    let mut bytes = hex::decode(hex_str).expect("valid 64-char block hash hex");
    assert_eq!(bytes.len(), 32);
    bytes.reverse();
    bytes.try_into().expect("length 32")
}

/// BIP16 exception block (mainnet). `SCRIPT_VERIFY_NONE` in Core.
static MAINNET_SCRIPT_FLAG_EXCEPTION_BIP16: LazyLock<Hash> = LazyLock::new(|| {
    block_hash_from_rpc_hex("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22")
});
/// Taproot exception block (mainnet). `SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS` in Core.
static MAINNET_SCRIPT_FLAG_EXCEPTION_TAPROOT: LazyLock<Hash> = LazyLock::new(|| {
    block_hash_from_rpc_hex("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad")
});
/// BIP16 exception block (testnet3). `SCRIPT_VERIFY_NONE` in Core.
static TESTNET_SCRIPT_FLAG_EXCEPTION_BIP16: LazyLock<Hash> = LazyLock::new(|| {
    block_hash_from_rpc_hex("00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105")
});

/// Consensus script-flag override for `block_hash` on `network`, if any (partial map entry).
///
/// Bitcoin Core: `Consensus::Params::script_flag_exceptions` (`src/kernel/chainparams.cpp`).
#[spec_locked("5.2.6", "ScriptFlagExceptions")]
pub fn script_flag_exceptions_lookup(block_hash: &Hash, network: Network) -> Option<u32> {
    match network {
        Network::Mainnet => {
            if block_hash == &*MAINNET_SCRIPT_FLAG_EXCEPTION_BIP16 {
                Some(0)
            } else if block_hash == &*MAINNET_SCRIPT_FLAG_EXCEPTION_TAPROOT {
                Some(0x01 | 0x800) // SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS
            } else {
                None
            }
        }
        Network::Testnet => {
            if block_hash == &*TESTNET_SCRIPT_FLAG_EXCEPTION_BIP16 {
                Some(0)
            } else {
                None
            }
        }
        Network::Regtest => None,
    }
}

/// Bitcoin Core `GetBlockScriptFlags` (`validation.cpp`): start from `SCRIPT_VERIFY_P2SH |
/// SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_TAPROOT`, replace with the script-flag exception entry
/// when present, then OR buried deployments (BIP66, BIP65, CSV, BIP147-at-SegWit). Same **block-level**
/// bitmask is passed to script checks for every non-coinbase transaction.
///
/// This is **not** the Orange Paper §5.2.6 piecewise `GetBlockScriptFlags` (exception vs per-tx
/// `CalculateScriptFlags`); use [`get_block_script_flags`] for that formulation. `connect_block`
/// uses this function for **mainnet consensus parity** with Bitcoin Core.
pub fn get_block_script_verify_flags_core(
    block_hash: &Hash,
    height: u64,
    activation: &impl IsForkActive,
    network: Network,
) -> u32 {
    let mut flags = 0x01u32 | 0x800 | 0x8000; // P2SH | WITNESS | TAPROOT
    if let Some(v) = script_flag_exceptions_lookup(block_hash, network) {
        flags = v;
    }
    if activation.is_fork_active(ForkId::Bip66, height) {
        flags |= 0x04; // SCRIPT_VERIFY_DERSIG
    }
    if activation.is_fork_active(ForkId::Bip65, height) {
        flags |= 0x200; // CHECKLOCKTIMEVERIFY
    }
    if activation.is_fork_active(ForkId::Bip112, height) {
        flags |= 0x400; // CHECKSEQUENCEVERIFY
    }
    if activation.is_fork_active(ForkId::Bip147, height) {
        flags |= 0x10; // NULLDUMMY (Core: `DEPLOYMENT_SEGWIT`)
    }
    #[cfg(feature = "ctv")]
    if activation.is_fork_active(ForkId::Ctv, height) {
        flags |= 0x80000000;
    }
    flags
}

/// Orange Paper §5.2.6: exception table wins when defined; otherwise per-tx `CalculateScriptFlags`.
///
/// **Note:** Bitcoin Core’s `GetBlockScriptFlags` (`validation.cpp`) starts from a default mask, applies
/// this exception table, then ORs buried deployments (BIP66, BIP65, CSV, BIP147). **`connect_block`**
/// uses [`get_block_script_verify_flags_core`] for that Core behavior; this function remains the
/// Orange Paper §5.2.6 piecewise spec.
#[spec_locked("5.2.6", "GetBlockScriptFlags")]
pub fn get_block_script_flags(
    block_hash: &Hash,
    tx: &Transaction,
    has_witness: bool,
    height: u64,
    network: Network,
) -> u32 {
    if let Some(flags) = script_flag_exceptions_lookup(block_hash, network) {
        flags
    } else {
        calculate_script_flags_for_block_network(tx, has_witness, height, network)
    }
}

// ---------------------------------------------------------------------------
// Script-exec cache and overlay merge
// ---------------------------------------------------------------------------

/// Insert script exec cache keys for all txs in block (call when block validation passes).
#[cfg(all(feature = "production", feature = "rayon"))]
pub(super) fn insert_script_exec_cache_for_block(
    block: &Block,
    witnesses: &[Vec<Witness>],
    height: u64,
    context: &BlockValidationContext,
) {
    let block_hash = crate::crypto::OptimizedSha256::new().hash256(
        &crate::serialization::block::serialize_block_header(&block.header),
    );
    let block_script_verify_flags =
        get_block_script_verify_flags_core(&block_hash, height, context, context.network);
    for (i, tx) in block.transactions.iter().enumerate() {
        if is_coinbase(tx) {
            continue;
        }
        let wits = witnesses.get(i).map(|w| w.as_slice()).unwrap_or(&[]);
        let witnesses_vec: Vec<_> = if wits.len() == tx.inputs.len() {
            wits.to_vec()
        } else {
            (0..tx.inputs.len()).map(|_| Vec::new()).collect()
        };
        let key =
            crate::script_exec_cache::compute_key(tx, &witnesses_vec, block_script_verify_flags);
        crate::script_exec_cache::insert(&key);
    }
}

/// Merge overlay changes into cache. Updates bip30_index and optionally builds undo log.
/// When `undo_log` is None (IBD mode), skips undo entry construction entirely.
#[cfg(feature = "production")]
pub(super) fn merge_overlay_changes_to_cache(
    additions: &FxHashMap<OutPoint, std::sync::Arc<UTXO>>,
    deletions: &FxHashSet<crate::utxo_overlay::UtxoDeletionKey>,
    utxo_set: &mut UtxoSet,
    mut bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
    mut undo_log: Option<&mut crate::reorganization::BlockUndoLog>,
) {
    use crate::reorganization::UndoEntry;

    for del_key in deletions {
        let outpoint = crate::utxo_overlay::utxo_deletion_key_to_outpoint(del_key);
        if let Some(arc) = utxo_set.remove(&outpoint) {
            if let Some(idx) = bip30_index.as_deref_mut() {
                if arc.is_coinbase {
                    if let std::collections::hash_map::Entry::Occupied(mut o) =
                        idx.entry(outpoint.hash)
                    {
                        *o.get_mut() = o.get().saturating_sub(1);
                        if *o.get() == 0 {
                            o.remove();
                        }
                    }
                }
            }
            if let Some(ref mut log) = undo_log {
                log.entries.push(UndoEntry {
                    outpoint,
                    previous_utxo: Some(arc),
                    new_utxo: None,
                });
            }
        }
    }
    for (outpoint, arc) in additions {
        if let Some(ref mut log) = undo_log {
            log.entries.push(UndoEntry {
                outpoint: *outpoint,
                previous_utxo: None,
                new_utxo: Some(std::sync::Arc::clone(arc)),
            });
        }
        utxo_set.insert(*outpoint, std::sync::Arc::clone(arc));
    }
}

/// Compute BIP143/precomputed sighash for CCheckQueue path. Uses local refs and specs Vecs
/// (dropped before return) so buf borrow ends.
#[cfg(all(feature = "production", feature = "rayon"))]
pub(super) fn compute_bip143_and_precomp(
    tx: &Transaction,
    prevout_values: &[i64],
    script_pubkey_indices: &[(usize, usize)],
    script_pubkey_buffer: &[u8],
    has_witness: bool,
) -> (
    Option<crate::transaction_hash::Bip143PrecomputedHashes>,
    Vec<Option<[u8; 32]>>,
) {
    let buf = script_pubkey_buffer;
    let refs: Vec<&[u8]> = script_pubkey_indices
        .iter()
        .map(|&(s, l)| buf[s..s + l].as_ref())
        .collect();
    let refs: &[&[u8]] = &refs;
    if has_witness {
        let bip =
            crate::transaction_hash::Bip143PrecomputedHashes::compute(tx, prevout_values, refs);
        let mut precomp = vec![None; script_pubkey_indices.len()];
        let mut specs: Vec<(usize, u8, &[u8])> = Vec::new();
        for (j, &(s, l)) in script_pubkey_indices.iter().enumerate() {
            let spk = &buf[s..s + l];
            if spk.len() == 22 && spk[0] == OP_0 && spk[1] == PUSH_20_BYTES {
                let mut script_code = [0u8; 25];
                script_code[0] = OP_DUP;
                script_code[1] = OP_HASH160;
                script_code[2] = PUSH_20_BYTES;
                script_code[3..23].copy_from_slice(&spk[2..22]);
                script_code[23] = OP_EQUALVERIFY;
                script_code[24] = OP_CHECKSIG;
                let amount = prevout_values.get(j).copied().unwrap_or(0);
                if let Ok(h) = crate::transaction_hash::calculate_bip143_sighash(
                    tx,
                    j,
                    &script_code,
                    amount,
                    0x01,
                    Some(&bip),
                ) {
                    precomp[j] = Some(h);
                }
            } else if spk.len() == 23
                && spk[0] == OP_HASH160
                && spk[1] == PUSH_20_BYTES
                && spk[22] == OP_EQUAL
            {
                if let Some((sighash_byte, redeem)) =
                    crate::script::parse_p2sh_p2pkh_for_precompute(&tx.inputs[j].script_sig)
                {
                    specs.push((j, sighash_byte, redeem));
                }
            }
        }
        if !specs.is_empty() {
            if let Ok(hashes) = crate::transaction_hash::batch_compute_legacy_sighashes(
                tx,
                prevout_values,
                refs,
                &specs,
            ) {
                for (k, &(j, _, _)) in specs.iter().enumerate() {
                    precomp[j] = Some(hashes[k]);
                }
            }
        }
        (Some(bip), precomp)
    } else {
        let mut precomp = vec![None; script_pubkey_indices.len()];
        let mut specs: Vec<(usize, u8, &[u8])> = Vec::new();
        for (j, &(s, l)) in script_pubkey_indices.iter().enumerate() {
            let spk = &buf[s..s + l];
            if spk.len() == 25
                && spk[0] == OP_DUP
                && spk[1] == OP_HASH160
                && spk[2] == PUSH_20_BYTES
                && spk[23] == OP_EQUALVERIFY
                && spk[24] == OP_CHECKSIG
            {
                let script_sig = &tx.inputs[j].script_sig;
                if let Some((sig, _pubkey)) = crate::script::parse_p2pkh_script_sig(script_sig) {
                    if !sig.is_empty() {
                        specs.push((j, sig[sig.len() - 1], spk));
                    }
                }
            } else if spk.len() == 23
                && spk[0] == OP_HASH160
                && spk[1] == PUSH_20_BYTES
                && spk[22] == OP_EQUAL
            {
                if let Some((sighash_byte, redeem)) =
                    crate::script::parse_p2sh_p2pkh_for_precompute(&tx.inputs[j].script_sig)
                {
                    specs.push((j, sighash_byte, redeem));
                }
            }
        }
        if !specs.is_empty() {
            if let Ok(hashes) = crate::transaction_hash::batch_compute_legacy_sighashes(
                tx,
                prevout_values,
                refs,
                &specs,
            ) {
                for (k, &(j, _, _)) in specs.iter().enumerate() {
                    precomp[j] = Some(hashes[k]);
                }
            }
        }
        (None, precomp)
    }
}

#[cfg(test)]
mod script_flag_exceptions_tests {
    use super::{
        calculate_script_flags_for_block_network, get_block_script_flags,
        get_block_script_verify_flags_core, script_flag_exceptions_lookup,
    };
    use crate::activation::ForkActivationTable;
    use crate::crypto::OptimizedSha256;
    use crate::serialization::block::{deserialize_block_header, serialize_block_header};
    use crate::types::{Network, Transaction};

    fn hash_from_rpc_hex(hex_str: &str) -> [u8; 32] {
        let mut bytes = hex::decode(hex_str).unwrap();
        assert_eq!(bytes.len(), 32);
        bytes.reverse();
        bytes.try_into().unwrap()
    }

    #[test]
    fn mainnet_bip16_exception_zero() {
        let h =
            hash_from_rpc_hex("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
        assert_eq!(script_flag_exceptions_lookup(&h, Network::Mainnet), Some(0));
    }

    #[test]
    fn mainnet_taproot_exception_p2sh_witness() {
        let h =
            hash_from_rpc_hex("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad");
        assert_eq!(
            script_flag_exceptions_lookup(&h, Network::Mainnet),
            Some(0x01 | 0x800)
        );
    }

    #[test]
    fn testnet_bip16_exception_zero() {
        let h =
            hash_from_rpc_hex("00000000dd30457c001f4095d208cc1296b0eed002427aa599874af7a432b105");
        assert_eq!(script_flag_exceptions_lookup(&h, Network::Testnet), Some(0));
        assert_eq!(script_flag_exceptions_lookup(&h, Network::Mainnet), None);
    }

    #[test]
    fn regtest_has_no_exceptions() {
        let h =
            hash_from_rpc_hex("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
        assert_eq!(script_flag_exceptions_lookup(&h, Network::Regtest), None);
    }

    #[test]
    fn get_block_script_flags_uses_exception() {
        let h =
            hash_from_rpc_hex("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
        let tx = Transaction {
            version: 1,
            inputs: Default::default(),
            outputs: Default::default(),
            lock_time: 0,
        };
        let flags = get_block_script_flags(&h, &tx, false, 1_000_000, Network::Mainnet);
        assert_eq!(flags, 0);
    }

    #[test]
    fn get_block_script_flags_falls_back_to_calculate() {
        let h = [7u8; 32];
        let tx = Transaction {
            version: 1,
            inputs: Default::default(),
            outputs: Default::default(),
            lock_time: 0,
        };
        assert_eq!(
            get_block_script_flags(&h, &tx, false, 100, Network::Mainnet),
            calculate_script_flags_for_block_network(&tx, false, 100, Network::Mainnet),
        );
    }

    #[test]
    fn genesis_rpc_hex_matches_hash256_of_header() {
        let genesis_block_hex = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c010100000001000000000000000000000000000000000000000000000000000000000000000000ffffffff4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73ffffffff0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000";
        let bytes = hex::decode(genesis_block_hex).unwrap();
        let header = deserialize_block_header(&bytes[..80]).unwrap();
        let digest = OptimizedSha256::new().hash256(&serialize_block_header(&header));
        let expected =
            hash_from_rpc_hex("000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f");
        assert_eq!(digest, expected);
    }

    #[test]
    fn core_block_flags_non_exception_includes_p2sh_witness_taproot_and_deployments() {
        let table = ForkActivationTable::from_network(Network::Mainnet);
        let h = [0xabu8; 32];
        let flags = get_block_script_verify_flags_core(&h, 800_000, &table, Network::Mainnet);
        assert_eq!(flags & 0x8801, 0x8801, "P2SH | WITNESS | TAPROOT baseline");
        assert_ne!(flags & 0x04, 0, "DERSIG");
        assert_ne!(flags & 0x200, 0, "CLTV");
        assert_ne!(flags & 0x400, 0, "CSV");
        assert_ne!(flags & 0x10, 0, "NULLDUMMY");
    }

    #[test]
    fn core_block_flags_bip16_exception_orrs_buried_deployments() {
        let table = ForkActivationTable::from_network(Network::Mainnet);
        let h =
            hash_from_rpc_hex("00000000000002dc756eebf4f49723ed8d30cc28a5f108eb94b1ba88ac4f9c22");
        assert_eq!(script_flag_exceptions_lookup(&h, Network::Mainnet), Some(0));
        let flags = get_block_script_verify_flags_core(&h, 800_000, &table, Network::Mainnet);
        let deployment_mask = 0x04 | 0x200 | 0x400 | 0x10;
        assert_eq!(flags & 0x8801, 0);
        assert_eq!(flags & deployment_mask, deployment_mask);
        assert_eq!(flags & 0x800, 0);
        assert_eq!(flags & 0x8000, 0);
        assert_eq!(flags & 0x01, 0);
    }

    #[test]
    fn core_block_flags_taproot_exception_orrs_deployments() {
        let table = ForkActivationTable::from_network(Network::Mainnet);
        let h =
            hash_from_rpc_hex("0000000000000000000f14c35b2d841e986ab5441de8c585d5ffe55ea1e395ad");
        let flags = get_block_script_verify_flags_core(&h, 800_000, &table, Network::Mainnet);
        assert_eq!(flags & 0x8801, 0x801);
        assert_eq!(flags & 0x8000, 0);
        assert_ne!(flags & 0x800, 0);
        assert_ne!(flags & 0x04, 0);
    }
}
