//! Signature operation counting functions
//!
//! Implements Bitcoin Core's sigop counting for block validation.
//! Sigops are counted to enforce MAX_BLOCK_SIGOPS_COST limit (80,000).
//!
//! Reference: Bitcoin Core `tx_verify.cpp` and `script.cpp`

use crate::error::Result;
use crate::opcodes::*;
use crate::segwit::Witness;
use crate::types::*;
use blvm_spec_lock::spec_locked;

/// Maximum number of public keys in a multisig (for sigop counting)
/// This is used when we can't accurately determine the number from the script
const MAX_PUBKEYS_PER_MULTISIG: u32 = 20;

/// Witness scale factor for sigop cost calculation
/// Legacy sigops count as 4x their actual number in sigop cost
const WITNESS_SCALE_FACTOR: u64 = 4;

/// Count sigops in a script (legacy counting)
///
/// Counts OP_CHECKSIG, OP_CHECKSIGVERIFY, OP_CHECKMULTISIG, OP_CHECKMULTISIGVERIFY.
/// Matches Bitcoin Core's CScript::GetSigOpCount(bool fAccurate).
///
/// Uses GetOp-style iteration: each call to the loop body reads one opcode and
/// advances past any associated push data, exactly like Bitcoin Core's GetOp().
///
/// # Arguments
/// * `script` - Script to count sigops in
/// * `accurate` - If true, use OP_1-OP_16 before OP_CHECKMULTISIG to determine key count
///
/// # Returns
/// Number of sigops in the script
#[spec_locked("5.2.2")]
pub fn count_sigops_in_script(script: &ByteString, accurate: bool) -> u32 {
    let mut count = 0u32;
    let mut last_opcode: Option<u8> = None;
    let mut i = 0;

    while i < script.len() {
        let opcode = script[i];

        // Skip past push data (matches Bitcoin Core's GetOp)
        if opcode > 0 && opcode < OP_PUSHDATA1 {
            // Direct push: opcode IS the length (1-75 bytes)
            let len = opcode as usize;
            last_opcode = Some(opcode);
            i += 1 + len;
            continue;
        } else if opcode == OP_PUSHDATA1 {
            // OP_PUSHDATA1: next byte is length
            if i + 1 >= script.len() { break; }
            let len = script[i + 1] as usize;
            last_opcode = Some(opcode);
            i += 2 + len;
            continue;
        } else if opcode == OP_PUSHDATA2 {
            // OP_PUSHDATA2: next 2 bytes (little-endian) are length
            if i + 2 >= script.len() { break; }
            let len = u16::from_le_bytes([script[i + 1], script[i + 2]]) as usize;
            last_opcode = Some(opcode);
            i += 3 + len;
            continue;
        } else if opcode == OP_PUSHDATA4 {
            // OP_PUSHDATA4: next 4 bytes (little-endian) are length
            if i + 4 >= script.len() { break; }
            let len = u32::from_le_bytes([
                script[i + 1],
                script[i + 2],
                script[i + 3],
                script[i + 4],
            ]) as usize;
            last_opcode = Some(opcode);
            i += 5 + len;
            continue;
        }

        // OP_CHECKSIG and OP_CHECKSIGVERIFY count as 1 sigop each
        if opcode == OP_CHECKSIG || opcode == OP_CHECKSIGVERIFY {
            count = count.saturating_add(1);
        }
        // OP_CHECKMULTISIG and OP_CHECKMULTISIGVERIFY count as multiple sigops
        else if opcode == OP_CHECKMULTISIG || opcode == OP_CHECKMULTISIGVERIFY {
            if accurate {
                // If accurate mode and previous opcode is OP_1-OP_16, use that number
                if let Some(prev_op) = last_opcode {
                    if (OP_1..=OP_16).contains(&prev_op) {
                        // OP_1 = 0x51, OP_16 = 0x60
                        // Decode: OP_N = N - 0x50
                        let n = (prev_op - OP_1 + 1) as u32;
                        count = count.saturating_add(n);
                    } else {
                        count = count.saturating_add(MAX_PUBKEYS_PER_MULTISIG);
                    }
                } else {
                    count = count.saturating_add(MAX_PUBKEYS_PER_MULTISIG);
                }
            } else {
                // Not accurate: assume maximum
                count = count.saturating_add(MAX_PUBKEYS_PER_MULTISIG);
            }
        }

        last_opcode = Some(opcode);
        i += 1;
    }

    count
}

/// Check if a script is P2SH (Pay-to-Script-Hash)
///
/// P2SH scripts have the format: OP_HASH160 (0xa9) <20-byte-hash> OP_EQUAL (0x87)
fn is_pay_to_script_hash(script: &ByteString) -> bool {
    script.len() == 23
        && script[0] == OP_HASH160  // OP_HASH160
        && script[1] == 0x14  // Push 20 bytes
        && script[22] == OP_EQUAL // OP_EQUAL
}

/// Extract redeem script from P2SH scriptSig
///
/// For P2SH, the scriptSig pushes the redeem script. We need to extract
/// the last push data item from scriptSig.
fn extract_redeem_script_from_scriptsig(script_sig: &ByteString) -> Option<ByteString> {
    let mut i = 0;
    let mut last_data: Option<ByteString> = None;

    while i < script_sig.len() {
        let opcode = script_sig[i];

        if opcode <= OP_PUSHDATA4 {
            // Push data opcode
            let (len, advance) = if opcode < OP_PUSHDATA1 {
                // Direct push: opcode is the length
                let len = opcode as usize;
                (len, 1)
            } else if opcode == OP_PUSHDATA1 {
                // OP_PUSHDATA1
                if i + 1 >= script_sig.len() {
                    return None;
                }
                let len = script_sig[i + 1] as usize;
                (len, 2)
            } else if opcode == OP_PUSHDATA2 {
                // OP_PUSHDATA2
                if i + 2 >= script_sig.len() {
                    return None;
                }
                let len = u16::from_le_bytes([script_sig[i + 1], script_sig[i + 2]]) as usize;
                (len, 3)
            } else if opcode == OP_PUSHDATA4 {
                // OP_PUSHDATA4
                if i + 4 >= script_sig.len() {
                    return None;
                }
                let len = u32::from_le_bytes([
                    script_sig[i + 1],
                    script_sig[i + 2],
                    script_sig[i + 3],
                    script_sig[i + 4],
                ]) as usize;
                (len, 5)
            } else {
                // Other push opcodes (OP_1NEGATE, OP_RESERVED, OP_1-OP_16)
                (0, 1)
            };

            if i + advance + len > script_sig.len() {
                return None;
            }

            last_data = Some(script_sig[i + advance..i + advance + len].to_vec());
            i += advance + len;
        } else if (OP_1..=OP_16).contains(&opcode) {
            // OP_1 to OP_16: push single byte
            last_data = Some(vec![opcode - OP_N_BASE]); // Convert OP_N to value N
            i += 1;
        } else {
            // Other opcode: not a push, invalid for P2SH
            return None;
        }
    }

    last_data
}

/// Get legacy sigop count from transaction
///
/// Counts sigops in scriptSig and scriptPubKey of all inputs and outputs.
/// Matches Bitcoin Core's GetLegacySigOpCount().
///
/// # Arguments
/// * `tx` - Transaction to count sigops in
///
/// # Returns
/// Total number of legacy sigops
#[spec_locked("5.2.2")]
pub fn get_legacy_sigop_count(tx: &Transaction) -> u32 {
    let mut count = 0u32;

    // Count sigops in all input scriptSigs
    for input in &tx.inputs {
        count = count.saturating_add(count_sigops_in_script(&input.script_sig, false));
    }

    // Count sigops in all output scriptPubKeys
    for output in &tx.outputs {
        count = count.saturating_add(count_sigops_in_script(&output.script_pubkey, false));
    }

    count
}

/// Get P2SH sigop count from transaction
///
/// Counts sigops in P2SH redeem scripts. Only counts sigops for outputs
/// that are P2SH (Pay-to-Script-Hash).
/// Matches Bitcoin Core's GetP2SHSigOpCount().
///
/// # Arguments
/// * `tx` - Transaction to count sigops in
/// * `utxo_set` - UTXO set to lookup input prevouts
///
/// # Returns
/// Total number of P2SH sigops
#[spec_locked("5.2.2")]
pub fn get_p2sh_sigop_count(tx: &Transaction, utxo_set: &UtxoSet) -> Result<u32> {
    // Coinbase transactions have no P2SH sigops
    use crate::transaction::is_coinbase;
    if is_coinbase(tx) {
        return Ok(0);
    }

    let mut count = 0u32;

    for input in &tx.inputs {
        // Get the UTXO (scriptPubKey) for this input
        if let Some(utxo) = utxo_set.get(&input.prevout) {
            // Check if this is a P2SH output
            if is_pay_to_script_hash(&utxo.script_pubkey) {
                // Extract redeem script from scriptSig
                if let Some(redeem_script) = extract_redeem_script_from_scriptsig(&input.script_sig)
                {
                    // Count sigops in redeem script (use accurate counting)
                    count = count.saturating_add(count_sigops_in_script(&redeem_script, true));
                }
            }
        }
    }

    Ok(count)
}

/// Count witness sigops in transaction
///
/// Counts sigops in witness scripts for SegWit transactions.
/// This is a simplified version - full implementation would need to handle
/// P2WPKH, P2WSH, and Taproot witness scripts.
///
/// # Arguments
/// * `tx` - Transaction
/// * `witnesses` - Witness data for each input (slice of Witness vectors)
/// * `utxo_set` - UTXO set to lookup scriptPubKeys
/// * `flags` - Script verification flags
///
/// # Returns
/// Number of witness sigops
fn count_witness_sigops(
    tx: &Transaction,
    witnesses: &[Witness],
    utxo_set: &UtxoSet,
    flags: u32,
) -> Result<u64> {
    use crate::transaction::is_coinbase;

    // SegWit flag must be enabled
    if (flags & 0x800) == 0 {
        return Ok(0);
    }

    if is_coinbase(tx) {
        return Ok(0);
    }

    let mut count = 0u64;

    for (i, input) in tx.inputs.iter().enumerate() {
        if let Some(utxo) = utxo_set.get(&input.prevout) {
            let script_pubkey = &utxo.script_pubkey;

            // P2WPKH: OP_0 <20-byte-hash>
            if script_pubkey.len() == 22 && script_pubkey[0] == OP_0 && script_pubkey[1] == 0x14 {
                // P2WPKH has 1 sigop (the CHECKSIG in the witness script)
                if let Some(witness) = witnesses.get(i) {
                    if !witness.is_empty() {
                        count = count.saturating_add(1);
                    }
                }
            }
            // P2WSH: OP_0 <32-byte-hash>
            else if script_pubkey.len() == 34
                && script_pubkey[0] == OP_0
                && script_pubkey[1] == 0x20
            {
                // P2WSH: count sigops in witness script
                if let Some(witness) = witnesses.get(i) {
                    if let Some(witness_script) = witness.last() {
                        count = count
                            .saturating_add(count_sigops_in_script(witness_script, true) as u64);
                    }
                }
            }
            // Taproot: handled separately (no sigops in Taproot)
        }
    }

    Ok(count)
}

/// Get total transaction sigop cost
///
/// Calculates total sigop cost for a transaction, including:
/// - Legacy sigops × 4 (witness scale factor)
/// - P2SH sigops × 4 (if P2SH enabled)
/// - Witness sigops (actual count, not scaled)
///
/// Matches Bitcoin Core's GetTransactionSigOpCost().
///
/// # Arguments
/// * `tx` - Transaction to count sigops in
/// * `utxo_set` - UTXO set to lookup inputs
/// * `witness` - Witness data for this transaction (one Witness per input)
/// * `flags` - Script verification flags
///
/// # Returns
/// Total sigop cost
#[spec_locked("5.2.2")]
pub fn get_transaction_sigop_cost(
    tx: &Transaction,
    utxo_set: &UtxoSet,
    witness: Option<&Witness>,
    flags: u32,
) -> Result<u64> {
    // Legacy sigops × witness scale factor
    let legacy_count = get_legacy_sigop_count(tx) as u64;
    let mut total_cost = legacy_count.saturating_mul(WITNESS_SCALE_FACTOR);

    use crate::transaction::is_coinbase;
    if is_coinbase(tx) {
        return Ok(total_cost);
    }

    // P2SH sigops × witness scale factor (if P2SH enabled)
    if (flags & 0x01) != 0 {
        // SCRIPT_VERIFY_P2SH flag enabled
        let p2sh_count = get_p2sh_sigop_count(tx, utxo_set)? as u64;
        total_cost = total_cost.saturating_add(p2sh_count.saturating_mul(WITNESS_SCALE_FACTOR));
    }

    // Witness sigops (actual count, not scaled)
    if let Some(witness) = witness {
        // Convert single Witness to slice for count_witness_sigops
        let witness_slice = std::slice::from_ref(witness);
        let witness_count = count_witness_sigops(tx, witness_slice, utxo_set, flags)?;
        total_cost = total_cost.saturating_add(witness_count);
    }

    Ok(total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_sigops_checksig() {
        // Script with OP_CHECKSIG
        let script = vec![OP_1, OP_1, OP_CHECKSIG]; // OP_1, OP_1, OP_CHECKSIG
        assert_eq!(count_sigops_in_script(&script, false), 1);
    }

    #[test]
    fn test_count_sigops_checksigverify() {
        // Script with OP_CHECKSIGVERIFY
        let script = vec![OP_1, OP_1, OP_CHECKSIGVERIFY]; // OP_1, OP_1, OP_CHECKSIGVERIFY
        assert_eq!(count_sigops_in_script(&script, false), 1);
    }

    #[test]
    fn test_count_sigops_multisig() {
        // Script with OP_CHECKMULTISIG (defaults to 20)
        let script = vec![OP_1, OP_2, OP_CHECKMULTISIG]; // OP_1, OP_2, OP_CHECKMULTISIG
        assert_eq!(count_sigops_in_script(&script, false), 20);

        // Accurate mode: use OP_2 value (2 sigops)
        assert_eq!(count_sigops_in_script(&script, true), 2);
    }

    #[test]
    fn test_get_legacy_sigop_count() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1, OP_CHECKSIG], // OP_1, OP_CHECKSIG
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![OP_1, OP_CHECKSIGVERIFY].into(), // OP_1, OP_CHECKSIGVERIFY
            }]
            .into(),
            lock_time: 0,
        };

        assert_eq!(get_legacy_sigop_count(&tx), 2);
    }

    #[test]
    fn test_is_pay_to_script_hash() {
        // Valid P2SH script: OP_HASH160 <20 bytes> OP_EQUAL
        let mut p2sh_script = vec![OP_HASH160, 0x14]; // OP_HASH160, push 20
        p2sh_script.extend_from_slice(&[0u8; 20]);
        p2sh_script.push(OP_EQUAL); // OP_EQUAL

        assert!(is_pay_to_script_hash(&p2sh_script));

        // Invalid: wrong length
        assert!(!is_pay_to_script_hash(&vec![OP_HASH160, 0x14]));

        // Invalid: not P2SH format
        let p2pkh = vec![OP_DUP, OP_HASH160, 0x14]; // OP_DUP OP_HASH160
        assert!(!is_pay_to_script_hash(&p2pkh));
    }

    // ==========================================================================
    // REGRESSION TESTS: Push data must not be counted as sigops (block 310357 fix)
    // ==========================================================================
    // These tests prevent regression of the bug where bytes inside push data
    // (e.g., 0xAC = OP_CHECKSIG) were incorrectly counted as sigops.
    // This caused valid blocks to be rejected with "sigop cost exceeds maximum".

    #[test]
    fn test_pushdata1_containing_checksig_byte_not_counted() {
        // OP_PUSHDATA1 <len=3> <0xAC 0xAC 0xAC>
        // The 0xAC bytes are push DATA, not OP_CHECKSIG opcodes.
        // Sigop count must be 0.
        let script = vec![OP_PUSHDATA1, 0x03, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG];
        assert_eq!(
            count_sigops_in_script(&script, false), 0,
            "Push data containing 0xAC must NOT be counted as sigops"
        );
    }

    #[test]
    fn test_pushdata2_containing_checksig_byte_not_counted() {
        // OP_PUSHDATA2 <len=4 as u16 LE> <0xAC 0xAD 0xAE 0xAF>
        // These are push DATA bytes, not opcodes.
        let script = vec![OP_PUSHDATA2, 0x04, 0x00, OP_CHECKSIG, OP_CHECKSIGVERIFY, OP_CHECKMULTISIG, OP_CHECKMULTISIGVERIFY];
        assert_eq!(
            count_sigops_in_script(&script, false), 0,
            "Push data containing sigop-like bytes must NOT be counted"
        );
    }

    #[test]
    fn test_pushdata4_containing_checksig_byte_not_counted() {
        // OP_PUSHDATA4 <len=2 as u32 LE> <0xAC 0xAC>
        let script = vec![OP_PUSHDATA4, 0x02, 0x00, 0x00, 0x00, OP_CHECKSIG, OP_CHECKSIG];
        assert_eq!(
            count_sigops_in_script(&script, false), 0,
            "OP_PUSHDATA4 data containing 0xAC must NOT be counted"
        );
    }

    #[test]
    fn test_direct_push_containing_checksig_byte_not_counted() {
        // Direct push: opcode 0x05 means "push next 5 bytes"
        // Data contains OP_CHECKSIG byte which must NOT be counted.
        let script = vec![0x05, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG];
        assert_eq!(
            count_sigops_in_script(&script, false), 0,
            "Direct push data containing 0xAC must NOT be counted as sigops"
        );
    }

    #[test]
    fn test_push_data_then_real_checksig() {
        // Direct push of 3 bytes (containing 0xAC), then a REAL OP_CHECKSIG
        // Only the real OP_CHECKSIG (after push data) should count.
        let script = vec![0x03, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG]; // push 3, data, then OP_CHECKSIG
        assert_eq!(
            count_sigops_in_script(&script, false), 1,
            "Only real OP_CHECKSIG after push data should count"
        );
    }

    #[test]
    fn test_pushdata1_then_real_multisig() {
        // OP_PUSHDATA1 <len=2> <OP_CHECKSIG OP_CHECKSIG> then OP_2 OP_CHECKMULTISIG
        // The OP_CHECKSIG bytes in push data don't count. Only the real OP_CHECKMULTISIG counts.
        let script = vec![OP_PUSHDATA1, 0x02, OP_CHECKSIG, OP_CHECKSIG, OP_2, OP_CHECKMULTISIG];
        // Inaccurate mode: multisig = 20
        assert_eq!(
            count_sigops_in_script(&script, false), 20,
            "Only real OP_CHECKMULTISIG should count (inaccurate=20)"
        );
        // Accurate mode: OP_2 before OP_CHECKMULTISIG = 2
        assert_eq!(
            count_sigops_in_script(&script, true), 2,
            "Accurate mode: OP_2 before OP_CHECKMULTISIG = 2 sigops"
        );
    }

    #[test]
    fn test_empty_script_zero_sigops() {
        let script: Vec<u8> = vec![];
        assert_eq!(count_sigops_in_script(&script, false), 0);
        assert_eq!(count_sigops_in_script(&script, true), 0);
    }

    #[test]
    fn test_truncated_pushdata1_does_not_panic() {
        // OP_PUSHDATA1 at end of script (no length byte)
        let script = vec![OP_PUSHDATA1];
        assert_eq!(count_sigops_in_script(&script, false), 0);
    }

    #[test]
    fn test_truncated_pushdata2_does_not_panic() {
        // OP_PUSHDATA2 with only 1 of 2 length bytes
        let script = vec![OP_PUSHDATA2, 0x01];
        assert_eq!(count_sigops_in_script(&script, false), 0);
    }

    #[test]
    fn test_truncated_pushdata4_does_not_panic() {
        // OP_PUSHDATA4 with only 3 of 4 length bytes
        let script = vec![OP_PUSHDATA4, 0x01, 0x00, 0x00];
        assert_eq!(count_sigops_in_script(&script, false), 0);
    }

    #[test]
    fn test_large_push_data_with_many_checksig_bytes() {
        // Simulate a realistic script where push data contains many OP_CHECKSIG bytes
        // This is the kind of script that caused the block 310357 failure.
        // OP_PUSHDATA2 <len=100 as u16 LE> <100 bytes of OP_CHECKSIG>
        let mut script = vec![OP_PUSHDATA2, 100, 0x00]; // OP_PUSHDATA2, length=100
        script.extend_from_slice(&[OP_CHECKSIG; 100]); // 100 bytes of OP_CHECKSIG data
        assert_eq!(
            count_sigops_in_script(&script, false), 0,
            "100 bytes of OP_CHECKSIG in push data must count as 0 sigops"
        );
    }

    #[test]
    fn test_multiple_sigop_opcodes() {
        // OP_CHECKSIG OP_CHECKSIG OP_CHECKSIGVERIFY
        let script = vec![0xac, 0xac, 0xad];
        assert_eq!(count_sigops_in_script(&script, false), 3);
    }

    #[test]
    fn test_multisig_accurate_op_16() {
        // OP_16 (0x60) OP_CHECKMULTISIG
        let script = vec![0x60, 0xae];
        assert_eq!(count_sigops_in_script(&script, true), 16);
    }

    #[test]
    fn test_multisig_accurate_op_1() {
        // OP_1 (0x51) OP_CHECKMULTISIG
        let script = vec![0x51, 0xae];
        assert_eq!(count_sigops_in_script(&script, true), 1);
    }
}

