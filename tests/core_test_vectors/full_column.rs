//! Every committed script and transaction row, scored by our verifier.
//!
//! A row is executed, skipped for a named reason, or a mismatch. The pins
//! below are the counts from the files in `tests/test_data/core_vectors/`.
//! A new skip, or a row that changes accept/reject, fails the test.

use blvm_consensus::error::ConsensusError;
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY, SCRIPT_VERIFY_CHECKSEQUENCEVERIFY, SCRIPT_VERIFY_CLEANSTACK,
    SCRIPT_VERIFY_CONST_SCRIPTCODE, SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_DISCOURAGE_OP_SUCCESS,
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS, SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEYTYPE,
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION,
    SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM, SCRIPT_VERIFY_LOW_S,
    SCRIPT_VERIFY_MINIMALDATA, SCRIPT_VERIFY_MINIMALIF, SCRIPT_VERIFY_NULLDUMMY,
    SCRIPT_VERIFY_NULLFAIL, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_SIGPUSHONLY, SCRIPT_VERIFY_STRICTENC,
    SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS, SCRIPT_VERIFY_WITNESS_PUBKEYTYPE,
};
use blvm_consensus::script::verify_script_with_context;
use blvm_consensus::serialization::deserialize_transaction_with_witness;
use blvm_consensus::taproot::{TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root};
use blvm_consensus::types::{Network, TransactionOutput, Witness};
use blvm_consensus::{SEGWIT_ACTIVATION_MAINNET, TAPROOT_ACTIVATION_MAINNET};
use serde_json::Value;

use super::script_tests::{
    SCORE_LOCK, parse_core_script_asm, parse_flag_string_strict, witness_sample_tx,
};

const SCRIPT_TESTS_PATH: &str = "tests/test_data/core_vectors/scripts/script_tests.json";
const TX_VALID_PATH: &str = "tests/test_data/core_vectors/transactions/tx_valid.json";
const TX_INVALID_PATH: &str = "tests/test_data/core_vectors/transactions/tx_invalid.json";

const SCRIPT_PLACEHOLDER: &str = "#SCRIPT#";
const CONTROL_BLOCK_PLACEHOLDER: &str = "#CONTROLBLOCK#";
const TAPROOT_OUTPUT_ASM: &str = "0x51 0x20 #TAPROOTOUTPUT#";
const PROSE_SKIP: &str = "prose row";
const BADTX_SKIP: &str = "BADTX: check_transaction, not script";

/// `script_tests.json`: 1323 rows, 60 prose headers, the rest executed.
const SCRIPT_EXECUTED: usize = 1263;
const SCRIPT_SKIPPED: usize = 60;
const SCRIPT_FAILURES: usize = 0;

/// `tx_valid.json` has 122 script rows. `tx_invalid.json` has 84 script rows
/// and 9 `BADTX` rows. Prose headers are the rest of each file.
const TX_VALID_EXECUTED: usize = 122;
const TX_VALID_SKIPPED: usize = 133;
const TX_INVALID_EXECUTED: usize = 84;
const TX_INVALID_SKIPPED: usize = 117;
const TX_FAILURES: usize = 0;

/// Bits P2SH through DISCOURAGE_UPGRADABLE_PUBKEYTYPE. `tx_valid.json` names
/// the flags that stay off; every other bit in this mask is on. Bit 31 is a
/// template-hash flag and is not part of this set.
const ALL_TX_SCRIPT_FLAGS: u32 = SCRIPT_VERIFY_P2SH
    | SCRIPT_VERIFY_STRICTENC
    | SCRIPT_VERIFY_DERSIG
    | SCRIPT_VERIFY_LOW_S
    | SCRIPT_VERIFY_NULLDUMMY
    | SCRIPT_VERIFY_SIGPUSHONLY
    | SCRIPT_VERIFY_MINIMALDATA
    | SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS
    | SCRIPT_VERIFY_CLEANSTACK
    | SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY
    | SCRIPT_VERIFY_CHECKSEQUENCEVERIFY
    | SCRIPT_VERIFY_WITNESS
    | SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM
    | SCRIPT_VERIFY_MINIMALIF
    | SCRIPT_VERIFY_NULLFAIL
    | SCRIPT_VERIFY_WITNESS_PUBKEYTYPE
    | SCRIPT_VERIFY_CONST_SCRIPTCODE
    | SCRIPT_VERIFY_TAPROOT
    | SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION
    | SCRIPT_VERIFY_DISCOURAGE_OP_SUCCESS
    | SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEYTYPE;

#[derive(Default)]
struct Counts {
    executed: usize,
    skipped: usize,
    failed: usize,
    skip_reasons: Vec<String>,
}

impl Counts {
    fn skip(&mut self, reason: &str) {
        self.skipped += 1;
        if !self.skip_reasons.iter().any(|existing| existing == reason) {
            self.skip_reasons.push(reason.to_owned());
        }
    }
}

struct Tapleaf {
    control_block: Vec<u8>,
    output_key: [u8; 32],
}

fn tapleaf_for(script: &[u8]) -> Result<Tapleaf, String> {
    let mut secret = [0u8; 32];
    secret[31] = 1;
    let internal = blvm_secp256k1::schnorr::xonly_pubkey_from_secret(&secret)
        .ok_or_else(|| "the fixed secret 1 must be an x-only key".to_owned())?;
    let leaf = script.to_vec();
    let root = compute_script_merkle_root(&leaf, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT)
        .map_err(|err| format!("tapleaf root: {err}"))?;
    let (output_key, parity) =
        blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(&internal, &root)
            .map_err(|err| format!("taproot output key: {err}"))?;
    let mut control_block = Vec::with_capacity(33);
    control_block.push(TAPROOT_LEAF_VERSION_TAPSCRIPT | (parity & 1));
    control_block.extend_from_slice(&internal);
    Ok(Tapleaf {
        control_block,
        output_key,
    })
}

fn sats_from_btc(btc: f64) -> i64 {
    (btc * 100_000_000.0).round() as i64
}

fn vector_height(flags: u32) -> u64 {
    if flags & SCRIPT_VERIFY_TAPROOT != 0 {
        TAPROOT_ACTIVATION_MAINNET
    } else if flags & SCRIPT_VERIFY_WITNESS != 0 {
        SEGWIT_ACTIVATION_MAINNET
    } else {
        SEGWIT_ACTIVATION_MAINNET - 1
    }
}

fn verdict_matches(result: &Result<bool, ConsensusError>, expected: &str) -> bool {
    matches!(result, Ok(true)) == (expected == "OK")
}

fn describe(result: &Result<bool, ConsensusError>) -> String {
    match result {
        Ok(true) => "OK".to_owned(),
        Ok(false) => "false".to_owned(),
        Err(ConsensusError::ScriptErrorWithCode { code, .. }) => format!("{code:?}"),
        Err(other) => format!("{other}"),
    }
}

struct ScriptRow {
    script_sig: Vec<u8>,
    script_pubkey: Vec<u8>,
    witness: Option<Witness>,
    amount: i64,
    flags: u32,
    expected: String,
    comment: String,
    index: usize,
}

fn load_script_rows(counts: &mut Counts) -> Vec<ScriptRow> {
    let text = std::fs::read_to_string(SCRIPT_TESTS_PATH).expect("script_tests.json");
    let root: Value = serde_json::from_str(&text).expect("script_tests.json");
    let cases = root.as_array().expect("script_tests.json root array");
    let mut rows = Vec::new();

    for (index, case) in cases.iter().enumerate() {
        let Some(case) = case.as_array() else {
            counts.skip(PROSE_SKIP);
            continue;
        };
        if case.len() == 1 && case[0].is_string() {
            counts.skip(PROSE_SKIP);
            continue;
        }

        let (witness_raw, sig_at, pubkey_at, flags_at, expected_at, comment_at) =
            if case.first().is_some_and(Value::is_array) {
                (Some(&case[0]), 1, 2, 3, 4, 5)
            } else if case.len() >= 4 {
                (None, 0, 1, 2, 3, 4)
            } else {
                counts.skip("short script row");
                continue;
            };

        let mut leaf: Option<Tapleaf> = None;
        let (witness, amount) = if let Some(raw) = witness_raw {
            let Some(items) = raw.as_array() else {
                counts.skip("witness is not an array");
                continue;
            };
            let Some(btc) = items.last().and_then(Value::as_f64) else {
                counts.skip("witness amount is not a number");
                continue;
            };
            let mut stack = Vec::new();
            let mut failed = false;
            for item in &items[..items.len() - 1] {
                let Some(text) = item.as_str() else {
                    counts.skip("witness element is not a string");
                    failed = true;
                    break;
                };
                if let Some(asm) = text.strip_prefix(SCRIPT_PLACEHOLDER) {
                    let Some(script) = parse_core_script_asm(asm) else {
                        counts.skip("#SCRIPT# did not parse");
                        failed = true;
                        break;
                    };
                    match tapleaf_for(&script) {
                        Ok(built) => {
                            leaf = Some(built);
                            stack.push(script);
                        }
                        Err(err) => {
                            counts.skip(&format!("tapleaf: {err}"));
                            failed = true;
                            break;
                        }
                    }
                    continue;
                }
                if text == CONTROL_BLOCK_PLACEHOLDER {
                    let Some(built) = &leaf else {
                        counts.skip("#CONTROLBLOCK# with no #SCRIPT#");
                        failed = true;
                        break;
                    };
                    stack.push(built.control_block.clone());
                    continue;
                }
                match hex::decode(text) {
                    Ok(bytes) => stack.push(bytes),
                    Err(_) => {
                        counts.skip("witness hex did not parse");
                        failed = true;
                        break;
                    }
                }
            }
            if failed {
                continue;
            }
            (Some(stack), sats_from_btc(btc))
        } else {
            (None, 0)
        };

        let Some(script_sig_str) = case.get(sig_at).and_then(Value::as_str) else {
            counts.skip("scriptSig is not a string");
            continue;
        };
        let Some(script_pubkey_str) = case.get(pubkey_at).and_then(Value::as_str) else {
            counts.skip("scriptPubKey is not a string");
            continue;
        };
        let flags_str = case.get(flags_at).and_then(Value::as_str).unwrap_or("");
        let Some(expected) = case.get(expected_at).and_then(Value::as_str) else {
            counts.skip("expected result is not a string");
            continue;
        };
        let script_sig = match parse_core_script_asm(script_sig_str) {
            Some(script) => script,
            None => {
                counts.skip("scriptSig asm did not parse");
                continue;
            }
        };
        let script_pubkey = if script_pubkey_str == TAPROOT_OUTPUT_ASM {
            let Some(built) = &leaf else {
                counts.skip("#TAPROOTOUTPUT# with no leaf");
                continue;
            };
            let mut script = vec![0x51, 0x20];
            script.extend_from_slice(&built.output_key);
            script
        } else {
            match parse_core_script_asm(script_pubkey_str) {
                Some(script) => script,
                None => {
                    counts.skip("scriptPubKey asm did not parse");
                    continue;
                }
            }
        };
        let flags = match parse_flag_string_strict(flags_str) {
            Ok(flags) => flags,
            Err(err) => {
                counts.skip(&err);
                continue;
            }
        };
        let comment = case
            .get(comment_at)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        rows.push(ScriptRow {
            script_sig,
            script_pubkey,
            witness,
            amount,
            flags,
            expected: expected.to_owned(),
            comment,
            index,
        });
    }
    rows
}

fn score_script_rows(rows: &[ScriptRow], counts: &mut Counts) -> Vec<String> {
    let mut mismatches = Vec::new();
    for row in rows {
        counts.executed += 1;
        let (tx, prevouts) = witness_sample_tx(
            row.script_sig.clone(),
            row.script_pubkey.clone(),
            row.amount,
        );
        let result = verify_script_with_context(
            &tx.inputs[0].script_sig,
            &row.script_pubkey,
            row.witness.as_ref(),
            row.flags,
            &tx,
            0,
            &prevouts,
            Some(vector_height(row.flags)),
            Network::Mainnet,
        );
        if verdict_matches(&result, &row.expected) {
            continue;
        }
        counts.failed += 1;
        if mismatches.len() < 12 {
            mismatches.push(format!(
                "row {} expected {} got {} ({})",
                row.index,
                row.expected,
                describe(&result),
                row.comment
            ));
        }
    }
    mismatches
}

struct Prevout {
    txid: [u8; 32],
    vout: u32,
    script_pubkey: Vec<u8>,
    value: i64,
}

fn rpc_txid(text: &str) -> Option<[u8; 32]> {
    let mut bytes = hex::decode(text).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    bytes.reverse();
    bytes.try_into().ok()
}

fn json_vout(value: &Value) -> Option<u32> {
    if let Some(n) = value.as_u64() {
        return u32::try_from(n).ok();
    }
    let n = value.as_i64()?;
    u32::try_from(n)
        .ok()
        .or_else(|| i32::try_from(n).ok().map(|n| n as u32))
}

fn json_i64(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_u64()
            .and_then(|n| i64::try_from(n).ok())
            .or_else(|| value.as_f64().map(|n| n.round() as i64))
    })
}

fn load_prevouts(value: &Value) -> Option<Vec<Prevout>> {
    let items = value.as_array()?;
    let mut prevouts = Vec::new();
    for item in items {
        let item = item.as_array()?;
        if item.len() < 3 {
            return None;
        }
        let txid = rpc_txid(item[0].as_str()?)?;
        // A coinbase input is stored as index -1. That is 0xffffffff.
        let vout = json_vout(&item[1])?;
        let script_pubkey = parse_core_script_asm(item[2].as_str()?)?;
        let value = item.get(3).and_then(json_i64).unwrap_or(0);
        prevouts.push(Prevout {
            txid,
            vout,
            script_pubkey,
            value,
        });
    }
    Some(prevouts)
}

fn is_prose(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::Array(items) if items.len() == 1 => items[0].is_string(),
        _ => false,
    }
}

fn has_badtx(flags: &str) -> bool {
    flags.split(',').any(|name| name.trim() == "BADTX")
}

fn score_tx_file(path: &str, expect_valid: bool, counts: &mut Counts) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect(path);
    let root: Value = serde_json::from_str(&text).expect(path);
    let cases = root.as_array().expect("transaction vector root array");
    let mut mismatches = Vec::new();

    for (index, case) in cases.iter().enumerate() {
        if is_prose(case) {
            counts.skip(PROSE_SKIP);
            continue;
        }
        let Some(case) = case.as_array() else {
            counts.skip("transaction row is not an array");
            continue;
        };
        let flags_str = case.get(2).and_then(Value::as_str).unwrap_or("");
        if has_badtx(flags_str) {
            counts.skip(BADTX_SKIP);
            continue;
        }
        let Some(tx_hex) = case.get(1).and_then(Value::as_str) else {
            counts.skip("transaction hex is not a string");
            continue;
        };
        let Some(prevouts) = case.first().and_then(load_prevouts) else {
            counts.skip("prevouts did not parse");
            continue;
        };
        let named = match parse_flag_string_strict(flags_str) {
            Ok(flags) => flags,
            Err(err) => {
                counts.skip(&err);
                continue;
            }
        };
        // Valid rows name the flags that stay off. Invalid rows name the flags
        // that are on.
        let flags = if expect_valid {
            ALL_TX_SCRIPT_FLAGS & !named
        } else {
            named
        };
        let tx_bytes = match hex::decode(tx_hex) {
            Ok(bytes) => bytes,
            Err(_) => {
                counts.skip("transaction hex did not parse");
                continue;
            }
        };
        counts.executed += 1;
        let parsed = deserialize_transaction_with_witness(&tx_bytes);
        let (accepted, detail) = match parsed {
            Ok((tx, witnesses, _)) => {
                let empty = Witness::new();
                let mut ordered = Vec::with_capacity(tx.inputs.len());
                let mut missing = false;
                for input in &tx.inputs {
                    let Some(prevout) = prevouts.iter().find(|prev| {
                        prev.txid == input.prevout.hash && prev.vout == input.prevout.index
                    }) else {
                        missing = true;
                        break;
                    };
                    ordered.push(TransactionOutput {
                        value: prevout.value,
                        script_pubkey: prevout.script_pubkey.clone(),
                    });
                }
                if missing || tx.inputs.is_empty() {
                    (false, "missing prevout".to_owned())
                } else {
                    let mut all_ok = true;
                    let mut why = String::new();
                    for (input_index, input) in tx.inputs.iter().enumerate() {
                        let witness = if witnesses.is_empty() {
                            None
                        } else {
                            Some(witnesses.get(input_index).unwrap_or(&empty))
                        };
                        let result = verify_script_with_context(
                            &input.script_sig,
                            &ordered[input_index].script_pubkey,
                            witness,
                            flags,
                            &tx,
                            input_index,
                            &ordered,
                            Some(vector_height(flags)),
                            Network::Mainnet,
                        );
                        if !matches!(result, Ok(true)) {
                            all_ok = false;
                            why = format!("input {input_index}: {}", describe(&result));
                            break;
                        }
                    }
                    (all_ok, why)
                }
            }
            Err(err) => (false, format!("deserialize: {err}")),
        };
        if accepted == expect_valid {
            continue;
        }
        counts.failed += 1;
        if mismatches.len() < 12 {
            mismatches.push(format!(
                "{path} row {index}: expected valid={expect_valid} got {accepted} ({detail}) flags {flags_str}"
            ));
        }
    }
    mismatches
}

fn assert_pin(counts: &Counts, executed: usize, skipped: usize, failed: usize, reasons: &[&str]) {
    assert_eq!(
        counts.failed, failed,
        "mismatches changed; the first rows are reported by the caller"
    );
    assert_eq!(
        counts.skipped, skipped,
        "skip count changed ({:?}); a new skip shrinks the column",
        counts.skip_reasons
    );
    assert_eq!(
        counts.executed, executed,
        "executed count changed; update the pin with the skip reason"
    );
    for reason in &counts.skip_reasons {
        assert!(
            reasons.contains(&reason.as_str()),
            "unexpected skip reason `{reason}`; allowed {reasons:?}"
        );
    }
    assert_eq!(
        counts.skip_reasons.len(),
        reasons.len(),
        "skip reasons {:?} do not match the allow-list {reasons:?}",
        counts.skip_reasons
    );
}

#[test]
fn committed_script_tests_match_our_verifier() {
    let _guard = SCORE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    blvm_consensus::script::disable_caching(true);
    let mut counts = Counts::default();
    let rows = load_script_rows(&mut counts);
    let mismatches = score_script_rows(&rows, &mut counts);
    blvm_consensus::script::disable_caching(false);
    assert!(
        mismatches.is_empty(),
        "script_tests mismatches:\n{}",
        mismatches.join("\n")
    );
    assert_pin(
        &counts,
        SCRIPT_EXECUTED,
        SCRIPT_SKIPPED,
        SCRIPT_FAILURES,
        &[PROSE_SKIP],
    );
}

#[test]
fn committed_transaction_scripts_match_our_verifier() {
    let _guard = SCORE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    blvm_consensus::script::disable_caching(true);
    let mut valid = Counts::default();
    let mut invalid = Counts::default();
    let mut mismatches = score_tx_file(TX_VALID_PATH, true, &mut valid);
    mismatches.extend(score_tx_file(TX_INVALID_PATH, false, &mut invalid));
    blvm_consensus::script::disable_caching(false);
    assert!(
        mismatches.is_empty(),
        "transaction script mismatches:\n{}",
        mismatches.join("\n")
    );
    assert_pin(
        &valid,
        TX_VALID_EXECUTED,
        TX_VALID_SKIPPED,
        TX_FAILURES,
        &[PROSE_SKIP],
    );
    assert_pin(
        &invalid,
        TX_INVALID_EXECUTED,
        TX_INVALID_SKIPPED,
        TX_FAILURES,
        &[PROSE_SKIP, BADTX_SKIP],
    );
}
