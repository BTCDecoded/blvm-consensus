//! Core `script_tests.json` integration (COV-C-02a subset).
//!
//! Runs legacy `P2SH,STRICTENC` cases with expected `OK` through `verify_script`.

#[path = "script_asm.rs"]
mod script_asm;

use blvm_consensus::block::calculate_tx_id;
use blvm_consensus::opcodes::OP_0;
use blvm_consensus::script::verify_script;
use blvm_consensus::types::{Network, Witness};
use blvm_consensus::{
    OutPoint, SEGWIT_ACTIVATION_MAINNET, Transaction, TransactionInput, TransactionOutput,
};
pub use script_asm::parse_core_script_asm;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ScriptTestVector {
    pub script_sig_asm: String,
    pub script_pubkey_asm: String,
    pub script_sig: Vec<u8>,
    pub script_pubkey: Vec<u8>,
    pub flags: u32,
    pub expected_ok: bool,
    pub description: String,
}

pub fn parse_flag_string(flags_str: &str) -> u32 {
    let mut flags = 0u32;
    for flag_name in flags_str.split(',') {
        let flag_name = flag_name.trim();
        match flag_name {
            "P2SH" => flags |= 0x01,
            "STRICTENC" => flags |= 0x02,
            "DERSIG" => flags |= 0x04,
            "LOW_S" => flags |= 0x08,
            "NULLDUMMY" => flags |= 0x10,
            "SIGPUSHONLY" => flags |= 0x20,
            "MINIMALDATA" => flags |= 0x40,
            "DISCOURAGE_UPGRADABLE_NOPS" => flags |= 0x80,
            "CLEANSTACK" => flags |= 0x100,
            "CHECKLOCKTIMEVERIFY" => flags |= 0x200,
            "CHECKSEQUENCEVERIFY" => flags |= 0x400,
            "WITNESS" => flags |= 0x800,
            "DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM" => flags |= 0x1000,
            "MINIMALIF" => flags |= 0x2000,
            "WITNESS_PUBKEYTYPE" => flags |= 0x8000,
            "TAPROOT" => flags |= 0x20000,
            "NONE" | "" => {}
            _ => {}
        }
    }
    flags
}

fn parse_flags(value: &Value) -> u32 {
    match value {
        Value::String(s) => parse_flag_string(s),
        Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
        _ => 0,
    }
}

fn is_header_comment(case: &[Value]) -> bool {
    matches!(case.first(), Some(Value::String(s)) if s.starts_with("Format is"))
}

fn is_witness_case(case: &[Value]) -> bool {
    matches!(case.first(), Some(Value::Array(_)))
}

pub fn load_script_test_vectors(
    path: &str,
) -> Result<Vec<ScriptTestVector>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&content)?;
    let cases = json
        .as_array()
        .ok_or("script_tests.json root must be array")?;

    let mut vectors = Vec::new();
    for case in cases {
        let case = case.as_array().ok_or("case must be array")?;
        if case.len() < 4 || is_header_comment(case) || is_witness_case(case) {
            continue;
        }

        let Some(script_sig_str) = case[0].as_str() else {
            continue;
        };
        let Some(script_pubkey_str) = case[1].as_str() else {
            continue;
        };
        let flags = parse_flags(&case[2]);
        let expected = case[3].as_str().unwrap_or("");
        let description = case
            .get(4)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let Some(script_sig) = parse_core_script_asm(script_sig_str) else {
            continue;
        };
        let Some(script_pubkey) = parse_core_script_asm(script_pubkey_str) else {
            continue;
        };

        vectors.push(ScriptTestVector {
            script_sig_asm: script_sig_str.to_string(),
            script_pubkey_asm: script_pubkey_str.to_string(),
            script_sig,
            script_pubkey,
            flags,
            expected_ok: expected == "OK",
            description,
        });
    }

    Ok(vectors)
}

/// Run vectors; returns Ok(()) only when all vectors match expectation.
pub fn run_core_script_tests(
    vectors: &[ScriptTestVector],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed = 0usize;
    let mut failed = 0usize;

    for (i, vector) in vectors.iter().enumerate() {
        let result = verify_script(
            &vector.script_sig,
            &vector.script_pubkey,
            None,
            vector.flags,
        );

        let ok = match result {
            Ok(v) => v,
            Err(_) => false,
        };

        if ok == vector.expected_ok {
            passed += 1;
        } else {
            failed += 1;
            eprintln!(
                "Script vector {} failed: expected {}, got {} — {}",
                i,
                if vector.expected_ok { "OK" } else { "fail" },
                if ok { "OK" } else { "fail" },
                vector.description
            );
        }
    }

    eprintln!("Core script vectors: {passed} passed, {failed} failed");
    if failed > 0 {
        Err(format!("{failed} script vectors failed").into())
    } else {
        Ok(())
    }
}

/// Run vectors and return `(passed, failed)` without failing early.
pub fn score_core_script_tests(vectors: &[ScriptTestVector]) -> (usize, usize) {
    // The cache switch is process-global. Two scorers in one binary otherwise
    // turn it back on in the middle of the other scorer's loop.
    static SCORE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = SCORE_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    blvm_consensus::script::disable_caching(true);
    let mut passed = 0usize;
    let mut failed = 0usize;

    for vector in vectors {
        let result = verify_script(
            &vector.script_sig,
            &vector.script_pubkey,
            None,
            vector.flags,
        );
        let ok = matches!(result, Ok(v) if v);
        if ok == vector.expected_ok {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    blvm_consensus::script::disable_caching(false);
    (passed, failed)
}

/// Witness-row test vector from Core `script_tests.json`.
#[derive(Debug, Clone)]
pub struct WitnessScriptTestVector {
    pub witness: Witness,
    pub prevout_value: i64,
    pub script_sig: Vec<u8>,
    pub script_pubkey: Vec<u8>,
    pub flags: u32,
    pub expected_ok: bool,
    pub description: String,
}

fn witness_item_bytes(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    if text.contains('#') || text.contains(' ') {
        return None;
    }
    hex::decode(text).ok()
}

/// Core `script_tests.json` stores a witness as `[item, ..., amount_btc]`.
/// `amount_btc` is the final number. Items are hex. A `#SCRIPT#` placeholder
/// cannot be executed, so the row is skipped.
fn parse_witness_stack(value: &Value) -> Option<(Witness, i64)> {
    let items = value.as_array()?;
    if items.is_empty() {
        return Some((Vec::new(), 0));
    }
    if items
        .iter()
        .any(|item| item.is_string() || item.is_number())
    {
        let amount_btc = items.last().and_then(|v| v.as_f64())?;
        let sats = (amount_btc * 100_000_000.0).round() as i64;
        let mut witness = Vec::new();
        for item in &items[..items.len() - 1] {
            witness.push(witness_item_bytes(item.as_str()?)?);
        }
        return Some((witness, sats));
    }
    let mut witness: Witness = Vec::new();
    let mut amount = 0i64;
    for item in items {
        let pair = item.as_array()?;
        if pair.is_empty() {
            continue;
        }
        witness.push(witness_item_bytes(pair[0].as_str()?)?);
        if pair.len() >= 2 {
            amount = pair[1].as_f64().unwrap_or(0.0) as i64;
        }
    }
    Some((witness, amount))
}

/// Credit and spend transactions from Core's `script_tests` harness.
/// The JSON signatures are for this pair: version 1, locktime 0, one input,
/// one empty output of the same value, and the credit txid as the prevout.
fn witness_sample_tx(
    script_sig: Vec<u8>,
    script_pubkey: Vec<u8>,
    amount: i64,
) -> (Transaction, Vec<TransactionOutput>) {
    let credit = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffff_ffff,
            },
            script_sig: vec![OP_0, OP_0].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: amount,
            script_pubkey: script_pubkey.clone().into(),
        }]
        .into(),
        lock_time: 0,
    };
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: calculate_tx_id(&credit),
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: amount,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: amount,
        script_pubkey: script_pubkey.into(),
    }];
    (tx, prevouts)
}

pub fn load_witness_script_test_vectors(
    path: &str,
) -> Result<Vec<WitnessScriptTestVector>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&content)?;
    let cases = json
        .as_array()
        .ok_or("script_tests.json root must be array")?;

    let mut vectors = Vec::new();
    for case in cases {
        let case = match case.as_array() {
            Some(c) => c,
            None => continue,
        };
        if case.len() < 5 || !is_witness_case(case) {
            continue;
        }

        let Some((witness, amount)) = parse_witness_stack(&case[0]) else {
            continue;
        };
        let script_sig_str = case[1].as_str().unwrap_or("");
        let script_pubkey_str = case[2].as_str().unwrap_or("");
        let flags = parse_flags(&case[3]);
        let expected = case[4].as_str().unwrap_or("");
        let description = case
            .get(5)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let Some(script_sig) = parse_core_script_asm(script_sig_str) else {
            continue;
        };
        let Some(script_pubkey) = parse_core_script_asm(script_pubkey_str) else {
            continue;
        };

        vectors.push(WitnessScriptTestVector {
            witness,
            prevout_value: amount,
            script_sig,
            script_pubkey,
            flags,
            expected_ok: expected == "OK",
            description,
        });
    }

    Ok(vectors)
}

pub fn load_default_witness_script_vectors()
-> Result<Vec<WitnessScriptTestVector>, Box<dyn std::error::Error>> {
    if !Path::new(SCRIPT_VECTORS_PATH).exists() {
        return Ok(Vec::new());
    }
    load_witness_script_test_vectors(SCRIPT_VECTORS_PATH)
}

/// Score witness vectors via `verify_script_with_context`.
pub fn score_witness_script_tests(vectors: &[WitnessScriptTestVector]) -> (usize, usize) {
    use blvm_consensus::script::verify_script_with_context;

    let mut passed = 0usize;
    let mut failed = 0usize;
    let block_height = SEGWIT_ACTIVATION_MAINNET + 1;

    for vector in vectors {
        let (mut tx, prevouts) = witness_sample_tx(
            vector.script_sig.clone(),
            vector.script_pubkey.clone(),
            vector.prevout_value,
        );
        tx.inputs[0].script_sig = vector.script_sig.clone().into();

        let result = verify_script_with_context(
            &tx.inputs[0].script_sig,
            &vector.script_pubkey,
            Some(&vector.witness),
            vector.flags,
            &tx,
            0,
            &prevouts,
            Some(block_height),
            Network::Mainnet,
        );
        let ok = matches!(result, Ok(v) if v);
        if ok == vector.expected_ok {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    (passed, failed)
}

const SCRIPT_VECTORS_PATH: &str = "tests/test_data/core_vectors/scripts/script_tests.json";

pub fn default_script_vectors_path() -> PathBuf {
    PathBuf::from(SCRIPT_VECTORS_PATH)
}

pub fn load_default_script_vectors() -> Result<Vec<ScriptTestVector>, Box<dyn std::error::Error>> {
    if !Path::new(SCRIPT_VECTORS_PATH).exists() {
        return Ok(Vec::new());
    }
    load_script_test_vectors(SCRIPT_VECTORS_PATH)
}
