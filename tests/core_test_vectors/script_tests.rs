//! Core `script_tests.json` integration (COV-C-02a subset).
//!
//! Runs legacy `P2SH,STRICTENC` cases with expected `OK` through `verify_script`.

#[path = "script_asm.rs"]
mod script_asm;

use blvm_consensus::script::verify_script;
use blvm_consensus::types::{Network, Witness};
use blvm_consensus::{
    OutPoint, SEGWIT_ACTIVATION_MAINNET, Transaction, TransactionInput, TransactionOutput,
};
use script_asm::parse_core_script_asm;
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

/// Phase-1 smoke filter for COV-C-02a: a small Core baseline that must pass end-to-end.
pub fn is_cov_c02a_smoke_vector(sig_asm: &str, spk_asm: &str) -> bool {
    if sig_asm.contains("0x") || spk_asm.contains("0x") {
        return false;
    }
    let sig = sig_asm.trim();
    let spk = spk_asm.trim();
    spk == "DEPTH 0 EQUAL" || spk == "2 EQUALVERIFY 1 EQUAL" || (spk.is_empty() && sig == "1")
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

fn parse_witness_stack(value: &Value) -> Option<(Witness, i64)> {
    let items = value.as_array()?;
    let mut witness: Witness = Vec::new();
    let mut amount = 0i64;
    for item in items {
        let pair = item.as_array()?;
        if pair.is_empty() {
            continue;
        }
        let hex_str = pair[0].as_str()?;
        let bytes = if hex_str.is_empty() {
            vec![]
        } else {
            hex::decode(hex_str).ok()?
        };
        witness.push(bytes);
        if pair.len() >= 2 {
            amount = pair[1].as_f64().unwrap_or(0.0) as i64;
        }
    }
    Some((witness, amount))
}

fn witness_sample_tx(
    script_sig: Vec<u8>,
    script_pubkey: Vec<u8>,
) -> (Transaction, Vec<TransactionOutput>) {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0xcd; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 100_000,
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
            prevout_value: if amount > 0 { amount } else { 100_000 },
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
        let (mut tx, mut prevouts) =
            witness_sample_tx(vector.script_sig.clone(), vector.script_pubkey.clone());
        prevouts[0].value = vector.prevout_value;
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

#[cfg(test)]
mod tests {
    use super::*;
    use blvm_consensus::opcodes::{OP_1, OP_2};

    #[test]
    fn test_verify_depth_zero_equal_direct() {
        use blvm_consensus::script::verify_script;

        let flags = parse_flag_string("P2SH,STRICTENC");
        let spk = parse_core_script_asm("DEPTH 0 EQUAL").expect("parse spk");
        let sig = parse_core_script_asm("").expect("parse empty sig");
        let result = verify_script(&sig, &spk, None, flags);
        eprintln!("verify_script empty DEPTH 0 EQUAL: {result:?}, spk={spk:?}");
        assert_eq!(result.unwrap(), true, "Core baseline vector 0 should pass");

        let sig_one = parse_core_script_asm("1").expect("parse 1");
        let empty_spk: Vec<u8> = vec![];
        let result2 = verify_script(&sig_one, &empty_spk, None, flags);
        eprintln!("verify_script 1 empty spk: {result2:?}");
        assert_eq!(result2.unwrap(), true);
    }

    #[test]
    fn test_script_asm_roundtrip_samples() {
        let sig = parse_core_script_asm("1 2").unwrap();
        assert_eq!(sig, vec![OP_1, OP_2]);
        let spk = parse_core_script_asm("2 EQUALVERIFY 1 EQUAL").unwrap();
        assert!(spk.len() >= 3);
    }

    #[test]
    fn test_load_script_vectors_if_present() {
        let vectors = load_default_script_vectors().expect("load");
        if vectors.is_empty() {
            return;
        }
        assert!(vectors.iter().any(|v| v.expected_ok));
    }

    #[test]
    fn test_run_legacy_ok_smoke_if_present() {
        let vectors = load_default_script_vectors().expect("load");
        if vectors.is_empty() {
            return;
        }

        let smoke: Vec<_> = vectors
            .iter()
            .filter(|v| {
                v.expected_ok
                    && v.flags == parse_flag_string("P2SH,STRICTENC")
                    && is_cov_c02a_smoke_vector(&v.script_sig_asm, &v.script_pubkey_asm)
            })
            .cloned()
            .collect();

        assert!(
            !smoke.is_empty(),
            "expected parseable P2SH,STRICTENC smoke vectors when script_tests.json is present"
        );
        run_core_script_tests(&smoke).expect("Core script smoke vectors should pass");
    }

    #[test]
    fn test_run_legacy_ok_progress_if_present() {
        let vectors = load_default_script_vectors().expect("load");
        if vectors.is_empty() {
            return;
        }

        let subset: Vec<_> = vectors
            .iter()
            .filter(|v| v.expected_ok && v.flags == parse_flag_string("P2SH,STRICTENC"))
            .cloned()
            .collect();

        assert!(!subset.is_empty());
        let (passed, failed) = score_core_script_tests(&subset);
        eprintln!("Core legacy OK progress: {passed}/{} passing", subset.len());
        // All 439 parseable P2SH,STRICTENC OK vectors now pass.
        assert!(
            passed >= 439,
            "expected >=439 legacy OK vectors passing, got {passed} passed / {failed} failed"
        );
    }
}
