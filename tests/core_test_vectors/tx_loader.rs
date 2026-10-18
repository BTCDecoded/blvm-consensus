//! Shared Core transaction vector loader (tx_valid.json / tx_invalid.json).

use blvm_consensus::Transaction;
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction::{check_transaction, check_tx_inputs};
use blvm_consensus::{UtxoSet, ValidationResult};
use hex;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

/// Transaction test vector structure
#[derive(Debug, Clone)]
pub struct TransactionTestVector {
    pub transaction: Transaction,
    pub expected_result: bool,
    /// `BADTX` in the flag string: `check_transaction` must reject the structure.
    /// Any other flag is a script check. The transaction structure must pass.
    pub structural_failure: bool,
    pub flags: u32,
    pub description: String,
}

fn is_comment_row(item: &Value) -> bool {
    match item {
        Value::String(_) => true,
        Value::Array(arr) if arr.len() == 1 => matches!(arr.first(), Some(Value::String(_))),
        _ => false,
    }
}

fn flag_string(test_case: &[Value]) -> &str {
    test_case.get(2).and_then(|v| v.as_str()).unwrap_or("")
}

fn parse_flags_from_case(test_case: &[Value]) -> u32 {
    if test_case.len() >= 3 {
        if let Some(Value::String(flags_str)) = test_case.get(2) {
            return parse_flag_string(flags_str);
        }
        if let Some(Value::Number(n)) = test_case.get(2) {
            return n.as_u64().unwrap_or(0) as u32;
        }
    }
    0
}

fn parse_flag_string(flags_str: &str) -> u32 {
    let mut flags = 0u32;
    for flag_name in flags_str.split(',') {
        match flag_name.trim() {
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
            "NONE" | "" => {}
            _ => {}
        }
    }
    flags
}

fn try_parse_vector_case(
    test_case: &[Value],
    expected_valid: bool,
) -> Option<TransactionTestVector> {
    if test_case.len() < 2 {
        return None;
    }
    let tx_hex = test_case.get(1)?.as_str()?;
    if tx_hex.len() < 50 {
        return None;
    }
    let tx_bytes = hex::decode(tx_hex).ok()?;
    let transaction = deserialize_transaction(&tx_bytes).ok()?;
    let flags = parse_flags_from_case(test_case);
    let structural_failure = flag_string(test_case)
        .split(',')
        .any(|name| name.trim() == "BADTX");
    let description = format!(
        "tx {} (flags={}, badtx={structural_failure})",
        &tx_hex[..16.min(tx_hex.len())],
        flags
    );
    Some(TransactionTestVector {
        transaction,
        expected_result: expected_valid,
        structural_failure,
        flags,
        description,
    })
}

fn load_vectors_from_file(
    path: &PathBuf,
    expected_valid: bool,
) -> Result<Vec<TransactionTestVector>, Box<dyn std::error::Error>> {
    let mut vectors = Vec::new();
    if !path.exists() {
        return Ok(vectors);
    }
    let content = fs::read_to_string(path)?;
    let json: Value = serde_json::from_str(&content)?;
    let cases = json.as_array().ok_or("expected top-level JSON array")?;
    for item in cases {
        if is_comment_row(item) {
            continue;
        }
        if let Value::Array(test_case) = item {
            if let Some(vector) = try_parse_vector_case(test_case, expected_valid) {
                vectors.push(vector);
            }
        }
    }
    Ok(vectors)
}

/// `check_transaction`, plus the null-prevout rule that lives in `check_tx_inputs`.
/// A missing UTXO is not a structural failure.
pub fn structural_validation(tx: &Transaction) -> blvm_consensus::error::Result<ValidationResult> {
    match check_transaction(tx)? {
        ValidationResult::Invalid(reason) => Ok(ValidationResult::Invalid(reason)),
        ValidationResult::Valid => {
            let empty = UtxoSet::default();
            match check_tx_inputs(tx, &empty, 0)? {
                (ValidationResult::Invalid(reason), _) if reason.contains("null prevout") => {
                    Ok(ValidationResult::Invalid(reason))
                }
                _ => Ok(ValidationResult::Valid),
            }
        }
    }
}

/// Load transaction test vectors from `dir/tx_valid.json` and `dir/tx_invalid.json`.
pub fn load_transaction_test_vectors(
    dir: &str,
) -> Result<Vec<TransactionTestVector>, Box<dyn std::error::Error>> {
    let path = PathBuf::from(dir);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut vectors = load_vectors_from_file(&path.join("tx_valid.json"), true)?;
    vectors.extend(load_vectors_from_file(
        &path.join("tx_invalid.json"),
        false,
    )?);
    Ok(vectors)
}
