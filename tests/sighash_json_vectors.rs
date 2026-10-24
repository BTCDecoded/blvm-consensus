//! Loads Core `sighash.json` and checks the legacy sighash.

use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction_hash::legacy_sighash_for_vector;

#[test]
fn sighash_json_matches_the_core_rows() {
    let path = "tests/test_data/core_vectors/transactions/sighash.json";
    let text = std::fs::read_to_string(path).expect("sighash.json");
    let rows: Vec<serde_json::Value> = serde_json::from_str(&text).expect("json");
    let data: Vec<&serde_json::Value> = rows
        .iter()
        .filter(|row| row.as_array().is_some_and(|a| a.len() == 5))
        .collect();
    assert!(data.len() >= 500, "sighash.json data rows {}", data.len());
    for row in data {
        let fields = row.as_array().unwrap();
        let raw_tx = hex::decode(fields[0].as_str().unwrap()).unwrap();
        let script = hex::decode(fields[1].as_str().unwrap()).unwrap();
        let input_index = fields[2].as_u64().unwrap() as usize;
        let hash_type = fields[3].as_i64().unwrap() as i32;
        let expected = fields[4].as_str().unwrap();
        let tx = deserialize_transaction(&raw_tx).expect("tx");
        let got = legacy_sighash_for_vector(&tx, input_index, &script, hash_type).unwrap();
        let mut reversed = got;
        reversed.reverse();
        assert_eq!(
            hex::encode(reversed),
            expected,
            "input {input_index} type {hash_type}"
        );
    }
}
