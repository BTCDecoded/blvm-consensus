//! COV-C-02f: Hash opcode coverage (SHA256, RIPEMD160, HASH160, HASH256).

use bitcoin_hashes::{Hash as BitcoinHash, hash160, ripemd160, sha256};
use blvm_consensus::opcodes::{OP_HASH160, OP_HASH256, OP_PUSHDATA1, OP_RIPEMD160, OP_SHA256};
use blvm_consensus::script::{SigVersion, eval_script};

fn push_bytes(script: &mut Vec<u8>, data: &[u8]) {
    let len = data.len();
    if len <= 75 {
        script.push(len as u8);
    } else {
        script.push(OP_PUSHDATA1);
        script.push(len as u8);
    }
    script.extend_from_slice(data);
}

#[test]
fn test_op_sha256() {
    let data = b"blvm";
    let expected = sha256::Hash::hash(data);
    let mut script = Vec::new();
    push_bytes(&mut script, data);
    script.push(OP_SHA256);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.last().unwrap().as_slice(), expected.as_ref());
}

#[test]
fn test_op_ripemd160() {
    let data = b"blvm";
    let expected = ripemd160::Hash::hash(data);
    let mut script = Vec::new();
    push_bytes(&mut script, data);
    script.push(OP_RIPEMD160);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.last().unwrap().as_slice(), expected.as_ref());
}

#[test]
fn test_op_hash160() {
    let data = b"blvm";
    let expected = hash160::Hash::hash(data);
    let mut script = Vec::new();
    push_bytes(&mut script, data);
    script.push(OP_HASH160);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.last().unwrap().as_slice(), expected.as_ref());
}

#[test]
fn test_op_hash256() {
    let data = b"blvm";
    let once = sha256::Hash::hash(data);
    let expected = sha256::Hash::hash(once.as_ref());
    let mut script = Vec::new();
    push_bytes(&mut script, data);
    script.push(OP_HASH256);
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.last().unwrap().as_slice(), expected.as_ref());
}

#[test]
fn test_op_sha256_empty_stack_fails() {
    let mut stack = Vec::new();
    assert!(!eval_script(&[OP_SHA256], &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_ripemd160_empty_stack_fails() {
    let mut stack = Vec::new();
    assert!(!eval_script(&[OP_RIPEMD160], &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_hash160_empty_stack_fails() {
    let mut stack = Vec::new();
    assert!(!eval_script(&[OP_HASH160], &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_hash256_empty_stack_fails() {
    let mut stack = Vec::new();
    assert!(!eval_script(&[OP_HASH256], &mut stack, 0, SigVersion::Base).unwrap());
}
