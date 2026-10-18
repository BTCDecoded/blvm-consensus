//! P2SH scriptPubKey checks the hash of the pushed redeem script.

use blvm_consensus::error::{ConsensusError, ScriptErrorCode};
use blvm_consensus::opcodes::{OP_EQUAL, OP_HASH160};
use blvm_consensus::script::verify_script;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

fn hash160(data: &[u8]) -> [u8; 20] {
    let sha = Sha256::digest(data);
    let mut ripe = Ripemd160::new();
    ripe.update(sha);
    ripe.finalize().into()
}

fn p2sh_script(redeem: &[u8]) -> Vec<u8> {
    let digest = hash160(redeem);
    let mut script = vec![OP_HASH160, 0x14];
    script.extend(digest);
    script.push(OP_EQUAL);
    script
}

fn push_element(element: &[u8]) -> Vec<u8> {
    assert!(element.len() <= 75, "direct push only");
    let mut script = vec![element.len() as u8];
    script.extend(element);
    script
}

fn script_code(result: Result<bool, ConsensusError>) -> Option<ScriptErrorCode> {
    match result {
        Err(ConsensusError::ScriptErrorWithCode { code, .. }) => Some(code),
        _ => None,
    }
}

#[test]
fn p2sh_hash_matches_the_pushed_redeem_script() {
    let redeem = vec![0x51];
    let script_sig = push_element(&redeem);
    assert_eq!(
        verify_script(&script_sig, &p2sh_script(&redeem), None, 0x01).unwrap(),
        true
    );
    let mut flipped = redeem.clone();
    flipped[0] ^= 0xff;
    assert_eq!(
        verify_script(&script_sig, &p2sh_script(&flipped), None, 0x01).unwrap(),
        false
    );
}

#[test]
fn redeem_push_above_520_bytes_is_push_size() {
    let element = vec![0x11; 521];
    let mut script_sig = vec![0x4e];
    script_sig.extend((element.len() as u32).to_le_bytes());
    script_sig.extend(&element);
    let result = verify_script(&script_sig, &p2sh_script(&element), None, 0x01);
    assert_eq!(script_code(result), Some(ScriptErrorCode::PushSize));

    let at_limit = vec![0x11; 520];
    let mut script_sig = vec![0x4d];
    script_sig.extend((at_limit.len() as u16).to_le_bytes());
    script_sig.extend(&at_limit);
    assert_eq!(
        verify_script(&script_sig, &p2sh_script(&at_limit), None, 0x01).unwrap(),
        true
    );
}
