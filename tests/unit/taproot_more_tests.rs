use blvm_consensus::opcodes::PUSH_32_BYTES;
use blvm_consensus::taproot::*;

#[test]
fn test_validate_taproot_script_and_extract_key() {
    // BIP341 P2TR: OP_1 PUSH_32 <32-byte x-only key> = 34 bytes
    let key = [3u8; 32];
    let mut script = vec![TAPROOT_SCRIPT_PREFIX, PUSH_32_BYTES];
    script.extend_from_slice(&key);
    let valid = validate_taproot_script(&script).unwrap();
    assert!(valid);
    let extracted = extract_taproot_output_key(&script).unwrap();
    assert_eq!(extracted, Some(key));
}
