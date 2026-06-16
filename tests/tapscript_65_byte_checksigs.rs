//! Tapscript OP_CHECKSIG / OP_CHECKSIGVERIFY / OP_CHECKSIGADD accept 65-byte Schnorr (64 + sighash).

use blvm_consensus::bip348::try_parse_taproot_schnorr_witness_sig;

#[test]
fn tapscript_schnorr_witness_sig_accepts_65_bytes_with_hashtype() {
    let mut sig65 = [0u8; 65];
    sig65[64] = 0x01; // SIGHASH_ALL
    let (parsed, ty) = try_parse_taproot_schnorr_witness_sig(&sig65).unwrap();
    assert_eq!(parsed.len(), 64);
    assert_eq!(ty, 0x01);
    assert_eq!(parsed, sig65[..64]);
}

#[test]
fn tapscript_schnorr_witness_sig_rejects_65_bytes_with_default_hashtype() {
    let mut sig65 = [0u8; 65];
    sig65[64] = 0x00; // explicit SIGHASH_DEFAULT is invalid on wire
    assert!(try_parse_taproot_schnorr_witness_sig(&sig65).is_none());
}
