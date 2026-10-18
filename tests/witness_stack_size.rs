//! Witness element and Taproot witness-structure limits.

use blvm_consensus::witness::{
    validate_segwit_witness_structure, validate_taproot_witness_structure,
};

#[test]
fn segwit_witness_element_limit_is_520_bytes() {
    let at_limit = vec![vec![0x11; 520]];
    let over_limit = vec![vec![0x11; 521]];
    let many_small = vec![vec![0x11]; 101];
    assert!(validate_segwit_witness_structure(&at_limit).unwrap());
    assert!(!validate_segwit_witness_structure(&over_limit).unwrap());
    assert!(validate_segwit_witness_structure(&many_small).unwrap());
}

fn witness(items: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    items
}

#[test]
fn taproot_key_path_accepts_64_and_nonzero_65() {
    assert!(validate_taproot_witness_structure(&witness(vec![vec![0x11; 64]]), false).unwrap());
    let mut sixty_five = vec![0x11; 65];
    assert!(validate_taproot_witness_structure(&witness(vec![sixty_five.clone()]), false).unwrap());
    sixty_five[64] = 0x00;
    assert!(!validate_taproot_witness_structure(&witness(vec![sixty_five]), false).unwrap());
    assert!(!validate_taproot_witness_structure(&witness(vec![vec![0x11; 63]]), false).unwrap());
    assert!(!validate_taproot_witness_structure(&witness(vec![vec![0x11; 66]]), false).unwrap());
}

#[test]
fn taproot_script_path_control_block_is_33_plus_32n() {
    let script = vec![0x51];
    assert!(
        validate_taproot_witness_structure(&witness(vec![script.clone(), vec![0x11; 33]]), true)
            .unwrap()
    );
    assert!(
        validate_taproot_witness_structure(&witness(vec![script.clone(), vec![0x11; 65]]), true)
            .unwrap()
    );
    assert!(
        !validate_taproot_witness_structure(&witness(vec![script.clone(), vec![0x11; 32]]), true)
            .unwrap()
    );
    assert!(
        !validate_taproot_witness_structure(&witness(vec![script, vec![0x11; 34]]), true).unwrap()
    );
    assert!(!validate_taproot_witness_structure(&witness(vec![vec![0x11; 33]]), true).unwrap());
}
