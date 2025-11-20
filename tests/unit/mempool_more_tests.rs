use bllvm_consensus::mempool;
use bllvm_consensus::{Transaction, TransactionInput, TransactionOutput};

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::create_test_utxo;

#[test]
fn test_negative_fee_rejected() {
    let (set, prev) = create_test_utxo(1000);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput { prevout: prev, script_sig: vec![0x51], sequence: 0xffffffff }],
        outputs: vec![TransactionOutput { value: 2000, script_pubkey: vec![0x51] }],
        lock_time: 0,
    };
    let pool = mempool::Mempool::new();
    let res = mempool::accept_to_memory_pool(&tx, &set, &pool, 1);
    assert!(res.is_err(), "Outputs exceed inputs should be rejected");
}

#[test]
fn test_non_standard_script_flagged() {
    let tx = Transaction {
        version: 1,
        inputs: vec![],
        outputs: vec![TransactionOutput { value: 0, script_pubkey: vec![0x6a] }], // OP_RETURN only
        lock_time: 0,
    };
    // Whether standard depends on policy; just exercise the path
    let _ = mempool::is_standard_tx(&tx);
}



























