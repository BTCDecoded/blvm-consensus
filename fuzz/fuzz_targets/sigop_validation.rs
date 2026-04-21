#![no_main]

use libfuzzer_sys::fuzz_target;
use blvm_consensus::sigop::*;
use blvm_consensus::types::*;
use blvm_consensus::segwit::Witness;
use blvm_consensus::utxo_overlay::UtxoLookup;
use std::collections::HashMap;

// Simple UTXO lookup
struct FuzzUtxoLookup {
    map: HashMap<OutPoint, UTXO>,
}

impl UtxoLookup for FuzzUtxoLookup {
    fn get(&self, point: &OutPoint) -> Option<&UTXO> {
        self.map.get(point)
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

fuzz_target!(|data: &[u8]| {
    // -----------------------------
    // 1. Split input (simple slicing)
    // -----------------------------
    if data.len() < 10 {
        return;
    }

    let split1 = data.len() / 3;
    let split2 = 2 * data.len() / 3;

    let script_sig = data[..split1].to_vec();
    let script_pubkey = data[split1..split2].to_vec();
    let witness_raw = data[split2..].to_vec();

    // -----------------------------
    // 2. Build minimal transaction
    // -----------------------------
    let prevout = OutPoint {
        hash: [0u8; 32].into(),
        index: 0,
    };

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: script_sig.clone().into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: script_pubkey.clone().into(),
        }]
        .into(),
        lock_time: 0,
    };

    // -----------------------------
    // 3. Build UTXO set
    // -----------------------------
    let utxo = UTXO {
        value: 1000,
        script_pubkey: script_pubkey.clone().into(),
        height: 0,
        is_coinbase: false,
    };

    let mut map = HashMap::new();
    map.insert(prevout, utxo);

    let lookup = FuzzUtxoLookup { map };

    let utxo_refs: Vec<Option<&UTXO>> = vec![lookup.get(&prevout)];

    // -----------------------------
    // 4. Witness construction
    // -----------------------------
    let witness: Witness = if witness_raw.is_empty() {
        vec![]
    } else {
        vec![witness_raw]
    };

    let witnesses = vec![witness];

    // -----------------------------
    // 5. Flags (controlled chaos)
    // -----------------------------
    let flags = if data.len() >= 4 {
        u32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        0
    };

    // -----------------------------
    // 6. Execute targets
    // -----------------------------
    let cost_with_utxos = get_transaction_sigop_cost_with_utxos(
        &tx,
        &utxo_refs,
        Some(&witnesses),
        flags,
    );

    let cost_with_lookup = get_transaction_sigop_cost_with_witness_slices(
        &tx,
        &lookup,
        Some(&witnesses),
        flags,
    );

    // -----------------------------
    // 7. Invariants
    // -----------------------------
    if let (Ok(a), Ok(b)) = (cost_with_utxos, cost_with_lookup) {
        // Cross-path consistency
        assert_eq!(a, b);

        // Legacy baseline
        let legacy = get_legacy_sigop_count(&tx) as u64;
        assert!(a >= legacy * 4);

        // Accurate vs inaccurate
        let acc = get_legacy_sigop_count_accurate(&tx);
        let inacc = get_legacy_sigop_count(&tx);
        assert!(acc <= inacc);
    }
});