//! Tapscript must not enforce the legacy 201 non-push opcode limit (BIP342 / Core EvalScript).
//!
//! Regression for sort-merge step6 failures at blocks 812363–814371 (OpCount on inscription paths).

use blvm_consensus::script::{SigVersion, eval_script};
use blvm_consensus::types::ByteString;

#[test]
fn tapscript_exceeds_legacy_op_limit_without_opcount_error() {
    // 210 × OP_NOP (0x61) + OP_1 (0x51) — exceeds MAX_SCRIPT_OPS (201) for Base/WitnessV0.
    let mut script: ByteString = vec![0x61; 210];
    script.push(0x51);

    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Tapscript);

    // Must not fail with OpCount; script may still fail cleanstack (only OP_1 on stack is truthy).
    match result {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                !msg.contains("OpCount") && !msg.contains("Operation limit"),
                "Tapscript must not hit legacy op limit: {msg}"
            );
        }
        Ok(_) => {}
    }
}
