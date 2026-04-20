#![no_main]
//! Script interpreter with P2SH and witness verification flags.
use blvm_consensus::script::{eval_script, SigVersion, StackElement};
use libfuzzer_sys::fuzz_target;

const SCRIPT_VERIFY_P2SH: u32 = 0x01;
const SCRIPT_VERIFY_WITNESS: u32 = 0x800;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let script = data;
    let mut stack: Vec<StackElement> = Vec::new();
    let flags = SCRIPT_VERIFY_P2SH
        | (if (data[0] & 2) != 0 {
            SCRIPT_VERIFY_WITNESS
        } else {
            0
        })
        | u32::from_le_bytes(
            data.get(1..5)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0; 4]),
        );
    let sigv = if (flags & SCRIPT_VERIFY_WITNESS) != 0 {
        SigVersion::WitnessV0
    } else {
        SigVersion::Base
    };
    let _ = eval_script(script, &mut stack, flags, sigv);
});
