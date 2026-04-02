//! BIP66 adds strict DER (SCRIPT_VERIFY_DERSIG) to consensus — not STRICTENC or LOW_S
//! (those are standardness / mempool policy; mainnet block 364000 includes legacy high-S sigs).

use blvm_consensus::activation::{ForkActivationTable, IsForkActive};
use blvm_consensus::block::calculate_base_script_flags_for_block_network;
use blvm_consensus::types::{ForkId, Network};

#[test]
fn bip66_base_flags_include_dersig_not_strictenc_or_low_s() {
    let table = ForkActivationTable::from_network(Network::Mainnet);
    let h = 364_000u64;
    assert!(table.is_fork_active(ForkId::Bip66, h));
    let flags = calculate_base_script_flags_for_block_network(h, Network::Mainnet);
    assert!(
        flags & 0x04 != 0,
        "expected SCRIPT_VERIFY_DERSIG after BIP66"
    );
    assert!(
        flags & 0x02 == 0,
        "STRICTENC is policy-only; not part of block consensus base flags"
    );
    assert!(
        flags & 0x08 == 0,
        "LOW_S is policy-only for legacy; high-S sigs are valid on-chain pre-segwit"
    );
}
