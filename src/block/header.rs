//! Block header validation (Orange Paper Section 5.3, §5.3.1).
//!
//! Single place for structural and time header rules. Part of a larger validation pipeline.
//!
//! ## What this module checks (H01, H03–H06 of §5.3.1)
//!
//! - **H01** — version ≥ 1 (floor; version 0 is rejected unconditionally)
//! - **H03** — timestamp ≠ 0
//! - **H04** — timestamp ≤ network_time + MAX_FUTURE_BLOCK_TIME (requires [`TimeContext`])
//! - **H05** — timestamp > median_time_past / BIP113 MTP (requires [`TimeContext`])
//! - **H06** — bits ≠ 0
//! - merkle_root ≠ all-zeros (structural sanity only; full merkle verification is in ConnectBlock)
//!
//! ## What this module does NOT check
//!
//! - **H02** — height-dependent version minimums (version ≥ 2/3/4 after BIP34/66/65): see
//!   [`crate::bip_validation::check_bip90`], called by `connect_block_inner`.
//! - **H07** — proof of work (hash vs compact target): see [`crate::pow::check_proof_of_work`].
//!   The header's compact bits must also equal [`crate::pow::next_required_bits`] when connect
//!   is given ancestor headers. That comparison is not done here.
//! - **H08** — parent hash linkage: pure predicate [`validate_prev_block_hash`]; the node chain
//!   layer supplies the parent header/hash and rejects blocks that fail H08 before `connect_block`.
//!
//! Callers connecting a block must invoke H02, H07, and H08 (via the predicate) in addition to
//! [`validate_block_header`] to satisfy `ValidBlockHeader` in full.
//!
//! ## Refactor / audit notes (coordinate with `blvm-spec-lock` before changing shape)
//!
//! - **Early returns** encode consensus rejects (`Ok(false)`). Do not duplicate the same condition
//!   with `assert!` below — that only adds panic risk if someone reorders code.
//! - The tautological `assert!(result || !result)` (below) is **on purpose**: formal verification /
//!   spec-lock tooling hooks here. Do not delete without verifier sign-off.
//! - **Version `0`** is rejected by `version < 1` (H01). Version 1 is valid before BIP34 and
//!   invalid after it — that boundary is enforced by `check_bip90` (H02), not here.
//! - **Merkle root** field is checked for all-zeros only (structural guard). Cryptographic
//!   verification of the merkle root against block transactions happens in `connect_block_inner`.

use crate::error::Result;
use crate::types::{
    BlockHeader, Hash, Network, OutPoint, TimeContext, Transaction, TransactionInput,
    TransactionOutput,
};
use blvm_spec_lock::spec_locked;

/// Validate block header structural and time rules (H01, H03–H06 of §5.3.1).
///
/// Returns `Ok(true)` if all checks pass, `Ok(false)` if any check fails.
///
/// This is one component of `ValidBlockHeader`. Callers connecting a block must also invoke:
/// - [`crate::bip_validation::check_bip90`] — H02: height-dependent version minimums
/// - [`crate::pow::check_proof_of_work`] — H07: hash vs compact target
///
/// Parent hash linkage (H08): [`validate_prev_block_hash`]; orchestration is in the node layer.
///
/// # Arguments
///
/// * `header` - Block header to validate
/// * `time_context` - Optional time context for timestamp validation (BIP113).
///   If `None`, only H01/H03/H06 (version, non-zero timestamp, bits) are enforced.
///   If `Some`, also enforces H04 (timestamp ≤ network_time + MAX_FUTURE_BLOCK_TIME)
///   and H05 (timestamp > median_time_past).
#[allow(clippy::overly_complex_bool_expr, clippy::redundant_comparisons)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3.1", "ValidBlockHeader")]
#[inline]
pub(crate) fn validate_block_header(
    header: &BlockHeader,
    time_context: Option<&TimeContext>,
) -> Result<bool> {
    if header.version < 1 {
        return Ok(false);
    }
    if header.timestamp == 0 {
        return Ok(false);
    }
    if let Some(ctx) = time_context {
        let max_ts = ctx
            .network_time
            .saturating_add(crate::constants::MAX_FUTURE_BLOCK_TIME);
        if header.timestamp > max_ts {
            return Ok(false);
        }
        if header.timestamp <= ctx.median_time_past {
            return Ok(false);
        }
    }
    if header.bits == 0 {
        return Ok(false);
    }
    if header.merkle_root == [0u8; 32] {
        return Ok(false);
    }

    // Formal-verification anchor (spec-lock): keep `result` and the tautology; omit a second
    // `assert!(result)` — success is `Ok(true)` below.
    let result = true;
    #[allow(clippy::eq_op)]
    {
        assert!(result || !result, "Validation result must be boolean");
    }
    Ok(result)
}

/// Double-SHA256 hash of an 80-byte serialized block header (Bitcoin block id).
#[spec_locked("5.3.1", "BlockHeaderHash")]
#[inline]
pub fn block_header_hash(header: &BlockHeader) -> crate::types::Hash {
    use blvm_primitives::crypto::hash256;
    use blvm_primitives::serialization::serialize_block_header;
    hash256(&serialize_block_header(header))
}

/// Header id of the network genesis block.
///
/// Connecting that block does not add its coinbase to the UTXO set.
pub fn genesis_header_hash(network: Network) -> Hash {
    block_header_hash(&genesis_header(network))
}

pub(crate) fn genesis_coinbase(network: Network) -> Transaction {
    let (script_sig, script_pubkey): (Vec<u8>, Vec<u8>) = if network == Network::Testnet4 {
        let msg = b"03/May/2024 000000000000000000001ebd58c244970b3aa9d783bb001011fbe8ea8e98e00e";
        let mut script_sig = vec![
            0x04,
            0xff,
            0xff,
            0x00,
            0x1d,
            0x01,
            0x04,
            0x4c,
            msg.len() as u8,
        ];
        script_sig.extend_from_slice(msg);
        let mut script_pubkey = vec![0x21];
        script_pubkey.extend_from_slice(&[0u8; 33]);
        script_pubkey.push(0xac);
        (script_sig, script_pubkey)
    } else {
        (
            vec![
                0x04, 0xff, 0xff, 0x00, 0x1d, 0x01, 0x04, 0x45, 0x54, 0x68, 0x65, 0x20, 0x54, 0x69,
                0x6d, 0x65, 0x73, 0x20, 0x30, 0x33, 0x2f, 0x4a, 0x61, 0x6e, 0x2f, 0x32, 0x30, 0x30,
                0x39, 0x20, 0x43, 0x68, 0x61, 0x6e, 0x63, 0x65, 0x6c, 0x6c, 0x6f, 0x72, 0x20, 0x6f,
                0x6e, 0x20, 0x62, 0x72, 0x69, 0x6e, 0x6b, 0x20, 0x6f, 0x66, 0x20, 0x73, 0x65, 0x63,
                0x6f, 0x6e, 0x64, 0x20, 0x62, 0x61, 0x69, 0x6c, 0x6f, 0x75, 0x74, 0x20, 0x66, 0x6f,
                0x72, 0x20, 0x62, 0x61, 0x6e, 0x6b, 0x73,
            ],
            vec![
                0x41, 0x04, 0x67, 0x8a, 0xfd, 0xb0, 0xfe, 0x55, 0x48, 0x27, 0x19, 0x67, 0xf1, 0xa6,
                0x71, 0x30, 0xb7, 0x10, 0x5c, 0xd6, 0xa8, 0x28, 0xe0, 0x39, 0x09, 0xa6, 0x79, 0x62,
                0xe0, 0xea, 0x1f, 0x61, 0xde, 0xb6, 0x49, 0xf6, 0xbc, 0x3f, 0x4c, 0xef, 0x38, 0xc4,
                0xf3, 0x55, 0x04, 0xe5, 0x1e, 0xc1, 0x12, 0xde, 0x5c, 0x38, 0x4d, 0xf7, 0xba, 0x0b,
                0x8d, 0x57, 0x8a, 0x4c, 0x70, 0x2b, 0x6b, 0xf1, 0x1d, 0x5f, 0xac,
            ],
        )
    };
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig,
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000_000_000,
            script_pubkey,
        }]
        .into(),
        lock_time: 0,
    }
}

pub(crate) fn genesis_header(network: Network) -> BlockHeader {
    header_for_genesis_coinbase(network, &genesis_coinbase(network))
}

pub(crate) fn header_for_genesis_coinbase(network: Network, coinbase: &Transaction) -> BlockHeader {
    let merkle_root = crate::mining::calculate_merkle_root(std::slice::from_ref(coinbase))
        .expect("genesis coinbase has a merkle root");
    let (timestamp, bits, nonce) = match network {
        Network::Mainnet => (1_231_006_505, 0x1d00ffff, 2_083_236_893),
        Network::Testnet => (1_296_688_602, 0x1d00ffff, 414_098_458),
        Network::Regtest => (1_296_688_602, 0x207fffff, 2),
        Network::Signet => (1_598_918_400, 0x1e0377ae, 52_613_770),
        Network::Testnet4 => (1_714_777_860, 0x1d00ffff, 393_743_547),
    };
    BlockHeader {
        version: 1,
        prev_block_hash: [0u8; 32],
        merkle_root,
        timestamp,
        bits,
        nonce,
    }
}

/// H08 (§5.3.1): `child.prev_block_hash` MUST equal `block_header_hash(parent)`.
///
/// Pure predicate for spec-lock binding. The node layer calls this before `connect_block` when
/// the parent header is known; IBD/header sync enforces the same invariant while walking the chain.
#[spec_locked("5.3.1", "ValidatePrevBlockHash")]
#[blvm_spec_lock::ensures(result == true || result == false)]
#[inline]
pub fn validate_prev_block_hash(child: &BlockHeader, parent: &BlockHeader) -> bool {
    child.prev_block_hash == block_header_hash(parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BlockHeader;

    #[test]
    fn validate_prev_block_hash_accepts_matching_parent() {
        let parent = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: 1,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let parent_hash = block_header_hash(&parent);
        let child = BlockHeader {
            prev_block_hash: parent_hash,
            merkle_root: [2u8; 32],
            ..parent
        };
        assert!(validate_prev_block_hash(&child, &parent));
    }

    #[test]
    fn validate_prev_block_hash_rejects_mismatch() {
        let parent = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: 1,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let child = BlockHeader {
            prev_block_hash: [9u8; 32],
            merkle_root: [2u8; 32],
            ..parent
        };
        assert!(!validate_prev_block_hash(&child, &parent));
    }

    #[test]
    fn validate_block_header_rejects_timestamp_equal_to_median() {
        use crate::types::TimeContext;

        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: 1_000,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let ctx = TimeContext {
            network_time: 2_000,
            median_time_past: 1_000,
        };
        assert!(!validate_block_header(&header, Some(&ctx)).unwrap());

        let earlier = BlockHeader {
            timestamp: 999,
            ..header
        };
        assert!(!validate_block_header(&earlier, Some(&ctx)).unwrap());

        let later = BlockHeader {
            timestamp: 1_001,
            ..header
        };
        assert!(validate_block_header(&later, Some(&ctx)).unwrap());
    }

    #[test]
    fn timestamp_past_the_two_hour_window_is_rejected() {
        use crate::constants::MAX_FUTURE_BLOCK_TIME;
        use crate::types::TimeContext;

        let network_time = 1_600_000_000;
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: network_time + MAX_FUTURE_BLOCK_TIME + 1,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let ctx = TimeContext {
            network_time,
            median_time_past: 0,
        };
        assert!(!validate_block_header(&header, Some(&ctx)).unwrap());
    }

    #[test]
    fn genesis_header_hashes_match_the_chain() {
        let cases = [
            (
                Network::Mainnet,
                "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
            ),
            (
                Network::Testnet,
                "000000000933ea01ad0ee984209779baaec3ced90fa3f408719526f8d77f4943",
            ),
            (
                Network::Regtest,
                "0f9188f13cb7b2c71f2a335e3a4fc328bf5beb436012afca590b1a11466e2206",
            ),
            (
                Network::Signet,
                "00000008819873e925422c1ff0f99f7cc9bbb232af63a077a480a3633bee1ef6",
            ),
            (
                Network::Testnet4,
                "00000000da84f2bafbbc53dee25a72ae507ff4914b867c565be350b0da8bf043",
            ),
        ];
        for (network, display) in cases {
            let mut hash = genesis_header_hash(network);
            hash.reverse();
            assert_eq!(hex::encode(hash), display, "{network:?}");
        }
    }
}
