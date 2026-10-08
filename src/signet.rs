//! Signet block-solution validation (BIP325, Orange Paper §11.5).

use crate::block::calculate_tx_id;
use crate::error::Result;
use crate::mining::compute_merkle_root_and_mutated;
use crate::opcodes::{OP_0, OP_PUSHDATA1, OP_PUSHDATA2, OP_PUSHDATA4, OP_RETURN};
use crate::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_NULLDUMMY, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS,
};
use crate::script::{SigVersion, verify_script_with_context_full};
use crate::segwit::Witness;
use crate::types::{
    Block, ByteString, Hash, Network, OutPoint, Transaction, TransactionInput, TransactionOutput,
};
use blvm_primitives::serialization::varint::decode_varint;
use blvm_spec_lock::spec_locked;

/// BIP325 signet magic prefix embedded in witness commitment pushdata.
const SIGNET_HEADER: [u8; 4] = [0xec, 0xc7, 0xda, 0xa2];

const BLOCK_SCRIPT_VERIFY_FLAGS: u32 =
    SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_NULLDUMMY;

/// Default signet challenge script bytes (signet default challenge).
pub fn default_signet_challenge() -> &'static [u8] {
    const CHALLENGE: [u8; 71] = [
        0x51, 0x21, 0x03, 0xad, 0x5e, 0x0e, 0xda, 0xd1, 0x8c, 0xb1, 0xf0, 0xfc, 0x0d, 0x28, 0xa3,
        0xd4, 0xf1, 0xf3, 0xe4, 0x45, 0x64, 0x03, 0x37, 0x48, 0x9a, 0xbb, 0x10, 0x40, 0x4f, 0x2d,
        0x1e, 0x08, 0x6b, 0xe4, 0x30, 0x21, 0x03, 0x59, 0xef, 0x50, 0x21, 0x96, 0x4f, 0xe2, 0x2d,
        0x6f, 0x8e, 0x05, 0xb2, 0x46, 0x3c, 0x95, 0x40, 0xce, 0x96, 0x88, 0x3f, 0xe3, 0xb2, 0x78,
        0x76, 0x0f, 0x04, 0x8f, 0x51, 0x89, 0xf2, 0xe6, 0xc4, 0x52, 0xae,
    ];
    &CHALLENGE
}

/// SignetChallenge(n): script for network `n`, or None when signet rules do not apply.
pub fn signet_challenge(network: Network, override_script: Option<&[u8]>) -> Option<ByteString> {
    match network {
        Network::Signet => Some(
            override_script
                .map(|s| s.to_vec())
                .unwrap_or_else(|| default_signet_challenge().to_vec()),
        ),
        _ => None,
    }
}

/// CheckSignetBlockSolution: valid when challenge is empty, or witness commitment satisfies challenge.
#[spec_locked("11.5", "CheckSignetBlockSolution")]
pub fn check_signet_block_solution(block: &Block, challenge: &[u8], height: u64) -> Result<bool> {
    if height == 0 {
        return Ok(true);
    }
    if challenge.is_empty() {
        return Ok(true);
    }
    let Some(pair) = build_signet_solution_txs(block, challenge)? else {
        return Ok(false);
    };
    let prevout_value = pair.to_spend.outputs[0].value;
    let prevout_script = pair.to_spend.outputs[0].script_pubkey.as_ref();
    verify_script_with_context_full(
        &pair.to_sign.inputs[0].script_sig,
        prevout_script,
        Some(&pair.witness),
        BLOCK_SCRIPT_VERIFY_FLAGS,
        &pair.to_sign,
        0,
        &[prevout_value],
        &[prevout_script],
        Some(height),
        None,
        Network::Signet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    )
}

struct SignetTxPair {
    to_spend: Transaction,
    to_sign: Transaction,
    witness: Witness,
}

fn build_signet_solution_txs(block: &Block, challenge: &[u8]) -> Result<Option<SignetTxPair>> {
    if block.transactions.is_empty() {
        return Ok(None);
    }
    let coinbase = &block.transactions[0];
    let Some(commit_idx) = witness_commitment_output_index(&coinbase.outputs) else {
        return Ok(None);
    };
    let mut modified_cb = coinbase.clone();
    let mut commitment_script = modified_cb.outputs[commit_idx].script_pubkey.to_vec();
    let signet_solution = fetch_and_clear_commitment_section(&mut commitment_script);
    modified_cb.outputs[commit_idx].script_pubkey = commitment_script;

    let mut tx_ids = Vec::with_capacity(block.transactions.len());
    tx_ids.push(calculate_tx_id(&modified_cb));
    for tx in block.transactions.iter().skip(1) {
        tx_ids.push(calculate_tx_id(tx));
    }
    let (signet_merkle, _) = compute_merkle_root_and_mutated(&tx_ids)?;

    let block_data = serialize_signet_block_data(
        block.header.version as i32,
        &block.header.prev_block_hash,
        &signet_merkle,
        block.header.timestamp as u32,
    );

    // BIP325: null prevout (index 0xffffffff) and scriptSig = OP_0 PUSH72(block_data).
    // Both are committed by the to_spend txid, which the solution signature signs.
    let to_spend = Transaction {
        version: 0,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: Hash::from([0u8; 32]),
                index: 0xffff_ffff,
            },
            script_sig: signet_to_spend_script_sig(&block_data),
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: challenge.to_vec(),
        }]
        .into(),
        lock_time: 0,
    };

    let spend_id = calculate_tx_id(&to_spend);

    let (script_sig, witness) = if let Some(solution) = signet_solution {
        match parse_signet_solution(&solution) {
            Ok(parsed) => parsed,
            Err(_) => return Ok(None),
        }
    } else {
        (Vec::new(), Witness::new())
    };

    let to_sign = Transaction {
        version: 0,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: spend_id,
                index: 0,
            },
            script_sig,
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![OP_RETURN],
        }]
        .into(),
        lock_time: 0,
    };

    Ok(Some(SignetTxPair {
        to_spend,
        to_sign,
        witness,
    }))
}

fn witness_commitment_output_index(outputs: &[TransactionOutput]) -> Option<usize> {
    const MAGIC: [u8; 4] = [0xaa, 0x21, 0xa9, 0xed];
    let mut last = None;
    for (i, output) in outputs.iter().enumerate() {
        let spk: &[u8] = output.script_pubkey.as_ref();
        if spk.len() >= 38 && spk[0] == OP_RETURN && spk[1] == 0x24 && spk[2..6] == MAGIC {
            last = Some(i);
        }
    }
    last
}

/// BIP325 to_spend scriptSig: `OP_0` then a push of the 72-byte header image.
fn signet_to_spend_script_sig(block_data: &[u8]) -> Vec<u8> {
    let mut script_sig = Vec::with_capacity(2 + block_data.len());
    script_sig.push(OP_0);
    push_script_data(&mut script_sig, block_data);
    script_sig
}

/// Minimal data push: direct length below 76, then PUSHDATA1/2/4.
fn push_script_data(out: &mut Vec<u8>, data: &[u8]) {
    let len = data.len();
    if len < OP_PUSHDATA1 as usize {
        out.push(len as u8);
    } else if len <= 0xff {
        out.push(OP_PUSHDATA1);
        out.push(len as u8);
    } else if len <= 0xffff {
        out.push(OP_PUSHDATA2);
        out.extend_from_slice(&(len as u16).to_le_bytes());
    } else if let Ok(len) = u32::try_from(len) {
        out.push(OP_PUSHDATA4);
        out.extend_from_slice(&len.to_le_bytes());
    } else {
        return;
    }
    out.extend_from_slice(data);
}

/// One opcode. Empty push data means the opcode byte itself is kept (`OP_0`, a zero-length push).
fn read_script_op(script: &[u8], i: usize) -> Option<(u8, &[u8], usize)> {
    if i >= script.len() {
        return None;
    }
    let opcode = script[i];
    if opcode > OP_PUSHDATA4 {
        return Some((opcode, &[], i + 1));
    }
    let (len, data_at) = if opcode < OP_PUSHDATA1 {
        (opcode as usize, i + 1)
    } else if opcode == OP_PUSHDATA1 {
        let at = i + 1;
        if at >= script.len() {
            return None;
        }
        (script[at] as usize, at + 1)
    } else if opcode == OP_PUSHDATA2 {
        let at = i + 1;
        if at + 2 > script.len() {
            return None;
        }
        let len = u16::from_le_bytes([script[at], script[at + 1]]) as usize;
        (len, at + 2)
    } else {
        let at = i + 1;
        if at + 4 > script.len() {
            return None;
        }
        let len = u32::from_le_bytes([script[at], script[at + 1], script[at + 2], script[at + 3]])
            as usize;
        (len, at + 4)
    };
    let end = data_at.checked_add(len)?;
    if end > script.len() {
        return None;
    }
    Some((opcode, &script[data_at..end], end))
}

fn fetch_and_clear_commitment_section(script: &mut Vec<u8>) -> Option<Vec<u8>> {
    let bytes = script.clone();
    let mut replacement = Vec::with_capacity(bytes.len());
    let mut found_header = false;
    let mut result = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let Some((opcode, pushdata, next)) = read_script_op(&bytes, i) else {
            break;
        };
        i = next;
        if !pushdata.is_empty() {
            let mut pushdata = pushdata.to_vec();
            if !found_header
                && pushdata.len() > SIGNET_HEADER.len()
                && pushdata[..SIGNET_HEADER.len()] == SIGNET_HEADER
            {
                result.extend_from_slice(&pushdata[SIGNET_HEADER.len()..]);
                pushdata.truncate(SIGNET_HEADER.len());
                found_header = true;
            }
            // Re-encode the remaining push. A stripped solution leaves the 4-byte
            // header, which is a direct push, not the original PUSHDATA opcode.
            push_script_data(&mut replacement, &pushdata);
        } else {
            replacement.push(opcode);
        }
    }
    if found_header {
        *script = replacement;
        Some(result)
    } else {
        None
    }
}

fn parse_signet_solution(data: &[u8]) -> Result<(Vec<u8>, Witness)> {
    let mut offset = 0;
    let (script_len, n) = decode_varint(&data[offset..]).map_err(|e| {
        crate::error::ConsensusError::BlockValidation(format!("signet scriptSig: {e}").into())
    })?;
    offset += n;
    let script_len = usize::try_from(script_len).map_err(|_| {
        crate::error::ConsensusError::BlockValidation("signet scriptSig length".into())
    })?;
    if offset + script_len > data.len() {
        return Err(crate::error::ConsensusError::BlockValidation(
            "signet scriptSig truncated".into(),
        ));
    }
    let script_sig = data[offset..offset + script_len].to_vec();
    offset += script_len;
    let (stack_len, n) = decode_varint(&data[offset..]).map_err(|e| {
        crate::error::ConsensusError::BlockValidation(format!("signet witness: {e}").into())
    })?;
    offset += n;
    let mut witness = Witness::new();
    for _ in 0..stack_len {
        let (item_len, n) = decode_varint(&data[offset..]).map_err(|e| {
            crate::error::ConsensusError::BlockValidation(
                format!("signet witness item: {e}").into(),
            )
        })?;
        offset += n;
        let item_len = usize::try_from(item_len).map_err(|_| {
            crate::error::ConsensusError::BlockValidation("signet witness item length".into())
        })?;
        if offset + item_len > data.len() {
            return Err(crate::error::ConsensusError::BlockValidation(
                "signet witness truncated".into(),
            ));
        }
        witness.push(data[offset..offset + item_len].to_vec());
        offset += item_len;
    }
    if offset != data.len() {
        return Err(crate::error::ConsensusError::BlockValidation(
            "signet solution extraneous data".into(),
        ));
    }
    Ok((script_sig, witness))
}

fn serialize_signet_block_data(
    version: i32,
    prev_hash: &Hash,
    signet_merkle: &Hash,
    timestamp: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(72);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(prev_hash);
    out.extend_from_slice(signet_merkle);
    out.extend_from_slice(&timestamp.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BlockHeader;

    #[test]
    fn signet_challenge_none_for_mainnet() {
        assert!(signet_challenge(Network::Mainnet, None).is_none());
    }

    #[test]
    fn signet_challenge_some_for_signet() {
        assert!(signet_challenge(Network::Signet, None).is_some());
    }

    #[test]
    fn check_signet_genesis_always_valid() {
        let block = Block {
            header: BlockHeader::default(),
            transactions: Box::new([]),
        };
        assert!(check_signet_block_solution(&block, default_signet_challenge(), 0).unwrap());
    }

    #[test]
    fn check_signet_empty_challenge_passes() {
        let block = Block {
            header: BlockHeader::default(),
            transactions: Box::new([]),
        };
        assert!(check_signet_block_solution(&block, &[], 1).unwrap());
    }

    #[test]
    fn check_signet_op_true_challenge_without_commitment_fails() {
        let block = Block {
            header: BlockHeader::default(),
            transactions: Box::new([]),
        };
        assert!(
            !check_signet_block_solution(&block, &[0x51], 1).unwrap(),
            "non-genesis without witness commitment must fail"
        );
    }

    fn witness_commitment_script(signet_payload: &[u8]) -> Vec<u8> {
        let mut push141 = vec![0xaa, 0x21, 0xa9, 0xed];
        push141.extend(std::iter::repeat_n(0xff, 32));
        assert!(signet_payload.len() <= 75);
        let mut script = vec![OP_RETURN, 0x24];
        script.extend(push141);
        script.push(signet_payload.len() as u8);
        script.extend(signet_payload);
        script
    }

    fn signet_test_block(signet_payload: &[u8]) -> Block {
        let coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: Hash::from([0u8; 32]),
                    index: 0xffffffff,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: witness_commitment_script(signet_payload).into(),
            }]
            .into(),
            lock_time: 0,
        };
        let dummy = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: Hash::from([1u8; 32]),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };
        Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: Hash::from([0u8; 32]),
                merkle_root: Hash::from([0u8; 32]),
                timestamp: 1_600_000_000,
                bits: 0x1e0377ae,
                nonce: 0,
            },
            transactions: vec![coinbase, dummy].into(),
        }
    }

    /// Signet parse smoke test (OP_TRUE challenge, header-only solution).
    #[test]
    fn check_signet_op_true_header_only_passes() {
        let block = signet_test_block(&SIGNET_HEADER);
        assert!(check_signet_block_solution(&block, &[0x51], 1).unwrap());
    }

    #[test]
    fn check_signet_op_true_premature_solution_fails() {
        let payload = [SIGNET_HEADER.as_slice(), &[0x01, 0x51]].concat();
        let block = signet_test_block(&payload);
        assert!(!check_signet_block_solution(&block, &[0x51], 1).unwrap());
    }

    #[test]
    fn check_signet_custom_challenge_override() {
        let challenge = signet_challenge(Network::Signet, Some(&[0x51])).unwrap();
        assert_eq!(challenge.as_ref() as &[u8], &[0x51]);
        let block = signet_test_block(&SIGNET_HEADER);
        assert!(check_signet_block_solution(&block, &[0x51], 1).unwrap());
    }

    #[test]
    fn signet_to_spend_uses_null_outpoint_and_op0_push() {
        let block = signet_test_block(&SIGNET_HEADER);
        let pair = build_signet_solution_txs(&block, &[0x51])
            .unwrap()
            .expect("commitment present");
        assert_eq!(pair.to_spend.inputs[0].prevout.hash, Hash::from([0u8; 32]));
        assert_eq!(pair.to_spend.inputs[0].prevout.index, 0xffff_ffff);
        assert_eq!(pair.to_sign.inputs[0].prevout.index, 0);
        let script: &[u8] = &pair.to_spend.inputs[0].script_sig;
        assert_eq!(script[0], OP_0);
        assert_eq!(script[1], 72);
        assert_eq!(script.len(), 74);
        assert_eq!(
            pair.to_sign.inputs[0].prevout.hash,
            calculate_tx_id(&pair.to_spend)
        );
    }

    fn test_header() -> BlockHeader {
        BlockHeader {
            version: 1,
            prev_block_hash: Hash::from([0u8; 32]),
            merkle_root: Hash::from([0u8; 32]),
            timestamp: 1_600_000_000,
            bits: 0x1e0377ae,
            nonce: 0,
        }
    }

    fn coinbase_and_dummy(commit_script: Vec<u8>) -> (Transaction, Transaction) {
        let coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: Hash::from([0u8; 32]),
                    index: 0xffffffff,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: commit_script.into(),
            }]
            .into(),
            lock_time: 0,
        };
        let dummy = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: Hash::from([1u8; 32]),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };
        (coinbase, dummy)
    }

    /// Expected to_spend scriptSig once the solution push has been reduced to the 4-byte header.
    fn expected_spend_script(
        header: &BlockHeader,
        modified_cb: &Transaction,
        dummy: &Transaction,
    ) -> Vec<u8> {
        let tx_ids = vec![calculate_tx_id(modified_cb), calculate_tx_id(dummy)];
        let (merkle, _) = compute_merkle_root_and_mutated(&tx_ids).unwrap();
        let block_data = serialize_signet_block_data(
            header.version as i32,
            &header.prev_block_hash,
            &merkle,
            header.timestamp as u32,
        );
        signet_to_spend_script_sig(&block_data)
    }

    #[test]
    fn signet_pushdata1_solution_strips_to_direct_header_push() {
        // Empty scriptSig and empty witness. Longer than a direct push so the
        // commitment uses PUSHDATA1; after the strip only the 4-byte header remains.
        let mut payload = SIGNET_HEADER.to_vec();
        payload.extend_from_slice(&[0x00, 0x00]);
        let mut commit = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
        commit.extend(std::iter::repeat_n(0xff, 32));
        commit.push(OP_PUSHDATA1);
        commit.push(payload.len() as u8);
        commit.extend_from_slice(&payload);

        let mut stripped = commit.clone();
        stripped.truncate(38);
        stripped.push(4);
        stripped.extend_from_slice(&SIGNET_HEADER);

        let header = test_header();
        let (coinbase, dummy) = coinbase_and_dummy(commit);
        let (modified_cb, _) = coinbase_and_dummy(stripped);
        let expected = expected_spend_script(&header, &modified_cb, &dummy);
        let block = Block {
            header,
            transactions: vec![coinbase, dummy].into(),
        };
        let pair = build_signet_solution_txs(&block, &[0x51])
            .unwrap()
            .expect("pushdata solution");
        assert_eq!(
            pair.to_spend.inputs[0].script_sig.as_slice(),
            expected.as_slice()
        );
        assert_eq!(pair.to_spend.inputs[0].prevout.index, 0xffff_ffff);
        assert!(check_signet_block_solution(&block, &[0x51], 1).unwrap());
    }

    #[cfg(feature = "production")]
    #[test]
    fn signet_p2pk_solution_verifies_bip325_message() {
        use crate::opcodes::OP_CHECKSIG;

        let seckey = [0x11u8; 32];
        let mut sec = blvm_secp256k1::scalar::Scalar::zero();
        assert!(!sec.set_b32(&seckey));
        assert!(!sec.is_zero());
        let pubkey = blvm_secp256k1::ecdsa::ge_to_compressed(
            &blvm_secp256k1::ecdsa::pubkey_from_secret(&sec),
        );
        let mut challenge = Vec::with_capacity(35);
        challenge.push(33);
        challenge.extend_from_slice(&pubkey);
        challenge.push(OP_CHECKSIG);

        let header = test_header();
        let mut stripped = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
        stripped.extend(std::iter::repeat_n(0xff, 32));
        stripped.push(4);
        stripped.extend_from_slice(&SIGNET_HEADER);
        let (modified_cb, dummy) = coinbase_and_dummy(stripped);
        let script_sig_spend = expected_spend_script(&header, &modified_cb, &dummy);

        let to_spend = Transaction {
            version: 0,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: Hash::from([0u8; 32]),
                    index: 0xffff_ffff,
                },
                script_sig: script_sig_spend.into(),
                sequence: 0,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: challenge.clone().into(),
            }]
            .into(),
            lock_time: 0,
        };
        let to_sign = Transaction {
            version: 0,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: calculate_tx_id(&to_spend),
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: vec![OP_RETURN].into(),
            }]
            .into(),
            lock_time: 0,
        };
        let sighash =
            crate::transaction_hash::compute_legacy_sighash_nocache(&to_sign, 0, &challenge, 0x01);
        let mut sig =
            blvm_secp256k1::ecdsa::ecdsa_sign_der_rfc6979(&sighash, &seckey).expect("sign");
        sig.push(0x01);
        let mut spend_script = Vec::with_capacity(1 + sig.len());
        spend_script.push(sig.len() as u8);
        spend_script.extend_from_slice(&sig);
        let mut solution = Vec::with_capacity(1 + spend_script.len() + 1);
        solution.push(spend_script.len() as u8);
        solution.extend_from_slice(&spend_script);
        solution.push(0);
        let mut payload = SIGNET_HEADER.to_vec();
        payload.extend_from_slice(&solution);
        assert!(payload.len() > 75, "solution must need PUSHDATA1");

        let mut commit = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
        commit.extend(std::iter::repeat_n(0xff, 32));
        commit.push(OP_PUSHDATA1);
        commit.push(payload.len() as u8);
        commit.extend_from_slice(&payload);
        let (coinbase, dummy) = coinbase_and_dummy(commit);
        let block = Block {
            header,
            transactions: vec![coinbase, dummy].into(),
        };
        assert!(check_signet_block_solution(&block, &challenge, 1).unwrap());
    }
}
