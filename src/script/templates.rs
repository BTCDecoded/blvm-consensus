//! Structural script-template scanner.
//!
//! Matches frequent script *shapes* (unexecuted IF + CHECKSIG remainder, OP_RETURN
//! prefixes, data-like n-of-m). Matching a template is not a protocol change and
//! does not interpret application-layer meaning.

use crate::error::Result;
use crate::opcodes::*;
use crate::script::{is_op_success, is_push_opcode, op_advance};
use crate::types::{ByteString, Transaction};

/// Known `OP_RETURN` payload prefixes (bytes after the push opcode, not brand names).
pub const NULLDATA_CNTRPRTY: &[u8] = b"CNTRPRTY";
pub const NULLDATA_OMNI: &[u8] = b"omni";
/// OpenTimestamps Bitcoin attestation tag (TimeAttestation.TAG).
pub const NULLDATA_OTS: &[u8] = &[0x05, 0x88, 0x96, 0x0d, 0x73, 0xd7, 0x19, 0x01];

const NULLDATA_MAGICS: &[&[u8]] = &[NULLDATA_CNTRPRTY, NULLDATA_OMNI, NULLDATA_OTS];

/// Parse result for a single script (spans live here, not on policy labels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptTemplate {
    UnexecIf {
        start: usize,
        end: usize,
        remainder: UnexecIfRemainder,
    },
    NullDataOp13,
    NullDataMagic {
        prefix: &'static [u8],
    },
    DataLikeMs {
        n: u8,
        m: u8,
        data_like_keys: usize,
    },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnexecIfRemainder {
    TapChecksig { pubkey: [u8; 32] },
}

/// Scan one script (tapscript, scriptPubKey, or witness element).
pub fn scan(script: &[u8]) -> ScriptTemplate {
    if script.is_empty() {
        return ScriptTemplate::Unknown;
    }
    if let Some(t) = scan_unexec_if(script) {
        return t;
    }
    if let Some(t) = scan_nulldata(script) {
        return t;
    }
    if let Some(t) = scan_data_like_ms(script) {
        return t;
    }
    ScriptTemplate::Unknown
}

/// Scan each witness element; first non-Unknown wins (UnexecIf is preferred).
pub fn scan_witness<'a, I>(elements: I) -> ScriptTemplate
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut first_other = ScriptTemplate::Unknown;
    for el in elements {
        match scan(el) {
            u @ ScriptTemplate::UnexecIf { .. } => return u,
            ScriptTemplate::Unknown => {}
            other => {
                if matches!(first_other, ScriptTemplate::Unknown) {
                    first_other = other;
                }
            }
        }
    }
    first_other
}

/// Classify a transaction: witnesses first, then outputs. First non-Unknown wins.
pub fn scan_tx(tx: &Transaction, witnesses: Option<&[crate::witness::Witness]>) -> ScriptTemplate {
    if let Some(wits) = witnesses {
        for w in wits {
            let hit = scan_witness(w.iter().map(|e| e.as_ref()));
            if !matches!(hit, ScriptTemplate::Unknown) {
                return hit;
            }
        }
    }
    for out in &tx.outputs {
        let hit = scan(out.script_pubkey.as_ref());
        if !matches!(hit, ScriptTemplate::Unknown) {
            return hit;
        }
    }
    ScriptTemplate::Unknown
}

fn has_op_success(script: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc < script.len() {
        if is_op_success(script[pc]) {
            return true;
        }
        let adv = op_advance(script, pc);
        if adv == 0 {
            return true;
        }
        pc = pc.saturating_add(adv);
        if pc > script.len() {
            return true;
        }
    }
    false
}

fn scan_unexec_if(script: &[u8]) -> Option<ScriptTemplate> {
    if has_op_success(script) {
        return None;
    }
    let (start, end) = find_single_unexec_if(script)?;
    let remainder = match_tap_checksig_remainder(script, start, end)?;
    Some(ScriptTemplate::UnexecIf {
        start,
        end,
        remainder,
    })
}

/// Exactly one `OP_0 OP_IF` … `OP_ENDIF` whose interior is data pushes only.
fn find_single_unexec_if(script: &[u8]) -> Option<(usize, usize)> {
    let mut pc = 0usize;
    let mut found: Option<(usize, usize)> = None;
    while pc < script.len() {
        if pc + 1 < script.len() && script[pc] == OP_0 && script[pc + 1] == OP_IF {
            let mut inner = pc + 2;
            loop {
                if inner >= script.len() {
                    return None;
                }
                let op = script[inner];
                if op == OP_ENDIF {
                    let end = inner + 1;
                    if found.is_some() {
                        return None;
                    }
                    found = Some((pc, end));
                    pc = end;
                    break;
                }
                if op == OP_IF || op == OP_NOTIF || op == OP_ELSE {
                    return None;
                }
                if !is_push_opcode(op) {
                    return None;
                }
                let adv = op_advance(script, inner);
                if adv == 0 || inner + adv > script.len() {
                    return None;
                }
                inner += adv;
            }
        } else {
            let adv = op_advance(script, pc);
            if adv == 0 || pc + adv > script.len() {
                return None;
            }
            pc += adv;
        }
    }
    found
}

fn match_tap_checksig_remainder(
    script: &[u8],
    start: usize,
    end: usize,
) -> Option<UnexecIfRemainder> {
    const LEAF_LEN: usize = 34;
    if start == LEAF_LEN && end == script.len() {
        return tap_checksig_leaf(&script[0..LEAF_LEN]);
    }
    if start == 0 && end + LEAF_LEN == script.len() {
        return tap_checksig_leaf(&script[end..]);
    }
    None
}

fn tap_checksig_leaf(leaf: &[u8]) -> Option<UnexecIfRemainder> {
    if leaf.len() != 34 || leaf[0] != PUSH_32_BYTES || leaf[33] != OP_CHECKSIG {
        return None;
    }
    let mut pubkey = [0u8; 32];
    pubkey.copy_from_slice(&leaf[1..33]);
    Some(UnexecIfRemainder::TapChecksig { pubkey })
}

fn scan_nulldata(script: &[u8]) -> Option<ScriptTemplate> {
    if script.is_empty() || script[0] != OP_RETURN {
        return None;
    }
    if script.len() == 1 {
        return None;
    }
    // `OP_RETURN OP_13` then zero or more pushes (runestone wrapper).
    if script[1] == OP_13 {
        let mut pc = 2usize;
        while pc < script.len() {
            if !is_push_opcode(script[pc]) {
                return None;
            }
            let adv = op_advance(script, pc);
            if adv == 0 || pc + adv > script.len() {
                return None;
            }
            pc += adv;
        }
        return Some(ScriptTemplate::NullDataOp13);
    }
    let payload = push_payload_after_op_return(script)?;
    for magic in NULLDATA_MAGICS {
        if payload.starts_with(magic) {
            return Some(ScriptTemplate::NullDataMagic { prefix: magic });
        }
    }
    None
}

/// Bytes after `OP_RETURN` and its first push opcode (compact-size or small-int push).
fn push_payload_after_op_return(script: &[u8]) -> Option<&[u8]> {
    if script.len() < 2 {
        return None;
    }
    let op = script[1];
    match op {
        0x01..=0x4b => {
            let n = op as usize;
            if script.len() < 2 + n {
                return None;
            }
            Some(&script[2..2 + n])
        }
        0x4c => {
            if script.len() < 3 {
                return None;
            }
            let n = script[2] as usize;
            if script.len() < 3 + n {
                return None;
            }
            Some(&script[3..3 + n])
        }
        0x4d => {
            if script.len() < 4 {
                return None;
            }
            let n = u16::from_le_bytes([script[2], script[3]]) as usize;
            if script.len() < 4 + n {
                return None;
            }
            Some(&script[4..4 + n])
        }
        OP_1..=OP_16 | OP_0 => Some(&[]),
        _ => None,
    }
}

fn scan_data_like_ms(script: &[u8]) -> Option<ScriptTemplate> {
    let (m, n, keys) = parse_bare_multisig_keys(script)?;
    if keys.is_empty() {
        return None;
    }
    let data_like = keys.iter().filter(|k| !looks_compressed_pubkey(k)).count();
    if data_like != keys.len() {
        return None;
    }
    Some(ScriptTemplate::DataLikeMs {
        n,
        m,
        data_like_keys: data_like,
    })
}

fn looks_compressed_pubkey(key: &[u8]) -> bool {
    key.len() == 33 && (key[0] == 0x02 || key[0] == 0x03)
}

fn parse_bare_multisig_keys(script: &[u8]) -> Option<(u8, u8, Vec<&[u8]>)> {
    if script.len() < 4 || script[script.len() - 1] != OP_CHECKMULTISIG {
        return None;
    }
    let n_op = script[script.len() - 2];
    if !(OP_1..=OP_16).contains(&n_op) {
        return None;
    }
    let n = n_op - OP_1 + 1;
    let m_op = script[0];
    if !(OP_1..=OP_16).contains(&m_op) {
        return None;
    }
    let m = m_op - OP_1 + 1;
    if m == 0 || m > n {
        return None;
    }
    let mut i = 1usize;
    let mut keys = Vec::with_capacity(n as usize);
    let trailer = script.len() - 2;
    for _ in 0..n {
        if i >= trailer {
            return None;
        }
        let first = script[i];
        let (start, len) = if (0x01..=0x4b).contains(&first) {
            (i + 1, first as usize)
        } else {
            return None;
        };
        if start + len > trailer {
            return None;
        }
        keys.push(&script[start..start + len]);
        i = start + len;
    }
    if i != trailer {
        return None;
    }
    Some((m, n, keys))
}

/// P2TR script-path: one UnexecIf + PUSH32 CHECKSIG remainder. Skips the interpreter walk.
#[cfg(feature = "production")]
#[allow(clippy::too_many_arguments)]
pub fn try_verify_p2tr_unexec_if_fast_path(
    script_sig: &ByteString,
    script_pubkey: &[u8],
    witness: &crate::witness::Witness,
    _flags: u32,
    tx: &Transaction,
    input_index: usize,
    prevout_values: &[i64],
    prevout_script_pubkeys: &[&[u8]],
    block_height: Option<u64>,
    network: crate::types::Network,
    schnorr_collector: Option<&crate::bip348::SchnorrSignatureCollector>,
) -> Option<Result<bool>> {
    use crate::activation::taproot_active_at_height;
    use crate::taproot::parse_taproot_script_path_witness;

    if !taproot_active_at_height(block_height, network) {
        return None;
    }
    if script_pubkey.len() != 34 || script_pubkey[0] != OP_1 || script_pubkey[1] != PUSH_32_BYTES {
        return None;
    }
    if !script_sig.is_empty() {
        return None;
    }
    if witness.len() < 2 {
        return None;
    }
    let mut output_key = [0u8; 32];
    output_key.copy_from_slice(&script_pubkey[2..34]);
    let (witness_body, annex_hash) = crate::taproot::strip_taproot_annex(witness);
    let parsed = match parse_taproot_script_path_witness(&witness_body, &output_key) {
        Ok(Some(p)) => p,
        Ok(None) | Err(_) => return None,
    };
    let (tapscript, stack_items, control_block) = parsed;
    if control_block.leaf_version != crate::taproot::TAPROOT_LEAF_VERSION_TAPSCRIPT {
        return None;
    }
    if stack_items.len() != 1 {
        return None;
    }
    let ScriptTemplate::UnexecIf {
        remainder: UnexecIfRemainder::TapChecksig { pubkey },
        ..
    } = scan(tapscript.as_ref())
    else {
        return None;
    };
    let (sig_bytes, sighash_type) =
        crate::bip348::try_parse_taproot_schnorr_witness_sig(stack_items[0].as_ref())?;
    let sighash = crate::taproot::compute_tapscript_signature_hash(
        tx,
        input_index,
        prevout_values,
        prevout_script_pubkeys,
        tapscript.as_ref(),
        control_block.leaf_version,
        0xffff_ffff,
        sighash_type,
        annex_hash.as_ref(),
    )
    .ok()?;
    let result = crate::bip348::verify_tapscript_schnorr_signature(
        &sighash,
        &pubkey,
        &sig_bytes,
        schnorr_collector,
    );
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push32_checksig(pk: [u8; 32]) -> Vec<u8> {
        let mut s = vec![PUSH_32_BYTES];
        s.extend_from_slice(&pk);
        s.push(OP_CHECKSIG);
        s
    }

    fn envelope(body: &[u8]) -> Vec<u8> {
        let mut s = vec![OP_0, OP_IF];
        s.extend_from_slice(body);
        s.push(OP_ENDIF);
        s
    }

    #[test]
    fn unexec_if_suffix_checksig() {
        let pk = [0x11u8; 32];
        let mut script = envelope(&[0x03, b'o', b'r', b'd']);
        script.extend(push32_checksig(pk));
        match scan(&script) {
            ScriptTemplate::UnexecIf {
                remainder: UnexecIfRemainder::TapChecksig { pubkey },
                ..
            } => assert_eq!(pubkey, pk),
            other => panic!("expected UnexecIf, got {other:?}"),
        }
    }

    #[test]
    fn unexec_if_prefix_checksig() {
        let pk = [0x22u8; 32];
        let mut script = push32_checksig(pk);
        script.extend(envelope(&[0x03, b'o', b'r', b'd']));
        match scan(&script) {
            ScriptTemplate::UnexecIf {
                remainder: UnexecIfRemainder::TapChecksig { pubkey },
                ..
            } => assert_eq!(pubkey, pk),
            other => panic!("expected UnexecIf, got {other:?}"),
        }
    }

    #[test]
    fn two_envelopes_unknown() {
        let mut script = envelope(&[0x01, 0xaa]);
        script.extend(envelope(&[0x01, 0xbb]));
        script.extend(push32_checksig([0x33; 32]));
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }

    #[test]
    fn nested_if_unknown() {
        let mut script = vec![OP_0, OP_IF, OP_0, OP_IF, OP_ENDIF, OP_ENDIF];
        script.extend(push32_checksig([0x44; 32]));
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }

    #[test]
    fn leftover_opcode_unknown() {
        let mut script = envelope(&[0x01, 0xaa]);
        script.push(OP_DUP);
        script.extend(push32_checksig([0x55; 32]));
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }

    #[test]
    fn leftover_after_checksig_unknown() {
        let mut script = envelope(&[0x01, 0xaa]);
        script.extend(push32_checksig([0x55; 32]));
        script.push(OP_DROP);
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }

    #[test]
    fn op_success_unknown() {
        // 0xba is OP_SUCCESS in tapscript (187..=254).
        let mut script = vec![OP_0, OP_IF, 0xba, OP_ENDIF];
        script.extend(push32_checksig([0x66; 32]));
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }

    #[test]
    fn nulldata_op13() {
        let script = vec![OP_RETURN, OP_13, 0x01, 0xff];
        assert_eq!(scan(&script), ScriptTemplate::NullDataOp13);
    }

    #[test]
    fn nulldata_cntrprty() {
        let mut script = vec![OP_RETURN, NULLDATA_CNTRPRTY.len() as u8];
        script.extend_from_slice(NULLDATA_CNTRPRTY);
        match scan(&script) {
            ScriptTemplate::NullDataMagic { prefix } => assert_eq!(prefix, NULLDATA_CNTRPRTY),
            other => panic!("expected NullDataMagic, got {other:?}"),
        }
    }

    #[test]
    fn data_like_ms_all_invalid_prefix() {
        // 1-of-2 with 8-byte "keys" (not 02/03+32).
        let mut script = vec![OP_1, 0x08];
        script.extend_from_slice(&[0x11; 8]);
        script.push(0x08);
        script.extend_from_slice(&[0x22; 8]);
        script.push(OP_2);
        script.push(OP_CHECKMULTISIG);
        match scan(&script) {
            ScriptTemplate::DataLikeMs {
                n,
                m,
                data_like_keys,
            } => {
                assert_eq!((m, n, data_like_keys), (1, 2, 2));
            }
            other => panic!("expected DataLikeMs, got {other:?}"),
        }
    }

    #[test]
    fn real_compressed_key_not_data_like() {
        let mut script = vec![OP_1, PUSH_33_BYTES, 0x02];
        script.extend_from_slice(&[0xab; 32]);
        script.push(OP_1);
        script.push(OP_CHECKMULTISIG);
        assert_eq!(scan(&script), ScriptTemplate::Unknown);
    }
}
