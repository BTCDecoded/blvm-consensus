//! Parse Bitcoin Core `script_tests.json` script assembly strings into bytecode.

use blvm_consensus::opcodes::*;

/// Push raw bytes using standard script push encoding.
pub fn push_data(script: &mut Vec<u8>, data: &[u8]) {
    let len = data.len();
    if len <= 75 {
        script.push(len as u8);
    } else if len <= 255 {
        script.push(OP_PUSHDATA1);
        script.push(len as u8);
    } else if len <= 65535 {
        script.push(OP_PUSHDATA2);
        script.extend_from_slice(&(len as u16).to_le_bytes());
    } else {
        script.push(OP_PUSHDATA4);
        script.extend_from_slice(&(len as u32).to_le_bytes());
    }
    script.extend_from_slice(data);
}

fn push_script_num(script: &mut Vec<u8>, n: i64) {
    if n == -1 {
        script.push(OP_1NEGATE);
        return;
    }
    if (0..=16).contains(&n) {
        script.push(if n == 0 { OP_0 } else { OP_1 + (n as u8) - 1 });
        return;
    }
    push_data(script, &encode_script_num(n));
}

/// Encode a signed integer as a minimal CScriptNum (Bitcoin script number encoding).
/// Little-endian bytes with sign bit packed into the MSB of the last byte.
fn encode_script_num(n: i64) -> Vec<u8> {
    if n == 0 {
        return vec![];
    }
    let neg = n < 0;
    let mut absvalue = n.unsigned_abs();
    let mut result = Vec::new();
    while absvalue > 0 {
        result.push((absvalue & 0xff) as u8);
        absvalue >>= 8;
    }
    if result.last().is_some_and(|&b| b & 0x80 != 0) {
        result.push(if neg { 0x80 } else { 0x00 });
    } else if neg {
        *result.last_mut().unwrap() |= 0x80;
    }
    result
}

fn push_opcode_name(script: &mut Vec<u8>, name: &str) -> bool {
    let op = match name {
        "0" | "OP_0" | "FALSE" => OP_0,
        "OP_1NEGATE" | "1NEGATE" => OP_1NEGATE,
        "1" | "OP_1" | "TRUE" => OP_1,
        "2" | "OP_2" => OP_2,
        "3" | "OP_3" => OP_3,
        "4" | "OP_4" => OP_4,
        "5" | "OP_5" => OP_5,
        "6" | "OP_6" => OP_6,
        "7" | "OP_7" => OP_7,
        "8" | "OP_8" => OP_8,
        "9" | "OP_9" => OP_9,
        "10" | "OP_10" => OP_10,
        "11" | "OP_11" => OP_11,
        "12" | "OP_12" => OP_12,
        "13" | "OP_13" => OP_13,
        "14" | "OP_14" => OP_14,
        "15" | "OP_15" => OP_15,
        "16" | "OP_16" => OP_16,
        "NOP" | "OP_NOP" => OP_NOP,
        "IF" | "OP_IF" => OP_IF,
        "NOTIF" | "OP_NOTIF" => OP_NOTIF,
        "ELSE" | "OP_ELSE" => OP_ELSE,
        "ENDIF" | "OP_ENDIF" => OP_ENDIF,
        "VERIFY" | "OP_VERIFY" => OP_VERIFY,
        "RETURN" | "OP_RETURN" => OP_RETURN,
        "TOALTSTACK" | "OP_TOALTSTACK" => OP_TOALTSTACK,
        "FROMALTSTACK" | "OP_FROMALTSTACK" => OP_FROMALTSTACK,
        "DEPTH" | "OP_DEPTH" => OP_DEPTH,
        "DROP" | "OP_DROP" => OP_DROP,
        "DUP" | "OP_DUP" => OP_DUP,
        "EQUAL" | "OP_EQUAL" => OP_EQUAL,
        "EQUALVERIFY" | "OP_EQUALVERIFY" => OP_EQUALVERIFY,
        "ADD" | "OP_ADD" => OP_ADD,
        "SUB" | "OP_SUB" => OP_SUB,
        "HASH160" | "OP_HASH160" => OP_HASH160,
        "HASH256" | "OP_HASH256" => OP_HASH256,
        "SHA1" | "OP_SHA1" => OP_SHA1,
        "SHA256" | "OP_SHA256" => OP_SHA256,
        "CHECKSIG" | "OP_CHECKSIG" => OP_CHECKSIG,
        "CHECKSIGVERIFY" | "OP_CHECKSIGVERIFY" => OP_CHECKSIGVERIFY,
        "CHECKMULTISIG" | "OP_CHECKMULTISIG" => OP_CHECKMULTISIG,
        "CHECKMULTISIGVERIFY" | "OP_CHECKMULTISIGVERIFY" => OP_CHECKMULTISIGVERIFY,
        "CHECKLOCKTIMEVERIFY" | "OP_CHECKLOCKTIMEVERIFY" | "NOP2" => OP_CHECKLOCKTIMEVERIFY,
        "CHECKSEQUENCEVERIFY" | "OP_CHECKSEQUENCEVERIFY" | "NOP3" => OP_CHECKSEQUENCEVERIFY,
        "IFDUP" | "OP_IFDUP" => OP_IFDUP,
        "2ROT" | "OP_2ROT" => OP_2ROT,
        "RIPEMD160" | "OP_RIPEMD160" => OP_RIPEMD160,
        "CODESEPARATOR" | "OP_CODESEPARATOR" => OP_CODESEPARATOR,
        "RESERVED1" | "OP_RESERVED1" => OP_RESERVED1,
        "RESERVED2" | "OP_RESERVED2" => OP_RESERVED2,
        "0NOTEQUAL" | "OP_0NOTEQUAL" => OP_0NOTEQUAL,
        "BOOLAND" | "OP_BOOLAND" => OP_BOOLAND,
        "BOOLOR" | "OP_BOOLOR" => OP_BOOLOR,
        "ABS" | "OP_ABS" => OP_ABS,
        "NEGATE" | "OP_NEGATE" => OP_NEGATE,
        "NUMEQUAL" | "OP_NUMEQUAL" => OP_NUMEQUAL,
        "NUMEQUALVERIFY" | "OP_NUMEQUALVERIFY" => OP_NUMEQUALVERIFY,
        "NUMNOTEQUAL" | "OP_NUMNOTEQUAL" => OP_NUMNOTEQUAL,
        "LESSTHAN" | "OP_LESSTHAN" => OP_LESSTHAN,
        "GREATERTHAN" | "OP_GREATERTHAN" => OP_GREATERTHAN,
        "LESSTHANOREQUAL" | "OP_LESSTHANOREQUAL" => OP_LESSTHANOREQUAL,
        "GREATERTHANOREQUAL" | "OP_GREATERTHANOREQUAL" => OP_GREATERTHANOREQUAL,
        "MIN" | "OP_MIN" => OP_MIN,
        "MAX" | "OP_MAX" => OP_MAX,
        "WITHIN" | "OP_WITHIN" => OP_WITHIN,
        "NOP1" | "OP_NOP1" => OP_NOP1,
        "NOP4" | "OP_NOP4" => OP_NOP4,
        "NOP5" | "OP_NOP5" => OP_NOP5,
        "NOP6" | "OP_NOP6" => OP_NOP6,
        "NOP7" | "OP_NOP7" => OP_NOP7,
        "NOP8" | "OP_NOP8" => OP_NOP8,
        "NOP9" | "OP_NOP9" => OP_NOP9,
        "NOP10" | "OP_NOP10" => OP_NOP10,
        "PICK" | "OP_PICK" => OP_PICK,
        "ROLL" | "OP_ROLL" => OP_ROLL,
        "ROT" | "OP_ROT" => OP_ROT,
        "SWAP" | "OP_SWAP" => OP_SWAP,
        "SIZE" | "OP_SIZE" => OP_SIZE,
        "NOT" | "OP_NOT" => OP_NOT,
        "AND" | "OP_AND" => OP_AND,
        "OR" | "OP_OR" => OP_OR,
        "XOR" | "OP_XOR" => OP_XOR,
        "OVER" | "OP_OVER" => OP_OVER,
        "NIP" | "OP_NIP" => OP_NIP,
        "TUCK" | "OP_TUCK" => OP_TUCK,
        "2DROP" | "OP_2DROP" => OP_2DROP,
        "2DUP" | "OP_2DUP" => OP_2DUP,
        "3DUP" | "OP_3DUP" => OP_3DUP,
        "2OVER" | "OP_2OVER" => OP_2OVER,
        "2SWAP" | "OP_2SWAP" => OP_2SWAP,
        "CAT" | "OP_CAT" => OP_CAT,
        "LEFT" | "OP_LEFT" => OP_LEFT,
        "RIGHT" | "OP_RIGHT" => OP_RIGHT,
        "INVERT" | "OP_INVERT" => OP_INVERT,
        "RESERVED" | "OP_RESERVED" => OP_RESERVED,
        "VER" | "OP_VER" => OP_VER,
        "VERIF" | "OP_VERIF" => OP_VERIF,
        "VERNOTIF" | "OP_VERNOTIF" => OP_VERNOTIF,
        _ => return false,
    };
    script.push(op);
    true
}

/// Parse a Core script assembly string (subset used by early `script_tests.json` cases).
pub fn parse_core_script_asm(script_str: &str) -> Option<Vec<u8>> {
    let mut script = Vec::new();
    let mut i = 0;
    let bytes = script_str.as_bytes();

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        if bytes[i] == b'\'' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'\'' {
                i += 1;
            }
            if i >= bytes.len() {
                return None;
            }
            push_data(&mut script, &bytes[start..i]);
            i += 1;
            continue;
        }

        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let token = std::str::from_utf8(&bytes[start..i]).ok()?;

        if let Some(hex) = token.strip_prefix("0x") {
            if hex.is_empty() || !hex.chars().all(|c| c.is_ascii_hexdigit()) || hex.len() % 2 != 0 {
                return None;
            }
            let raw = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
                .collect::<Option<Vec<u8>>>()?;
            script.extend_from_slice(&raw);
            continue;
        }

        if push_opcode_name(&mut script, token) {
            continue;
        }

        if let Ok(n) = token.parse::<i64>() {
            push_script_num(&mut script, n);
            continue;
        }

        return None;
    }

    Some(script)
}
