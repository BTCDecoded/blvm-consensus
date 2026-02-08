//! Bitcoin Script Opcode Constants
//!
//! Complete set of opcode constants for Bitcoin script execution.
//! All opcodes are defined with their hex values and descriptive comments.
//!
//! Reference: Bitcoin Core `script/script.h` and BIP specifications

// ============================================================================
// PUSH DATA OPCODES (0x00 - 0x4e)
// ============================================================================

/// OP_0 / OP_FALSE - Push empty array
pub const OP_0: u8 = 0x00;
pub const OP_FALSE: u8 = 0x00;

/// OP_PUSHDATA1 - Push next byte as data length
pub const OP_PUSHDATA1: u8 = 0x4c;

/// OP_PUSHDATA2 - Push next 2 bytes (little-endian) as data length
pub const OP_PUSHDATA2: u8 = 0x4d;

/// OP_PUSHDATA4 - Push next 4 bytes (little-endian) as data length
pub const OP_PUSHDATA4: u8 = 0x4e;

// ============================================================================
// PUSH VALUE OPCODES (0x4f - 0x60)
// ============================================================================

/// OP_1NEGATE - Push -1 onto stack
pub const OP_1NEGATE: u8 = 0x4f;

/// OP_RESERVED - Reserved opcode, transaction invalid if present
pub const OP_RESERVED: u8 = 0x50;

/// OP_1 / OP_TRUE - Push 1 onto stack
pub const OP_1: u8 = 0x51;
pub const OP_TRUE: u8 = 0x51;

/// OP_2 - Push 2 onto stack
pub const OP_2: u8 = 0x52;

/// OP_3 - Push 3 onto stack
pub const OP_3: u8 = 0x53;

/// OP_4 - Push 4 onto stack
pub const OP_4: u8 = 0x54;

/// OP_5 - Push 5 onto stack
pub const OP_5: u8 = 0x55;

/// OP_6 - Push 6 onto stack
pub const OP_6: u8 = 0x56;

/// OP_7 - Push 7 onto stack
pub const OP_7: u8 = 0x57;

/// OP_8 - Push 8 onto stack
pub const OP_8: u8 = 0x58;

/// OP_9 - Push 9 onto stack
pub const OP_9: u8 = 0x59;

/// OP_10 - Push 10 onto stack
pub const OP_10: u8 = 0x5a;

/// OP_11 - Push 11 onto stack
pub const OP_11: u8 = 0x5b;

/// OP_12 - Push 12 onto stack
pub const OP_12: u8 = 0x5c;

/// OP_13 - Push 13 onto stack
pub const OP_13: u8 = 0x5d;

/// OP_14 - Push 14 onto stack
pub const OP_14: u8 = 0x5e;

/// OP_15 - Push 15 onto stack
pub const OP_15: u8 = 0x5f;

/// OP_16 - Push 16 onto stack
pub const OP_16: u8 = 0x60;

// ============================================================================
// CONTROL FLOW OPCODES (0x61 - 0x6f)
// ============================================================================

/// OP_NOP - No operation
pub const OP_NOP: u8 = 0x61;

/// OP_VER - Reserved opcode, disabled
pub const OP_VER: u8 = 0x62;

/// OP_IF - If top stack value is true, statements are executed
pub const OP_IF: u8 = 0x63;

/// OP_NOTIF - If top stack value is false, statements are executed
pub const OP_NOTIF: u8 = 0x64;

/// OP_VERIF - Reserved opcode, disabled
pub const OP_VERIF: u8 = 0x65;

/// OP_VERNOTIF - Reserved opcode, disabled
pub const OP_VERNOTIF: u8 = 0x66;

/// OP_ELSE - If the preceding OP_IF or OP_NOTIF was not executed, these statements are
pub const OP_ELSE: u8 = 0x67;

/// OP_ENDIF - Ends an OP_IF/OP_NOTIF/OP_ELSE block
pub const OP_ENDIF: u8 = 0x68;

/// OP_VERIFY - Marks transaction as invalid if top stack value is not true
pub const OP_VERIFY: u8 = 0x69;

/// OP_RETURN - Marks transaction as invalid
pub const OP_RETURN: u8 = 0x6a;

// ============================================================================
// STACK OPERATIONS (0x6b - 0x7d)
// ============================================================================

/// OP_TOALTSTACK - Puts the input onto the top of the alt stack. Removes it from the main stack
pub const OP_TOALTSTACK: u8 = 0x6b;

/// OP_FROMALTSTACK - Puts the input onto the top of the main stack. Removes it from the alt stack
pub const OP_FROMALTSTACK: u8 = 0x6c;

/// OP_2DROP - Removes the top two stack items
pub const OP_2DROP: u8 = 0x6d;

/// OP_2DUP - Duplicates the top two stack items
pub const OP_2DUP: u8 = 0x6e;

/// OP_3DUP - Duplicates the top three stack items
pub const OP_3DUP: u8 = 0x6f;

/// OP_2OVER - Copies the pair of items two spaces back in the stack to the front
pub const OP_2OVER: u8 = 0x70;

/// OP_2ROT - The fifth and sixth items back are moved to the top of the stack
pub const OP_2ROT: u8 = 0x71;

/// OP_2SWAP - Swaps the top two pairs of items
pub const OP_2SWAP: u8 = 0x72;

/// OP_IFDUP - If the top stack value is not 0, duplicate it
pub const OP_IFDUP: u8 = 0x73;

/// OP_DEPTH - Puts the number of stack items onto the stack
pub const OP_DEPTH: u8 = 0x74;

/// OP_DROP - Removes the top stack item
pub const OP_DROP: u8 = 0x75;

/// OP_DUP - Duplicates the top stack item
pub const OP_DUP: u8 = 0x76;

/// OP_NIP - Removes the second-to-top stack item
pub const OP_NIP: u8 = 0x77;

/// OP_OVER - Copies the second-to-top stack item to the top
pub const OP_OVER: u8 = 0x78;

/// OP_PICK - The item n back in the stack is copied to the top
pub const OP_PICK: u8 = 0x79;

/// OP_ROLL - The item n back in the stack is moved to the top
pub const OP_ROLL: u8 = 0x7a;

/// OP_ROT - The top three items on the stack are rotated to the left
pub const OP_ROT: u8 = 0x7b;

/// OP_SWAP - The top two items on the stack are swapped
pub const OP_SWAP: u8 = 0x7c;

/// OP_TUCK - The item at the top of the stack is copied and inserted before the second-to-top item
pub const OP_TUCK: u8 = 0x7d;

// ============================================================================
// STRING OPERATIONS (0x7e - 0x8a)
// ============================================================================

/// OP_CAT - Concatenates two strings (disabled)
pub const OP_CAT: u8 = 0x7e;

/// OP_SUBSTR - Returns a section of a string (disabled)
pub const OP_SUBSTR: u8 = 0x7f;

/// OP_LEFT - Keeps only characters left of the specified point in a string (disabled)
pub const OP_LEFT: u8 = 0x80;

/// OP_RIGHT - Keeps only characters right of the specified point in a string (disabled)
pub const OP_RIGHT: u8 = 0x81;

/// OP_SIZE - Pushes the string length of the top element of the stack (without popping it)
pub const OP_SIZE: u8 = 0x82;

// ============================================================================
// BITWISE LOGIC (0x83 - 0x8a)
// ============================================================================

/// OP_INVERT - Flips all of the bits in the input (disabled)
pub const OP_INVERT: u8 = 0x83;

/// OP_AND - Boolean AND between each bit in the inputs (disabled)
pub const OP_AND: u8 = 0x84;

/// OP_OR - Boolean OR between each bit in the inputs (disabled)
pub const OP_OR: u8 = 0x85;

/// OP_XOR - Boolean exclusive OR between each bit in the inputs (disabled)
pub const OP_XOR: u8 = 0x86;

/// OP_EQUAL - Returns 1 if the inputs are exactly equal, 0 otherwise
pub const OP_EQUAL: u8 = 0x87;

/// OP_EQUALVERIFY - Same as OP_EQUAL, but runs OP_VERIFY afterward
pub const OP_EQUALVERIFY: u8 = 0x88;

/// OP_RESERVED1 - Reserved opcode
pub const OP_RESERVED1: u8 = 0x89;

/// OP_RESERVED2 - Reserved opcode
pub const OP_RESERVED2: u8 = 0x8a;

// ============================================================================
// NUMERIC OPERATIONS (0x8b - 0xa5)
// ============================================================================

/// OP_1ADD - 1 is added to the input
pub const OP_1ADD: u8 = 0x8b;

/// OP_1SUB - 1 is subtracted from the input
pub const OP_1SUB: u8 = 0x8c;

/// OP_2MUL - The input is multiplied by 2 (disabled)
pub const OP_2MUL: u8 = 0x8d;

/// OP_2DIV - The input is divided by 2 (disabled)
pub const OP_2DIV: u8 = 0x8e;

/// OP_NEGATE - The sign of the input is flipped
pub const OP_NEGATE: u8 = 0x8f;

/// OP_ABS - The input is made positive
pub const OP_ABS: u8 = 0x90;

/// OP_NOT - If the input is 0 or 1, it is flipped. Otherwise the output is 0
pub const OP_NOT: u8 = 0x91;

/// OP_0NOTEQUAL - Returns 0 if the input is 0. 1 otherwise
pub const OP_0NOTEQUAL: u8 = 0x92;

/// OP_ADD - a is added to b
pub const OP_ADD: u8 = 0x93;

/// OP_SUB - b is subtracted from a
pub const OP_SUB: u8 = 0x94;

/// OP_MUL - a is multiplied by b (disabled)
pub const OP_MUL: u8 = 0x95;

/// OP_DIV - a is divided by b (disabled)
pub const OP_DIV: u8 = 0x96;

/// OP_MOD - Returns the remainder after dividing a by b (disabled)
pub const OP_MOD: u8 = 0x97;

/// OP_LSHIFT - Shifts a left b bits, preserving sign (disabled)
pub const OP_LSHIFT: u8 = 0x98;

/// OP_RSHIFT - Shifts a right b bits, preserving sign (disabled)
pub const OP_RSHIFT: u8 = 0x99;

/// OP_BOOLAND - If both a and b are not 0, the output is 1. Otherwise 0
pub const OP_BOOLAND: u8 = 0x9a;

/// OP_BOOLOR - If a or b is not 0, the output is 1. Otherwise 0
pub const OP_BOOLOR: u8 = 0x9b;

/// OP_NUMEQUAL - Returns 1 if the numbers are equal, 0 otherwise
pub const OP_NUMEQUAL: u8 = 0x9c;

/// OP_NUMEQUALVERIFY - Same as OP_NUMEQUAL, but runs OP_VERIFY afterward
pub const OP_NUMEQUALVERIFY: u8 = 0x9d;

/// OP_NUMNOTEQUAL - Returns 1 if the numbers are not equal, 0 otherwise
pub const OP_NUMNOTEQUAL: u8 = 0x9e;

/// OP_LESSTHAN - Returns 1 if a is less than b, 0 otherwise
pub const OP_LESSTHAN: u8 = 0x9f;

/// OP_GREATERTHAN - Returns 1 if a is greater than b, 0 otherwise
pub const OP_GREATERTHAN: u8 = 0xa0;

/// OP_LESSTHANOREQUAL - Returns 1 if a is less than or equal to b, 0 otherwise
pub const OP_LESSTHANOREQUAL: u8 = 0xa1;

/// OP_GREATERTHANOREQUAL - Returns 1 if a is greater than or equal to b, 0 otherwise
pub const OP_GREATERTHANOREQUAL: u8 = 0xa2;

/// OP_MIN - Returns the smaller of a and b
pub const OP_MIN: u8 = 0xa3;

/// OP_MAX - Returns the larger of a and b
pub const OP_MAX: u8 = 0xa4;

/// OP_WITHIN - Returns 1 if x is within the specified range (left-inclusive), 0 otherwise
pub const OP_WITHIN: u8 = 0xa5;

// ============================================================================
// CRYPTOGRAPHIC OPERATIONS (0xa6 - 0xab)
// ============================================================================

/// OP_RIPEMD160 - The input is hashed using RIPEMD-160
pub const OP_RIPEMD160: u8 = 0xa6;

/// OP_SHA1 - The input is hashed using SHA-1
pub const OP_SHA1: u8 = 0xa7;

/// OP_SHA256 - The input is hashed using SHA-256
pub const OP_SHA256: u8 = 0xa8;

/// OP_HASH160 - The input is hashed twice: first with SHA-256 and then with RIPEMD-160
pub const OP_HASH160: u8 = 0xa9;

/// OP_HASH256 - The input is hashed two times with SHA-256
pub const OP_HASH256: u8 = 0xaa;

/// OP_CODESEPARATOR - All of the signature checking words will only match signatures to the data after the most recently-executed OP_CODESEPARATOR
pub const OP_CODESEPARATOR: u8 = 0xab;

// ============================================================================
// SIGNATURE OPERATIONS (0xac - 0xaf)
// ============================================================================

/// OP_CHECKSIG - The entire transaction's outputs, inputs, and script (from the most recently-executed OP_CODESEPARATOR to the end) are hashed. The signature used must match the transaction and the public key or script verification fails
pub const OP_CHECKSIG: u8 = 0xac;

/// OP_CHECKSIGVERIFY - Same as OP_CHECKSIG, but OP_VERIFY is executed afterward
pub const OP_CHECKSIGVERIFY: u8 = 0xad;

/// OP_CHECKMULTISIG - Compares the first signature against each public key until it finds an ECDSA match. Starting with the subsequent public key, it compares the second signature against each remaining public key until it finds an ECDSA match. The process is repeated until all signatures have been checked or not enough public keys remain to produce a successful result. All signatures need to match a public key. Because public keys are not checked again if they fail any signature comparison, signatures must be placed in the scriptSig using the same order as their corresponding public keys were placed in the scriptPubKey or script. If all signatures are valid, 1 is returned, 0 otherwise
pub const OP_CHECKMULTISIG: u8 = 0xae;

/// OP_CHECKMULTISIGVERIFY - Same as OP_CHECKMULTISIG, but OP_VERIFY is executed afterward
pub const OP_CHECKMULTISIGVERIFY: u8 = 0xaf;

// ============================================================================
// NOP OPCODES (0xb0 - 0xb9)
// ============================================================================

/// OP_NOP1 - Reserved for future use (was OP_EVAL, disabled)
pub const OP_NOP1: u8 = 0xb0;

/// OP_CHECKLOCKTIMEVERIFY (BIP65) - Marks transaction as invalid if the top stack item is greater than the transaction's nLockTime field, otherwise script evaluation continues as if an OP_NOP was executed
pub const OP_CHECKLOCKTIMEVERIFY: u8 = 0xb1;
pub const OP_NOP2: u8 = 0xb1; // Alias for OP_CHECKLOCKTIMEVERIFY

/// OP_CHECKSEQUENCEVERIFY (BIP112) - Marks transaction as invalid if the relative lock time of the input (enforced by BIP 68 with nSequence) is not equal to or longer than the value of the top stack item
pub const OP_CHECKSEQUENCEVERIFY: u8 = 0xb2;
pub const OP_NOP3: u8 = 0xb2; // Alias for OP_CHECKSEQUENCEVERIFY

/// OP_CHECKTEMPLATEVERIFY (BIP119) - Verifies that the transaction matches a template hash
pub const OP_CHECKTEMPLATEVERIFY: u8 = 0xb3;
pub const OP_NOP4: u8 = 0xb3; // Alias for OP_CHECKTEMPLATEVERIFY

/// OP_NOP5 - Reserved for future use
pub const OP_NOP5: u8 = 0xb4;

/// OP_NOP6 - Reserved for future use
pub const OP_NOP6: u8 = 0xb5;

/// OP_NOP7 - Reserved for future use
pub const OP_NOP7: u8 = 0xb6;

/// OP_NOP8 - Reserved for future use
pub const OP_NOP8: u8 = 0xb7;

/// OP_NOP9 - Reserved for future use
pub const OP_NOP9: u8 = 0xb8;

/// OP_NOP10 - Reserved for future use
pub const OP_NOP10: u8 = 0xb9;

// ============================================================================
// TAPSCRIPT OPCODES (0xba - 0xcc)
// ============================================================================

/// OP_CHECKSIGADD (BIP342) - Tapscript opcode for signature aggregation
pub const OP_CHECKSIGADD: u8 = 0xba;

/// OP_CHECKSIGFROMSTACK (BIP348) - Verifies a BIP340 Schnorr signature against an arbitrary message
pub const OP_CHECKSIGFROMSTACK: u8 = 0xcc;

// ============================================================================
// HELPER CONSTANTS
// ============================================================================

/// Base value for OP_1 through OP_16 (OP_1 = 0x50 + 1 = 0x51)
pub const OP_N_BASE: u8 = 0x50;

/// Range for OP_1 through OP_16
pub const OP_1_RANGE_START: u8 = OP_1;
pub const OP_1_RANGE_END: u8 = OP_16;

/// Range for OP_NOP opcodes (0xb0 - 0xb9)
pub const OP_NOP_RANGE_START: u8 = OP_NOP1;
pub const OP_NOP_RANGE_END: u8 = OP_NOP10;

/// Range for disabled opcodes (string operations)
pub const OP_DISABLED_STRING_RANGE_START: u8 = OP_CAT;
pub const OP_DISABLED_STRING_RANGE_END: u8 = OP_RIGHT;

/// Range for disabled opcodes (bitwise operations)
pub const OP_DISABLED_BITWISE_RANGE_START: u8 = OP_INVERT;
pub const OP_DISABLED_BITWISE_RANGE_END: u8 = OP_XOR;

/// Range for disabled opcodes (numeric operations)
pub const OP_DISABLED_NUMERIC_RANGE_START: u8 = OP_2MUL;
pub const OP_DISABLED_NUMERIC_RANGE_END: u8 = OP_RSHIFT;

// ============================================================================
// PUSH DATA HELPER CONSTANTS
// ============================================================================

/// Push 20 bytes (used in P2WPKH: OP_0 0x14 <20-byte-hash>)
pub const PUSH_20_BYTES: u8 = 0x14;

/// Push 32 bytes (used in P2WSH and P2TR: OP_0/OP_1 0x20 <32-byte-hash>)
pub const PUSH_32_BYTES: u8 = 0x20;

// ============================================================================
// PROTOCOL CONSTANTS
// ============================================================================

/// DER signature prefix (0x30 = SEQUENCE tag in DER encoding)
pub const DER_SIGNATURE_PREFIX: u8 = 0x30;

/// Maximum value for 1-byte varint encoding (< 0xfd)
pub const VARINT_1BYTE_MAX: u8 = 0xfc;

/// Prefix for 2-byte varint encoding (0xfd + 2 bytes)
pub const VARINT_2BYTE_PREFIX: u8 = 0xfd;

/// Prefix for 4-byte varint encoding (0xfe + 4 bytes)
pub const VARINT_4BYTE_PREFIX: u8 = 0xfe;

/// Prefix for 8-byte varint encoding (0xff + 8 bytes)
pub const VARINT_8BYTE_PREFIX: u8 = 0xff;

