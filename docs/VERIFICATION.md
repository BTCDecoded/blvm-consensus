# Formal Verification Documentation

## Mutation coverage

This is the spec-lock coverage figure. It is not a percentage of lines or of `#[spec_locked]` attributes.

A function is counted only when a semantic mutant of that function makes Z3 report UNSAT (the mutant contradicts the obligation). A tautology such as `result == true || result == false` does not count. Witness functions in `spec_witnesses.rs` are not the production functions.

| function | obligation | mutants caught / applied | encoding | scope limit |
|---|---|---|---|---|
| `check_transaction` | `F_NoDuplicateInputs`, `F_OutputSumBounded` | 5 / 5 (M1–M5) | signed 64-bit `bvadd` for the output sum; duplicate check is txid and vout | The witness proves a flag under `requires`. The mutant query is what fails if the `HashSet` check, the vout compare, the overflow `Err`, the `MAX_MONEY` bound, or the negative-output check is dropped. |
| `is_strict_der` | `F_StrictDERSoundness` (length, leading zero, high bit, tag, each as its own query) | 4 / 4 (M6–M9) | length is 32-bit; tag and R's first byte are 8-bit | No `#[spec_locked]` on the parser. Activation equality is `F_BIP66ActivationAt` / `F_BIP66ActivationBelow` on the witness, not a mutant of `check_bip66_network`. |
| `merkle_tree_from_hashes` | `F_MerkleMutationRejected` | 2 / 2 (M10–M11) | boolean mutation flag | The check is on the unpadded level. The odd-pad copy is not a mutation. IBD no longer skips the reject. |
| `check_proof_of_work` | `CheckProofOfWork` | 6 / 6 (M16–M21) | 256-bit compare at hash = target and at hash = target + 1; digest count, sign bit, zero target, and the helper call are read from the production source | SHA-256 compression is a named assumption. Equality is accepted. A hash one above the target, a negative compact encoding, a zero target, a single digest, or a skipped helper is rejected. |
| `validate_block_header` | `ValidBlockHeader` H01, H03, H04, H05, H06 | 9 / 9 (M22–M30) | version, timestamp, bits, and the future bound are 32-bit compares; H04 and H05 are read only inside the time-context branch | H02 is `check_bip90`. H08 is `validate_prev_block_hash`. A null merkle root is not a conjunct. Without a time context the function checks version, a non-zero timestamp, and non-zero bits. |
| `validate_prev_block_hash` | `ValidatePrevBlockHash` | 2 / 2 (M31–M32) | 256-bit equality | SHA-256 of the parent header is a named assumption. Equal hashes are accepted. A different hash is rejected. |
| `is_coinbase` | `IsCoinbase` | 5 / 5 (M33–M37) | one input, a 32-byte zero hash, and index `2^32 - 1` | Both cfg arms. The production arm calls `is_zero_hash`. Block uniqueness and the scriptSig length are other functions. |
| `check_coinbase_maturity` | `CheckCoinbaseMaturity` | 3 / 3 (M38–M40) | 64-bit compare against creation + R | R is 100. A non-coinbase spend passes. Equality at creation + R passes. One block earlier fails. |
| `get_block_subsidy` | `GetBlockSubsidy` | 3 / 3 (M41–M43) | 64-bit `bvudiv` and logical `bvlshr` | Heights 0, H, 2H, 3H, 33H, and 64H. An unguarded shift is taken mod 64, so k ≥ 64 stays zero only while the guard is present. `total_supply` is a different function. |
| `check_bip90` | `BIP90Check` | 9 / 9 (M44–M52) | version floors 2, 3, and 4 as 32-bit compares, each gated by its activation bit | The activation bit is an oracle. The numeric heights are `is_fork_active` and the table constants. |
| `is_fork_active` | activation compare | 14 / 14 (M53–M66) | 64-bit `>=` at the stored height; BIP30 is always active; CTV and CSFS are inactive at `u64::MAX` | The stored heights are fields. Historical BIP30 exemptions are `check_bip30`. |
| `validate_witness_program_length` | `ValidateWitnessProgramLength` | 4 / 4 (M67–M70) | v0 is exactly 20 or 32 bytes; v1 is exactly 32 bytes | An extra alternative in either arm is a separate mutant. |
| `is_sequence_disabled` | `IsSequenceDisabled` | 2 / 2 (M71–M72) | bit 31, mask `0x80000000` | The bit is set exactly when the mask is nonzero. |
| `extract_sequence_type_flag` | `ExtractSequenceTypeFlag` | 2 / 2 (M73–M74) | bit 22, mask `0x00400000` | The low 16 bits are `extract_sequence_locktime_value`. |
| `extract_sequence_locktime_value` | `ExtractSequenceLocktimeValue` | 2 / 2 (M75–M76) | low 16 bits, mask `0x0000ffff` | Bits 22 and 31 are the other two functions. |
| `get_locktime_type` | `GetLocktimeType` | 2 / 2 (M77–M78) | 32-bit `< 500_000_000` | Below the threshold is a block height. The timestamp twin has the same body. |
| `check_bip65` | `BIP65Check` | 2 / 2 (M79–M80) | types match and `tx_locktime >= stack_locktime` | The input sequence is not an argument of this function. |
| `is_push_opcode` | `IsPushOpcode` | 2 / 2 (M81–M82) | 8-bit `<= 0x60` | `0x00` through `0x60` is the push set. No `#[spec_locked]` attribute. |
| `is_op_success` | immediate success opcodes | 4 / 4 (M83–M86) | the script-section ranges, including `0x50`, `0x62`, and `0xbb..=0xfe` | `0x51` and `0xff` are not success opcodes. No `#[spec_locked]` attribute. |
| `weight_to_vsize` | `WeightToVSize` | 2 / 2 (M87–M88) | `ceil(weight / 4)` at weights 0, 1, 4, and 5 | Floor division fails the odd weight. |
| `check_bip54_tx_stripped_size` | `CheckBip54TxStrippedSize` | 3 / 3 (M89–M91) | coinbase, or stripped size not 64 | `calculate_transaction_size` is the stripped-size assumption. |
| `check_bip54_timewarp` | `BIP54TimewarpCheck` | 8 / 8 (M92–M99) | 64-bit compares; modulus `D_INTERVAL`; grace 7200 with saturating subtraction | Inactive and off-boundary blocks pass. A missing boundary fails. BIP94's 600-second rule is a different function. |
| `check_bip112` | CSV value compare | 2 / 2 (M100–M101) | type flags equal, then low 16 bits with `>=` | The disable bit and the version check are the caller. No `#[spec_locked]` attribute. |
| `calculate_transaction_weight_segwit` | `CalculateTransactionWeight` | 2 / 2 (M102–M103) | `3 * base_size + total_size` | Base size and total size are arguments. The body does not serialize the transaction. |
| `check_bip54_coinbase` | `BIP54CoinbaseCheck` | 5 / 5 (M104–M108) | lock time is `height - 1` by saturating subtraction; an empty input list fails; the first sequence is not `0xffffffff` | Activation is the caller. Height 0 keeps lock time 0. |
| `get_median_time_past` | `GetMedianTimePast` | 4 / 4 (M109–M112) | empty input is 0; the last 11 timestamps are sorted; the result is index `floor(n/2)` | `get_median_time_past_reversed` calls this function after reversing. It is not a second rule. |
| `is_bip54_active_at` | `IsBip54ActiveAt` | 3 / 3 (M113–M115) | an override height wins; otherwise the named network constant; active at and after that height | The stored mainnet constant is `u64::MAX`. The other network constants are the same value today, so swapping those names does not change the result. |
| `expand_target` | `ExpandTarget` | 8 / 8 (M116–M123) | exponent is `(bits >> 24) & 0xff`; mantissa is `bits & 0x007fffff`; domain is 3..=32; a zero mantissa is 0; otherwise `mantissa << 8*(exponent-3)` | Exponent 3 shifts by zero on both branches. The `shift >= 256` guard stays unreachable inside this domain. |
| `calculate_fee` | `CalculateFee` | 6 / 6 (M124–M129) | a coinbase fee is 0; otherwise input sum minus output sum, and a negative difference is rejected | The UTXO value is the query input. A missing prevout stays outside the formula. |
| `is_minimal_if_condition` | `IsMinimalIfCondition` | 4 / 4 (M130–M133) | empty is true; the one-byte true value is `0x01`; a longer string is false | `0x00` is not that true byte. |
| `cast_to_bool` | `CastToBool` | 4 / 4 (M134–M137) | empty and all-zero are false; a final `0x80` with a zero prefix is false; any other nonzero byte is true | Both cfg copies. An earlier `0x80` is an ordinary nonzero byte. |
| `is_witness_version_opcode` | witness version byte | 5 / 5 (M138–M142) | `OP_0`, or `OP_1` through `OP_16` | `OP_1NEGATE` and `OP_RESERVED` are ordinary opcodes. No `#[spec_locked]` attribute. `extract_witness_version` returns only v0 and v1. |
| `find_and_delete` | `FindAndDelete` | 5 / 5 (M143–M147) | an empty pattern and a longer pattern are no-ops; aligned copies are removed, including consecutive ones; a copy inside a push is kept | The opcode step is `script_get_op_advance`. |
| `total_supply` | `TotalSupply` | 4 / 4 (M148–M151) | each epoch contributes `(count) * (INITIAL_SUBSIDY >> k)`, including the block at the epoch start; 64 epochs | At `33 * H` the sum is `2,099,999,997,690,000`. |
| `is_null_dummy` | `IsNullDummy` | 10 / 10 (M152–M161) | an empty script is not a dummy; `OP_0` and a zero-length `PUSHDATA` are; only the first push counts; a truncated push and a non-push byte fail | Activation and the multisig scan stay in `check_bip147`. No `#[spec_locked]` attribute. |
| `validate_taproot_script` | `ValidateTaprootScript` | 4 / 4 (M162–M165) | length 34, first byte `0x51`, second byte `0x20` | A shorter script with the same prefix is rejected. |
| `extract_taproot_output_key` | `ExtractTaprootOutputKey` | 3 / 3 (M166–M168) | a valid script yields bytes `2..34`; any other script yields none | The slice starts after the two-byte prefix. |
| `derive_bip143_script_code` | `DeriveWitnessScriptCode` | 12 / 12 (M169–M180) | a 22-byte `OP_0 PUSH_20` program expands to the 25-byte P2PKH script; a 34-byte `OP_0 PUSH_32` program yields the witness script | The 25-byte body is `derive_bip143_script_code_p2wpkh`. |
| `append_stripped_codeseparators` | `SerializeScriptCode` | 6 / 6 (M181–M186) | `OP_CODESEPARATOR` (`0xab`) is removed at an opcode position; a direct push and each `PUSHDATA` keep that byte in the payload; every other opcode is kept | No `#[spec_locked]` attribute. |
| `strip_taproot_annex` | `StripTaprootAnnex` | 7 / 7 (M187–M193) | two or more elements and a last element starting `0x50` drops that element; the annex hash preimage is the compact size of the annex followed by the annex bytes | SHA-256 compression of that preimage is a named assumption. A one-element stack is not an annex. |
| `validate_segwit_witness_structure` | `ValidateSegWitWitnessStructure` | 3 / 3 (M194–M196) | every element is at most 520 bytes; an empty witness passes | A 520-byte element passes. A 521-byte element fails. |
| `check_bip66` | `BIP66Check` | 3 / 3 (M197–M199) | before activation every signature passes; after activation the signature passes only when `is_strict_der` does | The activation bit is an oracle. The DER predicate is the callee. |
| `is_pay_to_script_hash` | `IsP2SH` | 4 / 4 (M200–M203) | 23 bytes, `0xa9`, `0x14`, and a final `0x87` | A shorter script is rejected. |
| `sigop_count_push_opcode` | `F_SigopCountPushOpcode` | 2 / 2 (M204–M205) | an opcode at most `OP_16` (`0x60`), including `OP_1NEGATE` and `OP_RESERVED` | `OP_NOP` (`0x61`) is not a push. |
| `extract_redeem_script_from_scriptsig` | `ExtractRedeemScript` | 9 / 9 (M206–M214) | the result is the last instruction’s data; a direct push and each `PUSHDATA` supply that data; `OP_1NEGATE` through `OP_16` supply an empty string; a truncated push or an opcode above `OP_16` yields none | An empty script has no last instruction. No `#[spec_locked]` attribute. |
| `evaluate_sequence_locks` | `EvaluateSequenceLocks` | 7 / 7 (M215–M221) | each active minimum must be strictly below the block height or time; a negative minimum is disabled; zero is active | Equality at the minimum fails. Both constraints are required. |
| `is_bip30_repeat_block` | `BIP30Repeat` | 6 / 6 (M222–M227) | height 91,842 with its block hash, or height 91,880 with its block hash; every other pair is outside the set | Each height literal, each hash compare, and the default rejection are separate mutants. No `#[spec_locked]` attribute. |
| `is_witness_empty` | `IsWitnessEmpty` | 4 / 4 (M228–M231) | the stack has no items, or every item is empty | An empty item counts as empty. One nonempty byte, including `0x00`, counts as present. |
| `witness_stack_is_null` | `F_WitnessStackNullByLength` | 3 / 3 (M232–M234) | the stack is absent exactly when it has no items | One empty item is still present. No `#[spec_locked]` attribute. |
| `tx_has_nonempty_input_witness` | `HasNonEmptyInputWitness` | 4 / 4 (M235–M238) | some input stack is not `IsWitnessEmpty`; empty stacks and a missing witness vector are not | One nonempty stack among empty stacks is enough. A byte of `0x00` is nonempty. |
| `coinbase_reserved_nonce` | witness reserved value | 5 / 5 (M239–M243) | one stack of one 32-byte item; a missing stack, an empty stack, and any other shape yield none | A missing stack is not 32 zero bytes. No `#[spec_locked]` attribute. |
| `extract_witness_commitment` | `ExtractCommitment` | 6 / 6 (M244–M249) | `OP_RETURN`, push `0x24`, magic `0xaa21a9ed`, then the next 32 bytes | A 38-byte script passes. A shorter script, a bad opcode, a bad push, or a bad magic yields none. No `#[spec_locked]` attribute. |
| `last_witness_commitment_in_coinbase_outputs` | last commitment output | 4 / 4 (M250–M253) | the commitment of the last output that matches the witness pattern; none when no output matches | An earlier match does not win. A later non-matching output does not erase it. No `#[spec_locked]` attribute. |
| `validate_witness_commitment` | `ValidateWitnessCommitment` | 8 / 8 (M254–M261) | no commitment output passes; a commitment passes only when the reserved value is present and the stored word is the digest of `root \|\| nonce` | SHA-256 compression is a named assumption. A missing reserved value fails. |
| `block_witness_commitment_ok` | block witness commitment | 5 / 5 (M262–M266) | witness data requires a commitment output; that output then passes only when `validate_witness_commitment` does | No witness data and no commitment output passes. No `#[spec_locked]` attribute. |
| `witness_commitment_script` | witness commitment script | 8 / 8 (M267–M274) | `OP_RETURN`, push `0x24`, magic `0xaa21a9ed`, then the digest of `root \|\| nonce` | SHA-256 compression is a named assumption. No `#[spec_locked]` attribute. |
| `redeem_push_is_canonical` | nested redeem push | 4 / 4 (M275–M278) | a 22-byte or 34-byte redeem is a direct push of those bytes and nothing else | `OP_PUSHDATA1` of that redeem fails. No `#[spec_locked]` attribute. |
| `decode_locktime_value` | script-number operand | 7 / 7 (M279–M285) | at most 5 bytes, little-endian; the high bit of the last byte is the sign | Empty is 0. Negative zero, including `0x80`, is 0. `0x81` is −1. A 5-byte value passes. A 6-byte value yields none. No `#[spec_locked]` attribute. |
| `script_bool_bytes` | script result bytes | 3 / 3 (M286–M288) | false is an empty vector; true is the single byte `0x01` | `OP_EQUAL`, `OP_CHECKSIG`, and `OP_CHECKMULTISIG` push this encoding. No `#[spec_locked]` attribute. |
| `try_parse_taproot_schnorr_witness_sig` | `TryParseTaprootSchnorrWitnessSig` | 7 / 7 (M289–M295) | 64 bytes use sighash `0x00`; a 65-byte element keeps its suffix unless that byte is `0x00` | Any other length yields none. The signature bytes are the first 64. |
| `count_tapscript_sigops` | `CountTapscriptSigOps` | 11 / 11 (M296–M306) | opcode bytes `0xac`, `0xad`, and `0xba` outside a push payload | A direct push and each `PUSHDATA` skip that payload. `OP_CHECKMULTISIG` is not counted. An empty script is 0. |
| `tapscript_opcode_end` | truncated push is not an opcode | 14 / 14 (M307–M320) | a direct push or `PUSHDATA` whose header or payload does not fit | `OP_0` and every other opcode advance one byte. A fitting push ends after its payload. `PUSHDATA2` and `PUSHDATA4` lengths are little-endian. An end equal to the script length fits. No `#[spec_locked]` attribute. |
| `taproot_activation_height` | `TaprootActivationHeight` | 3 / 3 (M321–M323) | mainnet is 709,632, testnet is 2,011,968, and regtest is 0 | Each network is its own arm. |
| `opcode_position_at_byte` | codesep opcode index | 13 / 13 (M324–M336) | the 0-based opcode index of that byte | A direct push and each fitting `PUSHDATA` skip the payload. Any other byte is an opcode. A byte the walk does not land on is `0xffffffff`. No `#[spec_locked]` attribute. |
| `script_num_from_opcode` | `OP_1` through `OP_16` | 16 / 16 (M337–M352) | the number is `opcode - 0x50` | `OP_1` (`0x51`) is 1 and `OP_16` (`0x60`) is 16. Each opcode is its own arm. No `#[spec_locked]` attribute. |
| `is_zero_hash` | 32-byte zero hash | 2 / 2 (M353–M354) | every byte is `0` | One nonzero byte fails. The coinbase prevout uses this hash. No `#[spec_locked]` attribute. |
| `compute_script_merkle_root` | `ComputeScriptMerkleRoot` | 7 / 7 (M355–M361) | the leaf is `TapLeafHash(version, script)`; each proof hash is folded in order; siblings are the lexicographic min then max | Tagged SHA-256 is a named assumption. An empty proof is the leaf. |
| `validate_taproot_script_path_with_leaf_version` | `ValidateTaprootScriptPath` | 6 / 6 (M362–M367) | the result is true exactly when `compute_script_merkle_root` of the script, the proof, and the leaf version equals the given root | A different root is false. `validate_taproot_script_path` passes leaf version `0xc0` and is not a second rule. |
| `tap_leaf_hash` | `TapLeafHash` | 14 / 14 (M368–M381) | tag `TapLeaf`, then the version, the compact size of the script length, and the script | Both the default backend and the fallback. Tagged SHA-256 is a named assumption. A script of 253 bytes uses the `0xfd` form. No `#[spec_locked]` attribute. |
| `tap_branch_hash` | `TapBranchHash` | 10 / 10 (M382–M391) | tag `TapBranch`, then the left hash and the right hash | Both the default backend and the fallback. Sibling order stays in `compute_script_merkle_root`. Tagged SHA-256 is a named assumption. No `#[spec_locked]` attribute. |
| `tap_tweak_hash` | `ComputeTaprootTweak` | 5 / 5 (M392–M396) | tag `TapTweak`, then the internal key and the merkle root | Tagged SHA-256 is a named assumption. `compute_taproot_tweak` returns the tweaked output key. The fallback hashes the raw tag with one SHA-256. No `#[spec_locked]` attribute. |
| `bip34_magnitude` | BIP34 height magnitude | 6 / 6 (M397–M402) | little-endian bytes of the height; a trailing `0x00` when the high bit of the last byte is set | Height 0 has no magnitude bytes. Height 128 is `0x80 0x00`. The length byte and the trailing `0xff` stay in `encode_bip34_coinbase_script`. No `#[spec_locked]` attribute. |
| `script_contains_executable_opcode` | opcode at an opcode position | 11 / 11 (M403–M413) | a direct push and each `PUSHDATA` skip the payload; any other byte can match | `OP_0` is an opcode. A truncated `PUSHDATA1` header does not match. `PUSHDATA2` and `PUSHDATA4` lengths are little-endian. No `#[spec_locked]` attribute. |
| `bip54_deployment_mainnet` | BIP54 version-bits deployment | 9 / 9 (M414–M422) | signal bit 15, start time 0, timeout \(2^{64}-1\) | The testnet and regtest functions carry the same three constants. `bip54_deployment_for_network` only selects one of them. |
| `script_num_is_minimally_encoded` | minimal script number | 5 / 5 (M423–M427) | a trailing `0x00` or `0x80` is kept only when the previous byte’s high bit is set | Empty is minimal. A lone `0x80` is negative zero and is not minimal. No `#[spec_locked]` attribute. |
| `write_script_code_for_sighash` | legacy sighash script code | 5 / 5 (M428–M432) | compact size of the codesep-stripped script, then that script | A script with no `OP_CODESEPARATOR` is copied. One separator shortens the length. No `#[spec_locked]` attribute. |
| `calculate_base_script_flags_for_block` | base script flags | 11 / 11 (M433–M443) | P2SH `0x01`, DERSIG `0x04`, CLTV `0x200`, CSV `0x400`, NULLDUMMY `0x10`, each only while its fork is active | The word starts at 0. `STRICTENC` and `LOW_S` stay unset. Witness and taproot bits are the per-transaction function. |
| `script_sig_starts_with_height` | BIP34 height prefix | 10 / 10 (M444–M453) | height 0 is `OP_0`; heights 1..=16 are `OP_1`..=`OP_16`; a taller height is the length byte plus the magnitude | Bytes after that prefix are ignored. `OP_PUSHDATA` is not the prefix. No `#[spec_locked]` attribute. |
| `calculate_transaction_size` | `CalculateTransactionSize` | 3 / 3 (M454–M456) | the length of the witness-stripped serialization | Empty is 0. One extra byte is not that length. The witness serializer is a different function. |
| `block_header_hash` | `BlockHeaderHash` | 3 / 3 (M457–M459) | double SHA-256 of the 80-byte header serialization | SHA-256 compression is a named assumption. A single digest, another serializer, or an empty preimage is not that hash. |
| `stripped_transaction_size` | `\|Serialize(tx without witness)\|` | 7 / 7 (M460–M466) | the length of the witness-stripped serialization | The length is the production serializer. Version and locktime are 8 bytes. Each input's fixed fields are 40. Script bytes are counted. An empty input list with outputs adds the marker byte. A 252-byte length is one compact-size byte. No `#[spec_locked]` attribute. |
| `serialize_header` | 80-byte header serialization | 6 / 6 (M467–M472) | the bytes proof-of-work hashes | Those bytes are `serialize_block_header`. Version, previous hash, merkle root, time, bits, and nonce each have their own place. No `#[spec_locked]` attribute. |
| `encode_varint` | compact-size encoding | 7 / 7 (M473–M479) | the consensus variable-length integer | This is the mining encoder. The bytes are the serializer’s `encode_varint`. `0xfc` is one byte. `0xfd`, `0xfe`, and `0xff` keep their tags, and each payload is little-endian. The same body in `taproot.rs` is not a second rule. No `#[spec_locked]` attribute. |
| `write_varint_to_vec` | compact-size encoding | 7 / 7 (M480–M486) | the bytes appended to a sighash buffer | Those bytes are the serializer’s `encode_varint`. `0xfc` is one byte. `0xfd`, `0xfe`, and `0xff` keep their tags, and each payload is little-endian. `push_varint` and `update_varint` are the same four branches. No `#[spec_locked]` attribute. |
| `compact_size_len` | compact-size length | 7 / 7 (M487–M493) | the number of bytes in that encoding | The length is `encode_varint(n).len()`. Below `0xfd` is 1. Through `0xffff` is 3. Through `0xffffffff` is 5. A larger value is 9. The test helper in `transaction.rs` and the copy in `blvm-secp256k1` are the same four widths. No `#[spec_locked]` attribute. |
| `decode_varint` | compact-size decoding | 16 / 16 (M494–M509) | the value and width of a canonical compact size, and rejection of a bad one | The bytes are the serializer’s `encode_varint`. A byte below `0xfd` is that value and width 1. The `0xfd`, `0xfe`, and `0xff` forms are little-endian and have widths 3, 5, and 9. An empty buffer, a truncated payload, and a non-minimal value are rejected. No `#[spec_locked]` attribute. |
| `deserialize_block_header` | header round-trip | 8 / 8 (M510–M517) | the header parsed from the 80-byte serialization | Those bytes are `serialize_block_header`. Version is a signed little-endian `i32`. The previous hash and the merkle root are the next two 32-byte fields. Time, bits, and nonce are little-endian. Fewer than 80 bytes is rejected. No `#[spec_locked]` attribute. |
| `encode_locktime_value` | locktime encoding | 4 / 4 (M518–M521) | `DecodeLocktime(EncodeLocktime(lt)) = lt` | The decoder is `decode_locktime_value`. Zero is the byte `0x00`. `0x0201` keeps its low byte first, and the high bytes are not dropped. `0x80` keeps the trailing `0x00` so the value stays positive. No `#[spec_locked]` attribute. |
| `coinbase_or_empty_short_circuit` | coinbase input short-circuit | 4 / 4 (M522–M525) | an empty input list is invalid, and a coinbase fee is 0 | A coinbase is `(valid, 0)`. An empty input list is `(invalid, 0)`. A transaction with an ordinary input is not decided here. `is_coinbase` is the coinbase predicate. No `#[spec_locked]` attribute. |
| `sum_output_values` | output value sum | 4 / 4 (M526–M529) | the sum of the non-negative output values | The sum starts at 0. Each value is added. A negative value is rejected. `i64::MAX + 1` is rejected. The `MAX_MONEY` cap stays in `check_transaction`. No `#[spec_locked]` attribute. |
| `bip30_lookup_skippable` | BIP30 lookup window | 6 / 6 (M530–M535) | the duplicate lookup is skipped only after the known height-in-coinbase block, before height 1,983,702, and only on that chain | A chain whose activation hash does not match is checked. The activation block itself is checked. Height 1,983,702 is checked. Mainnet starts after 227,931. Testnet starts after 21,111. Regtest, signet, and testnet4 are checked. No `#[spec_locked]` attribute. |
| `op_mul_disabled` | disabled `OP_MUL` | 1 / 1 (M536) | `OP_MUL` fails unconditionally | A successful result accepts the opcode. `OP_DIV`, `OP_MOD`, `OP_LSHIFT`, and `OP_RSHIFT` are the same failure. No `#[spec_locked]` attribute. |
| `deserialize_transaction_with_witness` | transaction round trip | 14 / 14 (M537–M550) | `Deserialize(Serialize(tx)) = tx`, and the witness form when the stack count is the input count | The bytes are `serialize_transaction` or `serialize_transaction_with_witness`. Version, locktime, prevout, script, sequence, output value, and the witness item are little-endian where the serializer writes them. An empty input list with outputs keeps the marker byte. Fewer than four bytes is rejected. No `#[spec_locked]` attribute. |

**93 / 224.** Ninety-three functions carry a mutation-validated obligation. The denominator is 181 production `#[spec_locked]` anchors outside `spec_witnesses.rs`, plus `is_strict_der`, `is_push_opcode`, `is_op_success`, `check_bip112`, `is_witness_version_opcode`, `is_null_dummy`, `append_stripped_codeseparators`, `extract_redeem_script_from_scriptsig`, `is_bip30_repeat_block`, `witness_stack_is_null`, `coinbase_reserved_nonce`, `extract_witness_commitment`, `last_witness_commitment_in_coinbase_outputs`, `block_witness_commitment_ok`, `witness_commitment_script`, `redeem_push_is_canonical`, `decode_locktime_value`, `script_bool_bytes`, `tapscript_opcode_end`, `opcode_position_at_byte`, `script_num_from_opcode`, `is_zero_hash`, `tap_leaf_hash`, `tap_branch_hash`, `tap_tweak_hash`, `bip34_magnitude`, `script_contains_executable_opcode`, `script_num_is_minimally_encoded`, `write_script_code_for_sighash`, `script_sig_starts_with_height`, `stripped_transaction_size`, `serialize_header`, `encode_varint`, `write_varint_to_vec`, `compact_size_len`, `decode_varint`, `deserialize_block_header`, `encode_locktime_value`, `coinbase_or_empty_short_circuit`, `sum_output_values`, `bip30_lookup_skippable`, `op_mul_disabled`, and `deserialize_transaction_with_witness`, which have no anchor. The other 131 are test coverage only, or their Z3 obligation is not mutation-validated. They are not formally verified by this gate.

`checked_add`, `checked_sub`, and `checked_mul` are signed 64-bit bitvectors with an overflow predicate. A literal `<<` in `0..64` wraps at 64 bits. Right shifts, masks, and the other arithmetic are still unbounded `Int`. `checked_map_bvadd_overflows_at_i64_max` is SAT at `x = i64::MAX` because `x + 1` overflows. That query stays SAT.

The table regenerated by the harness is `blvm-spec-lock/docs/MUTATION_COVERAGE.md`. Consensus mutants caught there: 11 / 11.

## Overview

Implements checks for Bitcoin consensus rules using mathematical specifications, Z3 on annotated functions, and property-based testing. The mutation-coverage section above is the coverage figure. The stack below is what CI runs, which is a wider check than that figure.

## Verification Stack

Verification approach follows: **"Rust + Tests + Math Specs = Source of Truth"**

### Layer 1: Empirical Testing (Required, Must Pass)
- **Unit tests**: Comprehensive test coverage for all consensus functions
- **Property-based tests**: Randomized testing with `proptest` to discover edge cases
- **Integration tests**: Cross-system validation between consensus components

### Layer 2: Symbolic Verification (CI verify job)
- **blvm-spec-lock**: Z3 on `#[spec_locked]` functions against the Orange Paper. A passing verify job is not the mutation gate above.
- **Mathematical specifications**: Orange Paper sections named on those attributes
- **Traceability**: Annotated functions name an Orange Paper section. Witnesses in `spec_witnesses.rs` are not the production functions.

### Layer 3: CI Enforcement
- **Automated verification**: Tests and `cargo-spec-lock verify` run on pull requests. A commit message containing `[skip ci]` or `[skip_verify]`, or a workflow dispatch with `skip_tests`, skips that work.
- **OpenTimestamps**: not a step in `.github/workflows/ci.yml`
- **Coverage figure**: the mutation-coverage table above

## Mathematical Specifications

### Chain Selection (`src/reorganization.rs`)

**Mathematical Specification:**
```
∀ chains C₁, C₂: work(C₁) > work(C₂) ⇒ select(C₁)
```

**Invariants:**
- Selected chain has maximum cumulative work
- Work calculation is deterministic
- Empty chains are rejected
- Chain work is always non-negative

**Verified Functions:**
- `should_reorganize`: Proves longest chain selection
- `calculate_chain_work`: Verifies cumulative work calculation
- `expand_target`: Handles difficulty target edge cases

### Block Subsidy (`src/economic.rs`)

**Mathematical Specification:**
```
∀ h ∈ ℕ: subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 else 0
```

**Invariants:**
- Subsidy halves every 210,000 blocks
- After 64 halvings, subsidy becomes 0
- Subsidy is always non-negative
- Total supply approaches 21M BTC asymptotically

**Verified Functions:**
- `get_block_subsidy`: Verifies halving schedule
- `total_supply`: Proves monotonic increase
- `validate_supply_limit`: Ensures supply cap compliance

### Proof of Work (`src/pow.rs`)

**Mathematical Specification:**
```
∀ header H: CheckProofOfWork(H) = SHA256(SHA256(H)) < ExpandTarget(H.bits)
```

**Target Compression/Expansion (reference GetCompact/SetCompact):**
```
∀ bits ∈ [0x03000000, 0x1d00ffff]:
  Let expanded = expand_target(bits)
  Let compressed = compress_target(expanded)
  Let re_expanded = expand_target(compressed)
  
  Then:
  - re_expanded ≤ expanded (compression truncates, never increases)
  - re_expanded.0[2] = expanded.0[2] ∧ re_expanded.0[3] = expanded.0[3]
    (significant bits preserved exactly)
  - Precision loss in words 0, 1 is acceptable (compact format limitation)
```

**Invariants:**
- Hash must be less than target for valid proof of work
- Target expansion handles edge cases correctly
- Target compression preserves significant bits (words 2, 3) exactly
- Target compression may lose precision in lower bits (words 0, 1)
- Difficulty adjustment respects bounds [0.25, 4.0]
- Work calculation is deterministic

**Verified Functions:**
- `check_proof_of_work`: Verifies hash < target
- `expand_target`: Handles compact target representation
- `compress_target`: Implements reference GetCompact() exactly
- `expand_target`/`compress_target`: **Formally verified** via spec-lock - proves significant bits preserved
- `get_next_work_required`: Respects difficulty bounds

### Transaction Validation (`src/transaction.rs`)

**Mathematical Specification:**
```
∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟺ 
  (|tx.inputs| > 0 ∧ |tx.outputs| > 0 ∧ 
   ∀o ∈ tx.outputs: 0 ≤ o.value ≤ M_max ∧
   |tx.inputs| ≤ M_max_inputs ∧ |tx.outputs| ≤ M_max_outputs ∧
   |tx| ≤ M_max_tx_size)
```

**Invariants:**
- Valid transactions have non-empty inputs and outputs
- Output values are bounded [0, MAX_MONEY]
- Input/output counts respect limits
- Transaction size respects limits
- Coinbase transactions have special validation rules

**Verified Functions:**
- `check_transaction`: Validates structure rules
- `check_tx_inputs`: Handles coinbase correctly
- `is_coinbase`: Correctly identifies coinbase transactions

### Block Connection (`src/block.rs`)

**Mathematical Specification:**
```
∀ block B, UTXO set US, height h: ConnectBlock(B, US, h) = (valid, US') ⟺
  (ValidateHeader(B.header) ∧ 
   ∀ tx ∈ B.transactions: CheckTransaction(tx) ∧ CheckTxInputs(tx, US, h) ∧
   VerifyScripts(tx, US) ∧
   CoinbaseOutput ≤ TotalFees + GetBlockSubsidy(h) ∧
   US' = ApplyTransactions(B.transactions, US))
```

**Invariants:**
- Valid blocks have valid headers and transactions
- UTXO set consistency is preserved
- Coinbase output respects economic rules
- Transaction application is atomic

**Verified Functions:**
- `connect_block`: Validates complete block
- `apply_transaction`: Preserves UTXO consistency
- `calculate_tx_id`: Deterministic transaction identification

## Verification Tools

### blvm-spec-lock

**Purpose**: Z3-based formal verification against Orange Paper specifications
**Usage**: `cargo spec-lock verify --crate-path .`
**Coverage**: All functions with `#[spec_locked("section")]` annotations
**Traceability**: Links implementation to Orange Paper sections

**Example:**
```rust
#[spec_locked("5.2")]
pub fn expand_target(bits: u32) -> [u32; 4] {
    // Implementation verified against Orange Paper section 5.2
}
```

### Proptest Property Testing

**Purpose**: Randomized testing to discover edge cases
**Usage**: `cargo test` (runs automatically)
**Coverage**: All `proptest!` macros
**Strategy**: Generates random inputs within specified bounds

**Example:**
```rust
proptest! {
    #[test]
    fn prop_function_invariant(input in strategy) {
        let result = function_under_test(input);
        prop_assert!(result.property_holds());
    }
}
```

### OpenTimestamps

Not a step in `.github/workflows/ci.yml`. That workflow does not run `ots stamp`.

## CI Integration

### Verification Workflow

The **Verify** job in **`blvm-consensus/.github/workflows/ci.yml`** enforces **`check-drift`** (with spec paths) and a single **`cargo-spec-lock verify`** with **`--spec-path`** — **merged `F_*` registry** (inside **`verify`**) plus **`#[spec_locked]`** Rust rows; **`spec_lock_verify.json`** via **`--json-out`**. MIRI smoke steps run in that same CI file (non-blocking when unavailable).

See also **[SPEC_LOCK_DEPENDENCY.md](../SPEC_LOCK_DEPENDENCY.md)** for umbrella monorepo workflows that **mirror** **blvm-consensus** / **blvm-node** / **blvm-protocol** **`ci.yml`** checks but **do not** replace those authoritative gates.

### Local Development

**Run all tests:**
```bash
cargo test --all-features
```

**Run spec-lock verification** (needs sibling **`../blvm-spec`** or set **`SPEC_LOCK_SPEC_PATH`**, same as CI):

```bash
cargo spec-lock verify --crate-path . \
  --spec-path ../blvm-spec/PROTOCOL.md ../blvm-spec/ARCHITECTURE.md \
  --timeout 120 --json-out spec_lock_verify.json
```

**Run specific verification:**
```bash
cargo test --test consensus_property_tests
cargo test --test orange_paper_property_tests
```

## Adding New Verification

### Step 1: Add Mathematical Specification

Document the mathematical invariant as a comment:

```rust
/// Mathematical Specification:
/// ∀ input I: property(I) ⟺ invariant_holds(I)
/// 
/// Invariants:
/// - Invariant 1: Description
/// - Invariant 2: Description
```

### Step 2: Add Spec-Lock Annotation

Add `#[spec_locked("section")]` to link the function to the Orange Paper:

```rust
use blvm_spec_lock::spec_locked;

#[spec_locked("5.2")]
pub fn function_under_test(input: InputType) -> ResultType {
    // Implementation verified against Orange Paper section 5.2
}
```

### Step 3: Add Property Tests

Create randomized tests with proptest:

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_function_invariant(
            input in strategy
        ) {
            let result = function_under_test(input);
            prop_assert!(result.property_holds());
        }
    }
}
```

### Step 4: Update CI

The CI workflow automatically picks up new verification. No changes needed.

## Governance Integration

### Multi-Level Enforcement

The verification system integrates with BTCDecoded governance at multiple levels:

1. **Technical Enforcement (CI/CD)**: Automated verification blocks merge
2. **Governance Policy (Layer System)**: Layer 2 requires 6-of-7 signatures
3. **Governance App Enforcement**: Validates CI passed before signatures
4. **Cross-Layer Separation**: Prevents governance capture of consensus

### Ostrom Principles Compliance

- **Monitoring (#4)**: Automated CI verification enforces rules
- **Proportional Equivalence (#2)**: Constitutional changes require maximum consensus
- **Graduated Sanctions (#5)**: PRs without verification cannot progress
- **Clearly Defined Boundaries (#1)**: Separation preserved via module boundaries

## Security Considerations

### Attack Vectors Mitigated

1. **Consensus Rule Violations**: Spec-lock verification prevents invalid consensus
2. **Edge Case Exploits**: Property tests discover boundary conditions
3. **Implementation Bugs**: Mathematical specs document correct behavior
4. **Governance Capture**: Multi-level enforcement prevents override

### Defense Mechanisms

1. **Mathematical Proofs**: Spec-lock verifies invariants against Orange Paper
2. **Randomized Testing**: Proptest discovers unexpected behavior
3. **Verify artifact**: when the verify job runs, it can upload `spec_lock_verify.json`
4. **CI skip**: a commit message containing `[skip_verify]` or `[skip ci]`, or a workflow dispatch with `skip_tests`, skips that job

## Performance Considerations

### Verification Bounds

- **Spec-lock**: Verifies functions against Orange Paper specifications
- **Property tests**: Limited input ranges to prevent timeouts
- **CI timeouts**: 5-minute limit per verification step

### Optimization Strategies

1. **Bounded verification**: Focus on critical paths
2. **Parallel execution**: CI runs tests in parallel
3. **Incremental verification**: Only verify changed functions
4. **Caching**: CI caches dependencies and artifacts

## Future Enhancements

### Planned Improvements

1. **Expanded Coverage**: Add verification to all consensus functions
2. **Cross-Layer Verification**: Verify blvm-protocol and blvm-node
3. **Performance Optimization**: Reduce verification time
4. **Documentation**: Add more mathematical specifications

### Optional Integrations

1. **Coq Integration**: Via `coq-of-rust` (non-blocking)
2. **Additional Tools**: ESBMC, CBMC for redundancy
3. **Formal Specs**: TLA+ or Alloy specifications
4. **Interactive Proofs**: Manual verification assistance

## Troubleshooting

### Common Issues

**Spec-lock verification fails:**
- Check `#[spec_locked]` annotation matches Orange Paper section
- Verify function preconditions and postconditions
- Ensure function is deterministic

**Property test fails:**
- Check input strategy bounds
- Add filtering with `prop_filter!`
- Verify property is actually true

**CI verification fails:**
- Check all tests pass locally
- Verify spec-lock passes
- Check OpenTimestamps installation

### Debug Commands

```bash
# Debug property test
RUST_LOG=debug cargo test prop_function_invariant

# Check verification artifacts
ots verify proof-artifacts.tar.gz.ots
```

## References

- [blvm-spec-lock](https://github.com/BTCDecoded/blvm-spec-lock)
- [Proptest Documentation](https://docs.rs/proptest/)
- [OpenTimestamps Protocol](https://opentimestamps.org/)
- [BTCDecoded Governance](https://github.com/BTCDecoded/governance/blob/main/GOVERNANCE.md)
- [Orange Paper (blvm-spec)](https://github.com/BTCDecoded/blvm-spec)











