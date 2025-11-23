# BIP66 and BIP147 Integration Plan

**Purpose**: Integrate BIP66 (Strict DER) and BIP147 (NULLDUMMY) into script validation  
**Status**: Implementation Plan  
**Date**: 2025-01-XX

---

## Current Status

### BIP66 (Strict DER)

**Status**: ✅ Function exists, ⚠️ Not integrated

**Function**: `bllvm-consensus/src/bip_validation.rs::check_bip66()`
- ✅ Checks strict DER encoding
- ✅ Handles activation heights
- ✅ Returns `Result<bool>`

**Integration Point**: `bllvm-consensus/src/script.rs::verify_signature()`
- ⚠️ Currently uses `Signature::from_der()` which may accept non-strict DER
- ⚠️ No BIP66 check before signature parsing
- ⚠️ No height/network parameters available

### BIP147 (NULLDUMMY)

**Status**: ✅ Function exists, ⚠️ Not integrated, ⚠️ Simplified implementation, 🚨 **OP_CHECKMULTISIG NOT IMPLEMENTED**

**Function**: `bllvm-consensus/src/bip_validation.rs::check_bip147()`
- ✅ Checks NULLDUMMY requirement
- ✅ Handles activation heights
- ⚠️ Simplified implementation (needs improvement)
- ✅ Returns `Result<bool>`

**Integration Point**: `bllvm-consensus/src/script.rs::execute_opcode()` (OP_CHECKMULTISIG)
- 🚨 **CRITICAL**: OP_CHECKMULTISIG (0xae) is **NOT IMPLEMENTED**
- Currently falls through to default case: `_ => Ok(false)`
- ⚠️ No BIP147 check before OP_CHECKMULTISIG execution
- ⚠️ No height/network parameters available

**Prerequisite**: OP_CHECKMULTISIG must be implemented before BIP147 can be integrated.

---

## Integration Requirements

### 1. BIP66 Integration

**Location**: `bllvm-consensus/src/script.rs`

**Changes Required**:

1. **Modify `verify_signature()` signature**:
   ```rust
   // Current:
   fn verify_signature<C: Context + Verification>(
       secp: &Secp256k1<C>,
       pubkey_bytes: &[u8],
       signature_bytes: &[u8],
       sighash: &[u8; 32],
       _flags: u32,
   ) -> bool
   
   // New:
   fn verify_signature<C: Context + Verification>(
       secp: &Secp256k1<C>,
       pubkey_bytes: &[u8],
       signature_bytes: &[u8],
       sighash: &[u8; 32],
       flags: u32,
       height: Natural,                    // ADD
       network: crate::types::Network,      // ADD
   ) -> Result<bool>                       // CHANGE: Return Result
   ```

2. **Add BIP66 check before signature parsing**:
   ```rust
   // Check BIP66 if flag is set
   if flags & SCRIPT_VERIFY_DERSIG != 0 {
       if !crate::bip_validation::check_bip66(signature_bytes, height, network)? {
           return Ok(false);
       }
   }
   ```

3. **Update all call sites** to pass height and network:
   - `execute_opcode_with_context_full()` - OP_CHECKSIG, OP_CHECKSIGVERIFY
   - `batch_verify_signatures()` - Needs height/network per signature
   - Any other call sites

4. **Update function calls** to handle `Result<bool>`:
   - Change `verify_signature()` calls to handle errors
   - Propagate errors appropriately

### 2. BIP147 Integration

**Location**: `bllvm-consensus/src/script.rs`

**Changes Required**:

1. **Find OP_CHECKMULTISIG implementation**:
   - Search for opcode `0xae` in `execute_opcode()` or `execute_opcode_with_context_full()`
   - Currently in Kani proof, need to find actual implementation

2. **Add BIP147 check before OP_CHECKMULTISIG execution**:
   ```rust
   // In OP_CHECKMULTISIG handler (0xae):
   // Check BIP147 if flag is set
   if flags & SCRIPT_VERIFY_NULLDUMMY != 0 {
       // Get script_sig and script_pubkey from context
       // Check dummy element is empty (OP_0)
       if !crate::bip_validation::check_bip147(
           script_sig,
           script_pubkey,
           height,
           network,
       )? {
           return Ok(false);
       }
   }
   ```

3. **Improve BIP147 implementation**:
   - Current implementation is simplified
   - Need to properly parse scriptSig to find dummy element
   - Dummy element is first consumed by OP_CHECKMULTISIG
   - Must be empty (OP_0) after activation

4. **Update function signature** to include height/network:
   - `execute_opcode_with_context_full()` already has height
   - Need to add network parameter
   - Pass to BIP147 check

---

## Implementation Steps

### Step 1: Verify BIP Functions (30 min)

- [ ] Review `check_bip66()` implementation
- [ ] Review `check_bip147()` implementation
- [ ] Test with known strict/non-strict DER signatures
- [ ] Test with known NULLDUMMY cases

### Step 2: Integrate BIP66 (2 hours)

- [ ] Modify `verify_signature()` signature
- [ ] Add BIP66 check before signature parsing
- [ ] Update all call sites to pass height/network
- [ ] Update error handling
- [ ] Test integration

### Step 3: Integrate BIP147 (2 hours)

- [ ] Find OP_CHECKMULTISIG implementation
- [ ] Add BIP147 check before execution
- [ ] Improve BIP147 implementation (if needed)
- [ ] Update function signatures
- [ ] Test integration

### Step 4: Add Integration Tests (2 hours)

- [ ] Add `test_connect_block_rejects_bip66_violation`
- [ ] Add `test_connect_block_rejects_bip147_violation`
- [ ] Add `test_bip66_strict_der_enforcement`
- [ ] Add `test_bip147_null_dummy_enforcement`
- [ ] Follow pattern from BIP30/BIP34/BIP90 tests

### Step 5: Add to 6-Layer Verification (1 hour)

- [ ] Add smoke tests
- [ ] Add Kani proofs
- [ ] Add debug assertions
- [ ] Add compile-time checks

### Step 6: Test with Historical Blocks (1 hour)

- [ ] Test BIP66 with blocks after activation (363,724+)
- [ ] Test BIP147 with blocks after activation (481,824+)
- [ ] Verify rejection of violating blocks

---

## Code Changes Summary

### Files to Modify

1. `bllvm-consensus/src/script.rs`
   - Modify `verify_signature()` signature
   - Add BIP66 check
   - Add BIP147 check in OP_CHECKMULTISIG
   - Update call sites

2. `bllvm-consensus/src/bip_validation.rs`
   - Improve `check_bip147()` implementation (if needed)

3. `bllvm-consensus/tests/integration/bip_enforcement_tests.rs`
   - Add BIP66 violation test
   - Add BIP147 violation test

4. `bllvm-consensus/tests/integration/bip_integration_smoke_tests.rs`
   - Add BIP66 smoke test
   - Add BIP147 smoke test

5. `bllvm-consensus/tests/integration/bip_integration_kani_proofs.rs`
   - Add BIP66 Kani proof
   - Add BIP147 Kani proof

---

## Testing Strategy

### Unit Tests

- Test BIP66 with strict DER signatures
- Test BIP66 with non-strict DER signatures
- Test BIP147 with empty dummy
- Test BIP147 with non-empty dummy

### Integration Tests

- Test BIP66 violation rejection in `connect_block()`
- Test BIP147 violation rejection in `connect_block()`
- Test with historical blocks

### Historical Block Tests

- Test BIP66 with blocks after activation (363,724+)
- Test BIP147 with blocks after activation (481,824+)
- Verify blocks are accepted/rejected correctly

---

## Risk Assessment

### High Risk Areas

1. **Signature Verification**: Changing `verify_signature()` affects all signature checks
2. **OP_CHECKMULTISIG**: Changing multisig validation affects all multisig scripts
3. **Error Handling**: Changing return types affects error propagation

### Mitigation

1. **Comprehensive Testing**: Test all signature verification paths
2. **Incremental Changes**: Make changes incrementally, test after each
3. **Code Review**: Review all changes carefully
4. **Regression Testing**: Run full test suite after changes

---

## Success Criteria

- [ ] BIP66 integrated into signature verification
- [ ] BIP147 integrated into OP_CHECKMULTISIG
- [ ] All existing tests pass
- [ ] New integration tests pass
- [ ] Historical block tests pass
- [ ] 6-layer verification complete
- [ ] No regressions

---

## Estimated Time

- Step 1: 30 min
- Step 2: 2 hours
- Step 3: 2 hours
- Step 4: 2 hours
- Step 5: 1 hour
- Step 6: 1 hour

**Total**: 8.5 hours

---

## Notes

- This is consensus-critical code - review carefully
- Follow pattern from BIP30/BIP34/BIP90 integration
- Ensure backward compatibility where possible
- Document all changes

---

**Status**: Ready for Implementation  
**Priority**: High  
**Risk**: Medium (consensus-critical)

