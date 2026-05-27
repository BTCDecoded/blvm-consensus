//! Orange Paper **PROTOCOL.md §13.3.6** — **`F_SpecLockWitness`** formula anchor (**Phase 5** toolchain witness).
//!
//! The normative definition is **`**Formula** (**F_SpecLockWitness**):`** with **`$$ true $$`** in **PROTOCOL.md**. **`#[spec_locked("13.3", …)]`** cites § **13.3**, which subsumes **§13.3.6** per **`section_id_subsumes_formula_section`**.

use blvm_spec_lock::spec_locked;

#[spec_locked("13.3", "F_SpecLockWitness")]
pub(crate) fn spec_lock_formula_witness() -> bool {
    true
}
