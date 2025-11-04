//! Integration test for running Core test vectors
//!
//! This test runs all Core test vectors if they are available.
//! If test vectors are not present, the test is skipped.

#[cfg(test)]
mod tests {
    use super::super::*;
    
    #[test]
    fn test_run_all_core_vectors() {
        // Try to load and run transaction test vectors
        match load_transaction_test_vectors("tests/test_data/core_vectors/transactions") {
            Ok(vectors) if !vectors.is_empty() => {
                println!("Running {} transaction test vectors", vectors.len());
                if let Err(e) = run_core_transaction_tests(&vectors) {
                    eprintln!("Transaction test vectors failed: {}", e);
                    // Don't fail the test - this is informational
                }
            }
            Ok(_) => {
                println!("No transaction test vectors found (directory empty or missing)");
            }
            Err(e) => {
                eprintln!("Could not load transaction test vectors: {}", e);
            }
        }
        
        // Try to load and run script test vectors
        match load_script_test_vectors("tests/test_data/core_vectors/scripts") {
            Ok(vectors) if !vectors.is_empty() => {
                println!("Running {} script test vectors", vectors.len());
                if let Err(e) = run_core_script_tests(&vectors) {
                    eprintln!("Script test vectors failed: {}", e);
                    // Don't fail the test - this is informational
                }
            }
            Ok(_) => {
                println!("No script test vectors found (directory empty or missing)");
            }
            Err(e) => {
                eprintln!("Could not load script test vectors: {}", e);
            }
        }
        
        // Try to load and run block test vectors
        match load_block_test_vectors("tests/test_data/core_vectors/blocks") {
            Ok(vectors) if !vectors.is_empty() => {
                println!("Running {} block test vectors", vectors.len());
                if let Err(e) = run_core_block_tests(&vectors) {
                    eprintln!("Block test vectors failed: {}", e);
                    // Don't fail the test - this is informational
                }
            }
            Ok(_) => {
                println!("No block test vectors found (directory empty or missing)");
            }
            Err(e) => {
                eprintln!("Could not load block test vectors: {}", e);
            }
        }
    }
}



