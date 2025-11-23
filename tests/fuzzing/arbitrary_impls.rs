//! Arbitrary trait implementations for property-based testing (Phase 2.1)
//!
//! Enables comprehensive property-based testing with proptest by providing
//! Arbitrary implementations for consensus-critical types.
//!
//! This enhances property-based tests by allowing fuzzing of complete
//! Transaction, Block, and BlockHeader structures.

use proptest::prelude::*;
use bllvm_consensus::types::*;
use bllvm_consensus::{check_transaction, connect_block, UtxoSet};

impl Arbitrary for Transaction {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<u64>(), // version
            prop::collection::vec(
                (
                    any::<[u8; 32]>(), // prevout hash
                    any::<u64>(), // prevout index
                    prop::collection::vec(any::<u8>(), 0..100), // script_sig
                    any::<u64>(), // sequence
                ),
                0..10 // input count
            ),
            prop::collection::vec(
                (
                    any::<i64>(), // value
                    prop::collection::vec(any::<u8>(), 0..100), // script_pubkey
                ),
                0..10 // output count
            ),
            any::<u64>(), // lock_time
        )
            .prop_map(|(version, inputs, outputs, lock_time)| Transaction {
                version,
                inputs: inputs.into_iter().map(|(hash, index, script_sig, sequence)| {
                    TransactionInput {
                        prevout: OutPoint { hash, index },
                        script_sig,
                        sequence,
                    }
                }).collect(),
                outputs: outputs.into_iter().map(|(value, script_pubkey)| {
                    TransactionOutput {
                        value,
                        script_pubkey,
                    }
                }).collect(),
                lock_time,
            })
            .boxed()
    }
}

impl Arbitrary for BlockHeader {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<i32>(), // version
            any::<[u8; 32]>(), // prev_block_hash
            any::<[u8; 32]>(), // merkle_root
            any::<u64>(), // timestamp
            any::<u64>(), // bits
            any::<u64>(), // nonce
        )
            .prop_map(|(version, prev_block_hash, merkle_root, timestamp, bits, nonce)| {
                BlockHeader {
                    version,
                    prev_block_hash,
                    merkle_root,
                    timestamp,
                    bits,
                    nonce,
                }
            })
            .boxed()
    }
}

impl Arbitrary for Block {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<BlockHeader>(),
            prop::collection::vec(any::<Transaction>(), 0..100), // transactions
        )
            .prop_map(|(header, transactions)| Block {
                header,
            transactions: transactions.into(),
            })
            .boxed()
    }
}

impl Arbitrary for OutPoint {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<[u8; 32]>(), // hash
            any::<u64>(), // index
        )
            .prop_map(|(hash, index)| OutPoint { hash, index })
            .boxed()
    }
}

impl Arbitrary for UTXO {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (
            any::<i64>(), // value
            prop::collection::vec(any::<u8>(), 0..100), // script_pubkey
            any::<u64>(), // height
        )
            .prop_map(|(value, script_pubkey, height)| UTXO {
                value,
                script_pubkey,
                height,
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_arbitrary() {
        proptest!(|(tx: Transaction)| {
            // Should be able to generate arbitrary transactions
            let _result = check_transaction(&tx);
        });
    }

    #[test]
    fn test_block_header_arbitrary() {
        proptest!(|(header: BlockHeader)| {
            // Should be able to generate arbitrary block headers
            // Test that header structure is valid
            assert!(header.version >= -1 && header.version <= 2);
        });
    }

    #[test]
    fn test_block_arbitrary() {
        proptest!(|(block: Block)| {
            // Should be able to generate arbitrary blocks
            let utxo_set = UtxoSet::new();
            let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
            let _result = connect_block(&block, &witnesses, utxo_set, 0, None, crate::types::Network::Mainnet);
        });
    }
}

