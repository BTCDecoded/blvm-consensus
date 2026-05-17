# blvm-node RPC Integration for Testing Consensus

This document explains how to use blvm-node's RPC infrastructure to test blvm-consensus validation.

## Overview

blvm-node provides a complete Bitcoin node implementation with JSON-RPC 2.0 API that uses blvm-consensus for all consensus decisions. This allows testing consensus validation through RPC calls instead of direct function calls.

## Architecture

```
┌─────────────────┐
│  Test Suite     │
│  (blvm-         │
│   consensus)    │
└────────┬────────┘
         │ RPC Calls
         ▼
┌─────────────────┐
│  blvm-node      │
│  (RPC Server)   │
└────────┬────────┘
         │ Uses
         ▼
┌─────────────────┐
│ blvm-consensus  │
│  (Validation)   │
└─────────────────┘
```

## Starting blvm-node for Testing

### Method 1: In-Process Test Node

For unit tests **inside `blvm-node`**, construct a `Node` with `Node::new(data_dir, listen_addr, rpc_addr, Some(ProtocolVersion::Regtest))` (or `with_storage_config`) and start it the same way as the crate’s integration tests. See **`blvm-node/tests/node_tests.rs`** and related tests for a working pattern — do not use a nonexistent `config.network` field on **`NodeConfig`**.

### Method 2: Standalone Test Node

For integration tests, start the operator binary from the **`blvm`** crate (JSON-RPC on `--rpc-addr`; default `127.0.0.1:18332`):

```bash
# From workspace root (monorepo with `blvm/`):
cd blvm
cargo run -p blvm -- --network regtest --rpc-addr 127.0.0.1:18332
```

## RPC Methods for Testing Consensus

### Transaction Validation

Use `testmempoolaccept` to test transaction validation. See [RPC Reference](https://github.com/BTCDecoded/blvm-node/blob/main/docs/RPC_REFERENCE.md) for method details.

### Block Validation

Use `submitblock` to test block validation.

### Blockchain Queries

Use blockchain RPC methods (`getblock`, `getblockchaininfo`, `gettxoutsetinfo`) to query validated state.

## Testing Workflow

### 1. Start blvm-node

```bash
cd blvm
cargo run -p blvm -- --network regtest --rpc-addr 127.0.0.1:18332
```

### 2. Run Consensus Tests

```bash
cd blvm-consensus
cargo test
```

### 3. Verify Results

Tests will validate transactions and blocks via direct blvm-consensus API. For integration testing via RPC, use blvm-node's RPC server and compare results with direct validation.

## Benefits of RPC Testing

1. **Integration Testing**: Tests the full stack from RPC → node → blvm-consensus
2. **Real-World Scenarios**: Tests how blvm-consensus behaves in a real node context
3. **Protocol Validation**: Ensures RPC methods correctly use blvm-consensus
4. **Network Testing**: Can test across multiple nodes (future enhancement)

## Future Enhancements

1. **Multi-Node Testing**: Test consensus across multiple blvm-nodes
2. **Network Scenarios**: Test block/transaction propagation
3. **Performance Testing**: Benchmark consensus validation via RPC
4. **Differential Testing**: Compare blvm-node RPC results with Core RPC

## See Also

- [blvm-node RPC Reference](https://github.com/BTCDecoded/blvm-node/blob/main/docs/RPC_REFERENCE.md)

