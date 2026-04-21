//! Unified fork activation: table and trait for "is fork X active at height H?".
//!
//! Used by block validation to query activation from a single precomputed table
//! (built by the node from constants, version-bits, and config overrides)
//! without passing per-BIP parameters.

use crate::constants::*;
use crate::types::{ForkId, Network};

/// Trait for querying fork activation at a given height.
///
/// Implemented by `ForkActivationTable` and by `BlockValidationContext` (delegating to its table).
/// Allows validation code to ask "is fork F active?" without depending on a concrete context type.
pub trait IsForkActive {
    /// Returns true if the fork is active at the given block height.
    fn is_fork_active(&self, fork: ForkId, height: u64) -> bool;
}

/// Precomputed activation heights for all built-in forks.
///
/// Fixed-size storage: one field per fork (no HashMap). Built by the node from
/// chain params, version-bits (e.g. BIP54), and config overrides. Consensus only reads.
#[derive(Debug, Clone)]
pub struct ForkActivationTable {
    /// BIP30: active when height <= this (deactivation fork).
    pub bip30_deactivation: u64,
    /// Activation heights (active when height >= value; u64::MAX = never active).
    pub bip16: u64,
    pub bip34: u64,
    pub bip66: u64,
    pub bip65: u64,
    pub bip112: u64,
    pub bip147: u64,
    pub segwit: u64,
    pub taproot: u64,
    pub ctv: u64,
    pub csfs: u64,
    pub bip54: u64,
}

impl IsForkActive for ForkActivationTable {
    #[inline]
    fn is_fork_active(&self, fork: ForkId, height: u64) -> bool {
        match fork {
            ForkId::Bip30 => height <= self.bip30_deactivation,
            ForkId::Bip16 => height >= self.bip16,
            ForkId::Bip34 => height >= self.bip34,
            ForkId::Bip66 => height >= self.bip66,
            ForkId::Bip65 => height >= self.bip65,
            ForkId::Bip112 => height >= self.bip112,
            ForkId::Bip147 => height >= self.bip147,
            ForkId::SegWit => height >= self.segwit,
            ForkId::Taproot => height >= self.taproot,
            ForkId::Ctv => self.ctv != u64::MAX && height >= self.ctv,
            ForkId::Csfs => self.csfs != u64::MAX && height >= self.csfs,
            ForkId::Bip54 => height >= self.bip54,
        }
    }
}

impl ForkActivationTable {
    /// Build table from network and constants. BIP54 uses per-network constant (u64::MAX by default).
    pub fn from_network(network: Network) -> Self {
        Self::from_network_and_bip54_override(network, None)
    }

    /// Build table from network and optional BIP54 activation override (e.g. from version bits).
    pub fn from_network_and_bip54_override(
        network: Network,
        bip54_activation_override: Option<u64>,
    ) -> Self {
        let (
            bip30_deactivation,
            bip16,
            bip34,
            bip66,
            bip65,
            bip112,
            bip147,
            segwit,
            taproot,
            ctv,
            csfs,
        ) = match network {
            Network::Mainnet => (
                BIP30_DEACTIVATION_MAINNET,
                BIP16_P2SH_ACTIVATION_MAINNET,
                BIP34_ACTIVATION_MAINNET,
                BIP66_ACTIVATION_MAINNET,
                BIP65_ACTIVATION_MAINNET,
                BIP112_CSV_ACTIVATION_MAINNET,
                BIP147_ACTIVATION_MAINNET,
                SEGWIT_ACTIVATION_MAINNET,
                TAPROOT_ACTIVATION_MAINNET,
                if CTV_ACTIVATION_MAINNET == 0 {
                    u64::MAX
                } else {
                    CTV_ACTIVATION_MAINNET
                },
                if CSFS_ACTIVATION_MAINNET == 0 {
                    u64::MAX
                } else {
                    CSFS_ACTIVATION_MAINNET
                },
            ),
            Network::Testnet => (
                BIP30_DEACTIVATION_TESTNET,
                BIP16_P2SH_ACTIVATION_TESTNET,
                BIP34_ACTIVATION_TESTNET,
                BIP66_ACTIVATION_TESTNET,
                BIP65_ACTIVATION_TESTNET,
                BIP112_CSV_ACTIVATION_TESTNET,
                BIP147_ACTIVATION_TESTNET,
                SEGWIT_ACTIVATION_TESTNET,
                TAPROOT_ACTIVATION_TESTNET,
                if CTV_ACTIVATION_TESTNET == 0 {
                    u64::MAX
                } else {
                    CTV_ACTIVATION_TESTNET
                },
                if CSFS_ACTIVATION_TESTNET == 0 {
                    u64::MAX
                } else {
                    CSFS_ACTIVATION_TESTNET
                },
            ),
            Network::Regtest => (
                BIP30_DEACTIVATION_REGTEST,
                BIP16_P2SH_ACTIVATION_REGTEST,
                BIP34_ACTIVATION_REGTEST,
                BIP66_ACTIVATION_REGTEST,
                0,
                BIP112_CSV_ACTIVATION_REGTEST,
                0,
                0,
                0,
                CTV_ACTIVATION_REGTEST,
                CSFS_ACTIVATION_REGTEST,
            ),
        };

        let bip54 = bip54_activation_override.unwrap_or(match network {
            Network::Mainnet => BIP54_ACTIVATION_MAINNET,
            Network::Testnet => BIP54_ACTIVATION_TESTNET,
            Network::Regtest => BIP54_ACTIVATION_REGTEST,
        });

        Self {
            bip30_deactivation,
            bip16,
            bip34,
            bip66,
            bip65,
            bip112,
            bip147,
            segwit,
            taproot,
            ctv,
            csfs,
            bip54,
        }
    }
}

/// Taproot (BIP341) activation height for `network` (Core `chainparams` mainnet vs testnet3).
#[inline]
pub fn taproot_activation_height(network: Network) -> u64 {
    match network {
        Network::Mainnet => TAPROOT_ACTIVATION_MAINNET,
        Network::Testnet => TAPROOT_ACTIVATION_TESTNET,
        Network::Regtest => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Network;

    #[test]
    fn fork_table_testnet_matches_primitives() {
        let t = ForkActivationTable::from_network(Network::Testnet);
        assert_eq!(t.bip65, BIP65_ACTIVATION_TESTNET);
        assert_eq!(t.bip112, BIP112_CSV_ACTIVATION_TESTNET);
        assert_eq!(t.bip147, BIP147_ACTIVATION_TESTNET);
        assert_eq!(t.segwit, SEGWIT_ACTIVATION_TESTNET);
        assert_eq!(t.taproot, TAPROOT_ACTIVATION_TESTNET);
    }

    #[test]
    fn taproot_activation_height_matches_table() {
        for net in [Network::Mainnet, Network::Testnet, Network::Regtest] {
            let h = taproot_activation_height(net);
            let t = ForkActivationTable::from_network(net).taproot;
            assert_eq!(h, t, "{net:?}");
        }
    }
}
