//! Checked-in JSON fixture for default consensus configuration (serde shape + values).

use blvm_consensus::config::ConsensusConfig;

const CONSENSUS_CONFIG_DEFAULT_JSON: &str = include_str!("fixtures/consensus_config_default.json");

#[test]
fn consensus_config_default_json_fixture_matches_default() {
    let from_fixture: ConsensusConfig =
        serde_json::from_str(CONSENSUS_CONFIG_DEFAULT_JSON).expect("valid JSON fixture");
    assert_eq!(from_fixture, ConsensusConfig::default());
}

/// Overwrites `tests/fixtures/consensus_config_default.json` when defaults change intentionally.
#[test]
#[ignore]
fn write_consensus_config_default_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/consensus_config_default.json");
    let json = serde_json::to_string_pretty(&ConsensusConfig::default()).unwrap();
    std::fs::write(path, format!("{json}\n")).unwrap();
}
