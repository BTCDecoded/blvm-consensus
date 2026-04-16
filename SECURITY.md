# Security Policy

This document covers repo-specific security boundaries. See the [BTCDecoded Security Policy](https://github.com/BTCDecoded/.github/blob/main/SECURITY.md) for organization-wide policy.

## Dependency Version Pinning Strategy

All consensus-critical cryptographic dependencies are pinned to exact versions for supply chain security:

- **Consensus-critical cryptography**: Exact versions (`=`) for `secp256k1`, `ripemd`, `bitcoin_hashes`
- **Non-consensus utilities**: Exact versions for supply chain security (`serde`, `serde_json`, `anyhow`, `thiserror`)
- **Formal verification tools**: Exact versions for reproducibility (`proptest`, `blvm-spec-lock`)

**Rationale**: Consensus-critical code must be deterministic and reproducible. Exact version pinning prevents:
- Supply chain attacks through dependency updates
- Non-deterministic behavior from dependency changes
- Consensus divergence from dependency updates

See `Cargo.toml` for complete dependency list with exact versions.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**This is consensus-critical software handling Bitcoin's core validation rules. Security vulnerabilities could have severe consequences for the Bitcoin network.**

### Critical Security Issues

If you discover a security vulnerability in blvm-consensus, please report it immediately:

1. **DO NOT** create a public GitHub issue
2. **DO NOT** discuss the vulnerability publicly
3. **DO NOT** post on social media or forums

### How to Report

**Email:** security@btcdecoded.org  
**Subject:** [SECURITY] blvm-consensus vulnerability

Include the following information:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### Response Timeline

- **Acknowledgment:** Within 24 hours
- **Initial Assessment:** Within 72 hours
- **Fix Development:** 1-2 weeks (depending on severity)
- **Public Disclosure:** Coordinated with fix release

### Vulnerability Types

#### Critical (P0)
- Consensus rule bypasses
- Cryptographic vulnerabilities
- Memory safety issues
- Integer overflow/underflow
- Double-spending vulnerabilities

#### High (P1)
- Denial of service vulnerabilities
- Resource exhaustion
- Input validation bypasses
- Logic errors in validation

#### Medium (P2)
- Information disclosure
- Performance issues
- Documentation errors

### Responsible Disclosure

We follow responsible disclosure practices:

1. **Private reporting** - Report privately first
2. **Coordinated disclosure** - We'll work with you on timing
3. **Credit** - We'll credit you (unless you prefer anonymity)
4. **No legal action** - We won't pursue legal action for good-faith research

### Security Considerations

#### Consensus Rules
- All consensus rules must be implemented exactly as specified
- No undocumented deviations from the Orange Paper and mainnet consensus rules
- All edge cases must be handled correctly
- Integer arithmetic must be precise

#### Cryptographic Operations
- All cryptographic operations use exact versions of libraries
- No custom cryptographic implementations
- All hash functions must match Bitcoin specifications
- Signature verification must be exact

#### Memory Safety
- All code must be memory-safe
- No unsafe blocks without justification
- All array access must be bounds-checked
- No use-after-free or double-free

### Testing Requirements

Before reporting, please verify:
- [ ] The issue reproduces consistently
- [ ] The issue affects consensus validation
- [ ] The issue is not already known
- [ ] The issue is not a feature request

### Bug Bounty

We may offer bug bounties for critical vulnerabilities. Contact us for details.

### Security Updates

Security updates will be:
- Released as patch versions (0.1.x)
- Clearly marked as security fixes
- Backported to all supported versions
- Announced on our security mailing list

### Contact Information

- **Security Team:** security@btcdecoded.org
- **General Inquiries:** info@btcdecoded.org
- **Website:** https://btcdecoded.org

### Acknowledgments

We thank the security researchers who help keep Bitcoin secure through responsible disclosure.

---

**Remember:** This software handles real Bitcoin consensus rules. Any bugs could affect the entire Bitcoin network. Please report responsibly.
