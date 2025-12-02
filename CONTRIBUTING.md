# Contributing to blvm-consensus

Thank you for your interest in contributing to blvm-consensus! This document contains **repo-specific guidelines only**. For comprehensive contributing guidelines, see the [BLVM Documentation](https://docs.thebitcoincommons.org/development/contributing.html).

## Quick Links

- **[Complete Contributing Guide](https://docs.thebitcoincommons.org/development/contributing.html)** - Full developer workflow
- **[PR Process](https://docs.thebitcoincommons.org/development/pr-process.html)** - Governance tiers and review process
- **[Testing Infrastructure](https://docs.thebitcoincommons.org/development/testing.html)** - Testing guides

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). By participating, you agree to uphold this code.

## Repository-Specific Guidelines

### Consensus Rules

**CRITICAL:** This code implements Bitcoin consensus rules. Any changes must:

1. **Match Bitcoin Core behavior exactly** - No deviations
2. **Not deviate from the Orange Paper specifications** - Mathematical correctness required
3. **Handle all edge cases correctly** - Consensus code must be bulletproof
4. **Maintain mathematical precision** - No approximations

### Additional Requirements

- **Exact Version Pinning**: All consensus-critical dependencies must be pinned to exact versions
- **Pure Functions**: All functions must remain side-effect-free
- **Comprehensive Testing**: All mathematical functions must be thoroughly tested
- **Test Coverage**: >95% for consensus-critical code
- **Formal Verification**: Consensus-critical changes may require Kani proofs

### Development Setup

```bash
git clone https://github.com/BTCDecoded/blvm-consensus.git
cd blvm-consensus
cargo build
cargo test
```

### Running Tests

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run specific test categories
cargo test --test integration_tests
cargo test --test comprehensive_unit_tests
```

### Review Criteria

Reviewers will check:

1. **Correctness** - Does the code work as intended?
2. **Consensus compliance** - Does it match Bitcoin Core exactly?
3. **Test coverage** - Are all cases covered (>95%)?
4. **Performance** - No regressions?
5. **Documentation** - Is it clear and complete?
6. **Security** - Any potential vulnerabilities?
7. **Mathematical precision** - No approximations or shortcuts

### Approval Process

- **At least 2 approvals** required for consensus-critical changes
- **Security team review** for cryptographic changes
- **Performance review** for hot path changes
- **Formal verification review** for consensus-critical changes

## Getting Help

- **Documentation**: [docs.thebitcoincommons.org](https://docs.thebitcoincommons.org)
- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Security**: See [SECURITY.md](SECURITY.md)

Thank you for contributing to blvm-consensus!
