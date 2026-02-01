#!/bin/bash
# Setup blvm-spec-lock dependency for CI
#
# Dependency resolution priority:
# 1. Default: Use crate from crates.io (blvm-spec-lock = "^0.1.0" - allows compatible updates)
# 2. Fallback: Clone GitHub repo and use path dependency if crate not available
# 3. Local dev: Use .cargo/config.toml with [patch.crates-io] (not in CI)

set -e

echo "🔍 Checking blvm-spec-lock dependency availability..."

# Check if crate exists on crates.io
CRATE_EXISTS=$(curl -s --max-time 5 "https://crates.io/api/v1/crates/blvm-spec-lock" 2>/dev/null | grep -q '"name":"blvm-spec-lock"' && echo "true" || echo "false")

if [ "$CRATE_EXISTS" = "true" ]; then
    echo "✅ blvm-spec-lock crate found on crates.io - using crate dependency"
    
    # Remove patch section if it exists (crate is now available, don't override it)
    if grep -q "^\[patch.crates-io\]" Cargo.toml && grep -A 1 "^\[patch.crates-io\]" Cargo.toml | grep -q "blvm-spec-lock"; then
        echo "🧹 Removing [patch.crates-io] section (crate is now available on crates.io)"
        
        # Remove the patch section using sed
        # Delete from [patch.crates-io] line through the blvm-spec-lock line and any trailing empty lines
        sed -i '/^\[patch\.crates-io\]/,/^blvm-spec-lock = { path = "\.\.\/blvm-spec-lock" }$/d' Cargo.toml
        
        # Remove any trailing empty lines after the deletion
        sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' Cargo.toml
        
        echo "✅ Removed [patch.crates-io] section"
    fi
    
    echo "No setup needed, Cargo will fetch from crates.io"
    exit 0
fi

echo "⚠️  blvm-spec-lock crate not found on crates.io - cloning from GitHub"

# Clone blvm-spec-lock repo (go up one directory from blvm-consensus)
cd "$(dirname "$0")/.."
PARENT_DIR="$(pwd)"
cd "$PARENT_DIR/.." || cd "$PARENT_DIR"

if [ ! -d "blvm-spec-lock" ]; then
    echo "📦 Cloning blvm-spec-lock repository..."
    git clone --depth 1 https://github.com/BTCDecoded/blvm-spec-lock.git || {
        echo "❌ Failed to clone blvm-spec-lock repository"
        exit 1
    }
else
    echo "✅ blvm-spec-lock directory already exists"
fi

# Return to blvm-consensus directory
cd "$PARENT_DIR"

# Patch Cargo.toml to use path dependency
echo "📝 Patching Cargo.toml to use local blvm-spec-lock..."

if ! grep -q "^\[patch.crates-io\]" Cargo.toml; then
    # Add [patch.crates-io] section
    echo "" >> Cargo.toml
    echo "[patch.crates-io]" >> Cargo.toml
    echo "blvm-spec-lock = { path = \"../blvm-spec-lock\" }" >> Cargo.toml
    echo "✅ Added [patch.crates-io] section to use local blvm-spec-lock"
else
    # Check if blvm-spec-lock is already in patch section
    if ! grep -A 1 "^\[patch.crates-io\]" Cargo.toml | grep -q "blvm-spec-lock"; then
        # Add to existing patch section
        sed -i '/^\[patch.crates-io\]/a blvm-spec-lock = { path = "../blvm-spec-lock" }' Cargo.toml
        echo "✅ Added blvm-spec-lock to existing [patch.crates-io] section"
    else
        echo "✅ blvm-spec-lock already patched in Cargo.toml"
    fi
fi

echo "✅ blvm-spec-lock dependency setup complete"

