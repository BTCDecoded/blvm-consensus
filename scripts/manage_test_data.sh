#!/bin/bash
# Test Data Management Utility
#
# Provides utilities for managing test data:
# - List test data
# - Verify test data integrity
# - Clean test data
# - Show test data statistics
#
# Usage:
#   ./scripts/manage_test_data.sh [list|verify|clean|stats|help]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DATA_DIR="$PROJECT_ROOT/tests/test_data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# List all test data
list_test_data() {
    echo "=== Test Data Inventory ==="
    echo ""
    
    # Core vectors
    echo "Core Test Vectors:"
    if [ -d "$TEST_DATA_DIR/core_vectors/transactions" ]; then
        COUNT=$(find "$TEST_DATA_DIR/core_vectors/transactions" -name "*.json" 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  ✅ Transactions: $COUNT files"
            ls -lh "$TEST_DATA_DIR/core_vectors/transactions"/*.json 2>/dev/null | awk '{print "    - " $9 " (" $5 ")"}'
        else
            echo "  ⚠️  No transaction vectors found"
        fi
    else
        echo "  ❌ Directory not found"
    fi
    
    echo ""
    
    # Mainnet blocks
    echo "Mainnet Blocks:"
    if [ -d "$TEST_DATA_DIR/mainnet_blocks" ]; then
        HEX_COUNT=$(find "$TEST_DATA_DIR/mainnet_blocks" -name "block_*.hex" 2>/dev/null | wc -l)
        BIN_COUNT=$(find "$TEST_DATA_DIR/mainnet_blocks" -name "block_*.bin" 2>/dev/null | wc -l)
        if [ "$HEX_COUNT" -gt 0 ] || [ "$BIN_COUNT" -gt 0 ]; then
            echo "  ✅ Blocks: $HEX_COUNT hex, $BIN_COUNT binary"
            if [ "$HEX_COUNT" -gt 0 ]; then
                echo "    Heights:"
                find "$TEST_DATA_DIR/mainnet_blocks" -name "block_*.hex" 2>/dev/null | \
                    sed 's/.*block_\([0-9]*\)\.hex/\1/' | sort -n | \
                    awk '{printf "      - Height %s\n", $1}'
            fi
        else
            echo "  ⚠️  No blocks found"
        fi
    else
        echo "  ❌ Directory not found"
    fi
    
    echo ""
    
    # Checkpoints
    echo "Checkpoints:"
    if [ -d "$TEST_DATA_DIR/checkpoints" ]; then
        COUNT=$(find "$TEST_DATA_DIR/checkpoints" -name "*.json" 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  ✅ Checkpoints: $COUNT files"
        else
            echo "  ⚠️  No checkpoints found (optional)"
        fi
    else
        echo "  ❌ Directory not found"
    fi
}

# Verify test data integrity
verify_test_data() {
    echo "=== Verifying Test Data ==="
    echo ""
    
    local errors=0
    
    # Verify Core vectors
    echo "Verifying Core test vectors..."
    if [ -f "$TEST_DATA_DIR/core_vectors/transactions/tx_valid.json" ]; then
        if command -v jq &> /dev/null; then
            if jq empty "$TEST_DATA_DIR/core_vectors/transactions/tx_valid.json" 2>/dev/null; then
                echo "  ✅ tx_valid.json: Valid JSON"
            else
                echo -e "  ${RED}❌ tx_valid.json: Invalid JSON${NC}"
                errors=$((errors + 1))
            fi
        else
            echo "  ⚠️  jq not found, skipping JSON validation"
        fi
    else
        echo "  ⚠️  tx_valid.json: Not found"
    fi
    
    if [ -f "$TEST_DATA_DIR/core_vectors/transactions/tx_invalid.json" ]; then
        if command -v jq &> /dev/null; then
            if jq empty "$TEST_DATA_DIR/core_vectors/transactions/tx_invalid.json" 2>/dev/null; then
                echo "  ✅ tx_invalid.json: Valid JSON"
            else
                echo -e "  ${RED}❌ tx_invalid.json: Invalid JSON${NC}"
                errors=$((errors + 1))
            fi
        fi
    else
        echo "  ⚠️  tx_invalid.json: Not found"
    fi
    
    echo ""
    
    # Verify mainnet blocks
    echo "Verifying mainnet blocks..."
    if [ -d "$TEST_DATA_DIR/mainnet_blocks" ]; then
        for block_file in "$TEST_DATA_DIR/mainnet_blocks"/block_*.hex; do
            if [ -f "$block_file" ]; then
                # Check if file contains valid hex
                if grep -q "^[0-9a-fA-F]*$" "$block_file" 2>/dev/null; then
                    echo "  ✅ $(basename "$block_file"): Valid hex"
                else
                    echo -e "  ${RED}❌ $(basename "$block_file"): Invalid hex${NC}"
                    errors=$((errors + 1))
                fi
            fi
        done
    fi
    
    echo ""
    
    if [ $errors -eq 0 ]; then
        echo -e "${GREEN}✅ All test data verified successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Found $errors errors${NC}"
        return 1
    fi
}

# Clean test data
clean_test_data() {
    echo "=== Cleaning Test Data ==="
    echo ""
    echo "This will remove all downloaded test data."
    echo "Test data can be re-downloaded using: ./scripts/download_test_data.sh"
    echo ""
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        return 0
    fi
    
    # Remove Core vectors
    if [ -d "$TEST_DATA_DIR/core_vectors/transactions" ]; then
        rm -f "$TEST_DATA_DIR/core_vectors/transactions"/*.json
        echo "✅ Removed Core transaction vectors"
    fi
    
    # Remove mainnet blocks
    if [ -d "$TEST_DATA_DIR/mainnet_blocks" ]; then
        rm -f "$TEST_DATA_DIR/mainnet_blocks"/*.hex
        rm -f "$TEST_DATA_DIR/mainnet_blocks"/*.bin
        echo "✅ Removed mainnet blocks"
    fi
    
    # Keep checkpoints (they're generated, not downloaded)
    echo "⚠️  Checkpoints preserved (they're generated, not downloaded)"
    
    echo ""
    echo "✅ Clean complete"
}

# Show test data statistics
show_stats() {
    echo "=== Test Data Statistics ==="
    echo ""
    
    local total_size=0
    local file_count=0
    
    # Core vectors
    if [ -d "$TEST_DATA_DIR/core_vectors" ]; then
        SIZE=$(du -sb "$TEST_DATA_DIR/core_vectors" 2>/dev/null | cut -f1)
        COUNT=$(find "$TEST_DATA_DIR/core_vectors" -type f 2>/dev/null | wc -l)
        total_size=$((total_size + SIZE))
        file_count=$((file_count + COUNT))
        
        echo "Core Vectors:"
        echo "  Files: $COUNT"
        echo "  Size: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo "$SIZE bytes")"
        echo ""
    fi
    
    # Mainnet blocks
    if [ -d "$TEST_DATA_DIR/mainnet_blocks" ]; then
        SIZE=$(du -sb "$TEST_DATA_DIR/mainnet_blocks" 2>/dev/null | cut -f1)
        COUNT=$(find "$TEST_DATA_DIR/mainnet_blocks" -type f 2>/dev/null | wc -l)
        total_size=$((total_size + SIZE))
        file_count=$((file_count + COUNT))
        
        echo "Mainnet Blocks:"
        echo "  Files: $COUNT"
        echo "  Size: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo "$SIZE bytes")"
        echo ""
    fi
    
    # Checkpoints
    if [ -d "$TEST_DATA_DIR/checkpoints" ]; then
        SIZE=$(du -sb "$TEST_DATA_DIR/checkpoints" 2>/dev/null | cut -f1)
        COUNT=$(find "$TEST_DATA_DIR/checkpoints" -type f 2>/dev/null | wc -l)
        total_size=$((total_size + SIZE))
        file_count=$((file_count + COUNT))
        
        echo "Checkpoints:"
        echo "  Files: $COUNT"
        echo "  Size: $(numfmt --to=iec-i --suffix=B $SIZE 2>/dev/null || echo "$SIZE bytes")"
        echo ""
    fi
    
    echo "Total:"
    echo "  Files: $file_count"
    echo "  Size: $(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")"
}

# Show help
show_help() {
    cat << EOF
Test Data Management Utility

Usage:
    $0 [command]

Commands:
    list      List all test data files
    verify    Verify test data integrity
    clean     Remove all downloaded test data
    stats     Show test data statistics
    help      Show this help message

Examples:
    $0 list          # List all test data
    $0 verify        # Verify test data integrity
    $0 clean         # Clean all test data
    $0 stats         # Show statistics

EOF
}

# Main command dispatch
case "${1:-help}" in
    list)
        list_test_data
        ;;
    verify)
        verify_test_data
        ;;
    clean)
        clean_test_data
        ;;
    stats)
        show_stats
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

