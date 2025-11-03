#!/bin/bash
# Comprehensive fuzzing campaign runner
# Runs all fuzzing targets for 24+ hours each

set -e

FUZZ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FUZZ_DIR/.."

# Create directories
mkdir -p fuzz/corpus/{transaction_validation,block_validation,script_execution,compact_block_reconstruction,segwit_validation,mempool_operations,utxo_commitments,pow_validation,economic_validation,serialization,script_opcodes,differential_fuzzing}
mkdir -p fuzz/artifacts

# Function to run a fuzzing campaign
run_fuzz_campaign() {
    local target=$1
    local duration=${2:-86400}  # Default 24 hours
    
    echo "========================================="
    echo "Starting fuzzing campaign: $target"
    echo "Duration: $duration seconds ($(($duration / 3600)) hours)"
    echo "Start time: $(date)"
    echo "========================================="
    
    cargo +nightly fuzz run "$target" -- \
        -max_total_time="$duration" \
        -artifact_prefix=./fuzz/artifacts/ \
        -print_final_stats=1 \
        -timeout=60 \
        -max_len=100000 \
        2>&1 | tee "fuzz/artifacts/${target}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "========================================="
    echo "Fuzzing campaign completed: $target"
    echo "End time: $(date)"
    echo "========================================="
}

# Run campaigns
echo "Starting comprehensive fuzzing campaigns..."
echo "Note: Each campaign will run for 24 hours"
echo ""

# Check if running in background mode
if [ "$1" = "--background" ]; then
    echo "Running all campaigns in background..."
    
    # Transaction validation
    nohup bash -c "run_fuzz_campaign transaction_validation 86400" > fuzz/artifacts/transaction_validation_bg.log 2>&1 &
    echo "Started transaction_validation (PID: $!)"
    
    # Block validation
    nohup bash -c "run_fuzz_campaign block_validation 86400" > fuzz/artifacts/block_validation_bg.log 2>&1 &
    echo "Started block_validation (PID: $!)"
    
    # Script execution
    nohup bash -c "run_fuzz_campaign script_execution 86400" > fuzz/artifacts/script_execution_bg.log 2>&1 &
    echo "Started script_execution (PID: $!)"
    
    # Compact block reconstruction
    nohup bash -c "run_fuzz_campaign compact_block_reconstruction 86400" > fuzz/artifacts/compact_block_reconstruction_bg.log 2>&1 &
    echo "Started compact_block_reconstruction (PID: $!)"
    
    # SegWit validation
    nohup bash -c "run_fuzz_campaign segwit_validation 86400" > fuzz/artifacts/segwit_validation_bg.log 2>&1 &
    echo "Started segwit_validation (PID: $!)"
    
    # Mempool operations
    nohup bash -c "run_fuzz_campaign mempool_operations 86400" > fuzz/artifacts/mempool_operations_bg.log 2>&1 &
    echo "Started mempool_operations (PID: $!)"
    
    # UTXO commitments
    nohup bash -c "run_fuzz_campaign utxo_commitments 86400" > fuzz/artifacts/utxo_commitments_bg.log 2>&1 &
    echo "Started utxo_commitments (PID: $!)"
    
    # Proof of Work validation
    nohup bash -c "run_fuzz_campaign pow_validation 86400" > fuzz/artifacts/pow_validation_bg.log 2>&1 &
    echo "Started pow_validation (PID: $!)"
    
    # Economic validation
    nohup bash -c "run_fuzz_campaign economic_validation 86400" > fuzz/artifacts/economic_validation_bg.log 2>&1 &
    echo "Started economic_validation (PID: $!)"
    
    # Serialization
    nohup bash -c "run_fuzz_campaign serialization 86400" > fuzz/artifacts/serialization_bg.log 2>&1 &
    echo "Started serialization (PID: $!)"
    
    # Script opcodes
    nohup bash -c "run_fuzz_campaign script_opcodes 86400" > fuzz/artifacts/script_opcodes_bg.log 2>&1 &
    echo "Started script_opcodes (PID: $!)"
    
    # Differential fuzzing
    nohup bash -c "run_fuzz_campaign differential_fuzzing 86400" > fuzz/artifacts/differential_fuzzing_bg.log 2>&1 &
    echo "Started differential_fuzzing (PID: $!)"
    
    echo ""
    echo "All 12 campaigns started in background."
    echo "Monitor progress with: tail -f fuzz/artifacts/*_bg.log"
    echo "Check PIDs with: ps aux | grep 'cargo.*fuzz'"
else
    # Run sequentially (for shorter test runs)
    DURATION=${1:-300}  # Default 5 minutes for testing
    
    echo "Running short verification campaigns ($DURATION seconds each)..."
    
    run_fuzz_campaign transaction_validation "$DURATION"
    run_fuzz_campaign block_validation "$DURATION"
    run_fuzz_campaign script_execution "$DURATION"
    run_fuzz_campaign compact_block_reconstruction "$DURATION"
    run_fuzz_campaign segwit_validation "$DURATION"
    run_fuzz_campaign mempool_operations "$DURATION"
    run_fuzz_campaign utxo_commitments "$DURATION"
    run_fuzz_campaign pow_validation "$DURATION"
    run_fuzz_campaign economic_validation "$DURATION"
    run_fuzz_campaign serialization "$DURATION"
    run_fuzz_campaign script_opcodes "$DURATION"
    run_fuzz_campaign differential_fuzzing "$DURATION"
    
    echo ""
    echo "All 12 campaigns completed!"
fi

