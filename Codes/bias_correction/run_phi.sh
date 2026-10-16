#!/usr/bin/env bash
# phi_C from the extrapolated energy, for one noise (default gaussian): stored SGLM states at h = 1/16 and 1/8
# (sglm_states.jl, skipped when results/states_NOISE exists), the training (train_phi.py, GPU), grad phi on the
# validation states (eval_phi.py), and the coefficients and biases it implies (phi_coefficients.jl, against the exact
# means of exact_means.jl). Every step writes its own log in results/.
#
#   bash run_phi.sh [noise=gaussian] [epochs=100]   > results/run_phi_NOISE.log 2>&1

set -euo pipefail
cd "$(dirname "$0")"
NOISE="${1:-gaussian}"
EPOCHS="${2:-100}"
PY=python
mkdir -p results

echo "$(date '+%F %T') SGLM states"
[ -d "results/states_$NOISE" ] || julia -t 32 sglm_states.jl "$NOISE" > "results/sglm_states_$NOISE.log" 2>&1
echo "$(date '+%F %T') training (GPU)"
"$PY" -u train_phi.py "$NOISE" "$EPOCHS" > "results/train_phi_$NOISE.log" 2>&1
echo "$(date '+%F %T') grad phi on the validation states"
"$PY" -u eval_phi.py "$NOISE" > "results/eval_phi_$NOISE.log" 2>&1
echo "$(date '+%F %T') coefficients and biases"
julia -t 32 phi_coefficients.jl "$NOISE" > "results/phi_coefficients_$NOISE.log" 2>&1
cat "results/phi_coefficients_$NOISE.log"
echo "$(date '+%F %T') done"
