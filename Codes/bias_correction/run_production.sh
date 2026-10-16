#!/usr/bin/env bash
# Production run for one noise (default gaussian): SGLM at h = 1/8 and 1/16 with T = 2e7 per chain, the exact
# coefficients from as many exact samples as the h = 1/16 run, and the table of the numerical biases against the exact
# means pi(f) of exact_means.jl (computed once, for both noises). The shift column of the table comes from run_phi.sh.
# Every step writes its own log in results/.
#
#   bash run_production.sh [noise=gaussian] [exact|noexact]   > results/run_production_NOISE.log 2>&1
#
# With noexact the exact coefficients are not computed.

set -euo pipefail
cd "$(dirname "$0")"
NOISE="${1:-gaussian}"
EXACT="${2:-exact}"
PY=python
mkdir -p results

echo "$(date '+%F %T') SGLM runs"
julia -t 32 sglm_runs.jl "$NOISE" 2e7 > "results/sglm_runs_$NOISE.log" 2>&1

if [ "$EXACT" = "exact" ]; then
    echo "$(date '+%F %T') exact coefficients"
    julia -t 32 exact_coefficients.jl "$NOISE" 2e7 > "results/exact_coefficients_$NOISE.log" 2>&1
fi

echo "$(date '+%F %T') table"
[ -f results/exact_means.csv ] || julia -t 32 exact_means.jl 2e7 > results/exact_means.log 2>&1
"$PY" table.py "$NOISE" | tee "results/table_$NOISE.log"
echo "$(date '+%F %T') done"
