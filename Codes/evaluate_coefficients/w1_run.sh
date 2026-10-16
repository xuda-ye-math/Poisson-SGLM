#!/usr/bin/env bash
# W1 coefficients on the grids 25, 50, 100 (Delta = 0.72, 0.36, 0.18 on [-9, 9]^4), on the GPU:
# by default the Gaussian test (exact value 0.939986), then SGLD and SGLM with the Gaussian noise;
# other cases are given as arguments. Each grid
# appends a row to results/w1/w1_runs.csv and logs to results/w1/w1_coefficients.log; the console
# output of each case goes to results/w1/launch_CASE_n100.out. At the end, w1_table.py rewrites
# the final tables tables/w1_coefficients_gaussian.csv and tables/w1_coefficients_nongaussian.csv.
#
#   bash w1_run.sh                                    # test_gaussian SGLD_gaussian SGLM_gaussian
#   bash w1_run.sh SGLD_nongaussian SGLM_nongaussian
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/w1
if [ $# -gt 0 ]; then cases=("$@"); else cases=(test_gaussian SGLD_gaussian SGLM_gaussian); fi
for c in "${cases[@]}"; do
    # maxit 4e5: on 64^4 the SGLM case already needed 123,600 iterations
    julia --project=. -t 16 w1_coefficients.jl "$c" 25,50,100 maxit=400000 > "results/w1/launch_${c}_n100.out" 2>&1
done
python w1_table.py
