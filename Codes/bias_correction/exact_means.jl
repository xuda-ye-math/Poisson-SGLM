# Exact means pi(f) of the twelve test functions, from exact samples of pi only, as many as the samples of the h = 1/16
# run of sglm_runs.jl (32 chains x T x 16). The target pi does not depend on the gradient noise, so one file serves both
# noises; table.py subtracts these means to give the numerical biases.
#
#   julia -t 32 exact_means.jl [T=2e7]   > results/exact_means.log
#
# The samples are streamed; the standard error is taken over the estimates of the threads.
# Writes results/exact_means.csv.

include("model.jl")
using .HighDim
using Random, Statistics, Printf, Base.Threads

const T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 2e7
const N = 32 * round(Int, T * 16)
const REPORT = 30.0
const M = load_model(joinpath(@__DIR__, "model_d16.toml"))
const SAMPLER = RingSampler(M)

logline(s) = (println(s); flush(stdout))
logline(@sprintf("exact means pi(f), from %.4e exact samples, threads = %d", N, nthreads()))
nt = nthreads()
acc = zeros(NF, nt)                      # per thread: means of f
cnt = zeros(Int, nt)
t0 = time()
@threads for t in 1:nt
    rng = Xoshiro(91_000_000 + t)
    x, y, fv = zeros(M.d), zeros(M.d), zeros(NF)
    a = zeros(NF)
    n = 0
    last = time()
    for _ in t:nt:N
        sample_exact!(x, y, rng, SAMPLER, M)
        testfuns!(fv, x)
        @inbounds for j in 1:NF
            a[j] += fv[j]
        end
        n += 1
        if t == 1 && n % 1_000_000 == 0 && time() - last > REPORT
            last = time()
            logline(@sprintf("    thread 1 at %.0f%% after %.0f s", 100n * nt / N, last - t0))
        end
    end
    acc[:, t] = a ./ n
    cnt[t] = n
end
m = acc * (cnt ./ sum(cnt))
se = vec(std(acc; dims=2)) ./ sqrt(nt)

logline(@sprintf("%d exact samples in %.1f s:", sum(cnt), time() - t0))
open(joinpath(@__DIR__, "results", "exact_means.csv"), "w") do io
    println(io, "test,exact,standard_error")
    for j in 1:NF
        logline(@sprintf("  f%-2d %-28s %+.6f ± %.6f", j, TESTNAMES[j], m[j], se[j]))
        println(io, @sprintf("f%d,%.12e,%.3e", j, m[j], se[j]))
    end
end
logline("wrote results/exact_means.csv")
