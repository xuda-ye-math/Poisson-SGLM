# Exact coefficients 1/2 pi(f psi_C) = 1/2 sigma^2 Cov_pi(f, U) for the Gaussian noise (C = sigma^2 I), from exact samples
# of pi only, as many as the samples of the h = 1/16 run of sglm_runs.jl (32 chains x T x 16).
#
#   julia -t 32 exact_coefficients.jl [noise=gaussian] [T=2e7]   > results/exact_coefficients_NOISE.log
#
# The samples are streamed; the standard error is taken over the estimates of the threads.
# Writes results/exact_coefficients_NOISE.csv.

include("model.jl")
using .HighDim
using Random, Statistics, Printf, Base.Threads

const NOISE = length(ARGS) >= 1 ? ARGS[1] : "gaussian"
const T = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e7
const N = 32 * round(Int, T * 16)
const REPORT = 30.0
const M = load_model(joinpath(@__DIR__, "model_d16.toml"))
const SAMPLER = RingSampler(M)

logline(s) = (println(s); flush(stdout))
logline(@sprintf("exact coefficients 1/2 sigma^2 Cov(f, U), noise = %s, from %.4e exact samples, threads = %d", NOISE, N,
                 nthreads()))
nt = nthreads()
acc = zeros(2NF + 1, nt)                 # per thread: means of f, f U, U
cnt = zeros(Int, nt)
t0 = time()
@threads for t in 1:nt
    rng = Xoshiro(81_000_000 + t)
    x, y, fv = zeros(M.d), zeros(M.d), zeros(NF)
    a = zeros(2NF + 1)
    n = 0
    last = time()
    for _ in t:nt:N
        sample_exact!(x, y, rng, SAMPLER, M)
        testfuns!(fv, x)
        U = potential(M, x, y)
        @inbounds for j in 1:NF
            a[j] += fv[j]
            a[NF+j] += fv[j] * U
        end
        a[2NF+1] += U
        n += 1
        if t == 1 && n % 1_000_000 == 0 && time() - last > REPORT
            last = time()
            logline(@sprintf("    thread 1 at %.0f%% after %.0f s", 100n * nt / N, last - t0))
        end
    end
    acc[:, t] = a ./ n
    cnt[t] = n
end
coef_t = [M.σ^2 / 2 * (acc[NF+j, t] - acc[j, t] * acc[2NF+1, t]) for j in 1:NF, t in 1:nt]
w = cnt ./ sum(cnt)
s = acc * w
coef = [M.σ^2 / 2 * (s[NF+j] - s[j] * s[2NF+1]) for j in 1:NF]
se = vec(std(coef_t; dims=2)) ./ sqrt(nt)

logline(@sprintf("%d exact samples in %.1f s:", sum(cnt), time() - t0))
open(joinpath(@__DIR__, "results", "exact_coefficients_$(NOISE).csv"), "w") do io
    println(io, "test,exact,standard_error")
    for j in 1:NF
        logline(@sprintf("  f%-2d %-28s %+.5f ± %.5f", j, TESTNAMES[j], coef[j], se[j]))
        println(io, @sprintf("f%d,%.12e,%.3e", j, coef[j], se[j]))
    end
end
logline("wrote results/exact_coefficients_$(NOISE).csv")
