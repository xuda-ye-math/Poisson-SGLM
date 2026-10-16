# SGLD or SGLM on the 4D mixture of model.toml, with one of the two noises, at the step size
# h = 2^(-hexp). One chain per thread, started from an exact sample of pi. After a burn-in of
# T_BURN, every step adds all the test functions of the pool to the block sums, and every block
# of T_BLOCK time units appends its sums to blocks_cXX.f64.
#   fixed-T mode (T > 0): every chain runs T / T_BLOCK blocks, that is the time T after the burn-in;
#   target mode (T = 0):  the run stops when the standard error of every selected coefficient
#                         estimate (bias / h) is below TARGET times its theoretical value, after at
#                         least MIN_BLOCKS blocks per chain, or after WALL seconds.
#
#   julia -t 16 simulate.jl SCHEME NOISE HEXP [out=DIR] [T=0] [block=500] [seed=0] [target=0.05] [wall=1800]
#   SCHEME = SGLD | SGLM,  NOISE = gaussian | nongaussian
#
# Files in OUT/SCHEME_NOISE_hexpHEXP/:
#   run.toml          parameters, seeds, file layout; the [result] table is appended at the end
#   blocks_cXX.f64    little-endian Float64 rows (block index, steps, sum f_1, ..., sum f_NF) of
#                     chain XX, f in the order of the [[test]] tables of model.toml
#   progress.log      the progress lines
#   summary.csv       bias, standard error and coefficient estimate of every test function
# and one row per run appended to OUT/timing.csv (wall time, steps, ns per step per chain).
# (The dry runs in results/dryrun were made before the switch to sums: their blocks hold the
# block means, and their traj_cXX.f32 the position every 10 time units.)

include(joinpath(@__DIR__, "model.jl"))
using .Model
using Random, Printf, Statistics, Dates, Base.Threads

const DIR = @__DIR__
const SCHEME = ARGS[1]
const NOISE = ARGS[2]
const HEXP = parse(Int, ARGS[3])
const OPT = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[4:end])
@assert issubset(keys(OPT), ("out", "T", "block", "seed", "target", "wall")) "unknown option in $(ARGS[4:end])"
const OUTROOT = abspath(get(OPT, "out", joinpath(DIR, "results", "runs")))
const T_CHAIN = parse(Float64, get(OPT, "T", "0"))
const T_BLOCK = parse(Float64, get(OPT, "block", "500"))
const SEEDLABEL = parse(Int, get(OPT, "seed", "0"))
const TARGET = parse(Float64, get(OPT, "target", "0.05"))
const WALL = parse(Float64, get(OPT, "wall", "1800"))
@assert SCHEME in ("SGLD", "SGLM") && NOISE in ("gaussian", "nongaussian")
const SCHEME_ID = SCHEME == "SGLD" ? 1 : 2
const NOISE_ID = NOISE == "gaussian" ? 1 : 2
const H = 2.0^(-HEXP)
const FIXED = T_CHAIN > 0

const T_BURN = 200.0
const MIN_BLOCKS = 20
const REPORT = 30.0

const mix, tests, noise, selected = load_model(joinpath(DIR, "model.toml"))
const Ustar = U_STAR(mix)
const NF = length(tests)
const NCH = nthreads()
const STEPS_BURN = round(Int, T_BURN / H)
# a block lasts T_BLOCK; in target mode at least 10^4 steps, so that large h does not write too often
const STEPS_BLOCK = FIXED ? round(Int, T_BLOCK / H) : max(round(Int, T_BLOCK / H), 10_000)
const NBLOCKS = FIXED ? round(Int, T_CHAIN / T_BLOCK) : 0
FIXED && @assert NBLOCKS * T_BLOCK == T_CHAIN && STEPS_BLOCK * H == T_BLOCK "T and block must give whole blocks of whole steps"
const PI_F = [exact_mean(f, mix) for f in tests]
const SEL = [findfirst(f -> f.name == s, tests) for s in selected]

# theoretical coefficients of this scheme and noise, from results/coefficients.csv
const THEORY = let d = Dict{String,Float64}()
    for l in Iterators.drop(eachline(joinpath(DIR, "results", "coefficients.csv")), 1)
        p = split(l, ",")
        (p[2] == NOISE && p[3] == SCHEME) && (d[p[1]] = parse(Float64, p[4]))
    end
    [d[s] for s in selected]
end

const RUNDIR = joinpath(OUTROOT, "$(SCHEME)_$(NOISE)_hexp$(HEXP)")
isdir(RUNDIR) && error("$RUNDIR exists; a run never writes into the directory of an earlier run")
mkpath(RUNDIR)
seed(c) = 10_000_000 * SEEDLABEL + 1_000_000 * SCHEME_ID + 100_000 * NOISE_ID + 1000 * HEXP + c

function logline(s)
    println(s); flush(stdout)
    open(io -> println(io, s), joinpath(RUNDIR, "progress.log"), "a")
end

open(joinpath(RUNDIR, "run.toml"), "w") do io
    println(io, "scheme = \"$SCHEME\"\nnoise = \"$NOISE\"\nhexp = $HEXP\nh = $H")
    println(io, "chains = $NCH\nseed_label = $SEEDLABEL\nseeds = [", join(seed.(1:NCH), ", "), "]")
    println(io, "mode = \"$(FIXED ? "fixed_T" : "target")\"")
    FIXED ? println(io, "t_chain = $T_CHAIN\nblocks_per_chain = $NBLOCKS") :
            println(io, "target = $TARGET\nwall = $WALL\nmin_blocks = $MIN_BLOCKS")
    println(io, "t_burn = $T_BURN\nt_block = $(STEPS_BLOCK * H)\nsteps_block = $STEPS_BLOCK")
    println(io, "tests = [", join(("\"$(f.name)\"" for f in tests), ", "), "]")
    println(io, "pi_f = [", join((@sprintf("%.16e", p) for p in PI_F), ", "), "]")
    println(io, "blocks_columns = [\"block\", \"steps\", ", join(("\"$(f.name)\"" for f in tests), ", "), "]")
    println(io, "blocks_values = \"sum\"")
    println(io, "model = \"model.toml\"")
    println(io, "started = \"$(now())\"")
end
cp(joinpath(DIR, "model.toml"), joinpath(RUNDIR, "model.toml"))

@inline function step(x::V4, chi::V4, rng)
    U, gU, _, _ = potential(mix, x)
    η = NOISE_ID == 1 ? noise1(rng, noise) : noise2(rng, noise, x, U, Ustar)
    chin = (randn(rng), randn(rng), randn(rng), randn(rng))
    if SCHEME_ID == 1
        a = sqrt(2H)
        return ntuple(i -> x[i] - H * (gU[i] + η[i]) + a * chin[i], 4), chin
    else
        a = sqrt(H / 2)
        return ntuple(i -> x[i] - H * (gU[i] + η[i]) + a * (chi[i] + chin[i]), 4), chin
    end
end

function run_chain(c, stop, failed, store, lk)
    rng = Xoshiro(seed(c))
    k = something(findfirst(>=(rand(rng)), cumsum(mix.w)), length(mix.w))
    z = (randn(rng), randn(rng), randn(rng), randn(rng))
    L = mix.chol[k]
    x = ntuple(i -> mix.mu[k][i] + L[i][1] * z[1] + L[i][2] * z[2] + L[i][3] * z[3] + L[i][4] * z[4], 4)
    chi = (randn(rng), randn(rng), randn(rng), randn(rng))
    for _ in 1:STEPS_BURN
        x, chi = step(x, chi, rng)
    end
    sums = zeros(NF)
    row = zeros(2 + NF)
    bio = open(joinpath(RUNDIR, @sprintf("blocks_c%02d.f64", c)), "w")
    b = 0
    while !stop[] && (!FIXED || b < NBLOCKS)
        fill!(sums, 0.0)
        for _ in 1:STEPS_BLOCK
            @inbounds for t in 1:NF
                sums[t] += evalf(tests[t], x)
            end
            x, chi = step(x, chi, rng)
        end
        if !all(isfinite, x) || maximum(abs, x) > 1e6
            failed[] = true; stop[] = true
            break
        end
        b += 1
        row[1] = b; row[2] = STEPS_BLOCK
        row[3:end] .= sums
        write(bio, row); flush(bio)
        lock(lk) do
            append!(store[c], sums ./ STEPS_BLOCK)
        end
    end
    close(bio)
    return b
end

function estimate(store, lk)
    blocks = lock(lk) do
        [reshape(copy(s), NF, :) for s in store]                  # NF x (blocks of the chain)
    end
    nmin = minimum(A -> size(A, 2), blocks)
    M = reduce(hcat, blocks)
    nb = size(M, 2)
    nb < 2 && return nothing
    m = vec(mean(M, dims = 2)); se = vec(std(M, dims = 2)) ./ sqrt(nb)
    # the same with batches of 8 consecutive blocks of each chain, to check the batch length
    M8 = zeros(NF, sum(A -> size(A, 2) ÷ 8, blocks))
    j = 0
    for A in blocks, i in 1:8:(size(A, 2)-7)
        j += 1
        M8[:, j] .= vec(mean(view(A, :, i:i+7), dims = 2))
    end
    se8 = j >= 2 ? vec(std(M8, dims = 2)) ./ sqrt(j) : fill(NaN, NF)
    return (nmin = nmin, nb = nb, bias = m .- PI_F, se = se, se8 = se8)
end

function write_summary(e)
    open(joinpath(RUNDIR, "summary.csv"), "w") do io
        println(io, "test,pi_f,bias,se,se_batch8,coef_est,coef_se,coef_theory")
        for t in 1:NF
            th = t in SEL ? THEORY[findfirst(==(t), SEL)] : NaN
            @printf(io, "%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", tests[t].name, PI_F[t], e.bias[t], e.se[t],
                e.se8[t], e.bias[t] / H, e.se[t] / H, th)
        end
    end
end

function main()
    logline(@sprintf("%s %s h = 2^-%d = %g, %d chains, %d steps per block of %g, burn-in %g, seed label %d, ",
        SCHEME, NOISE, HEXP, H, NCH, STEPS_BLOCK, STEPS_BLOCK * H, T_BURN, SEEDLABEL) *
        (FIXED ? @sprintf("fixed T = %g per chain (%d blocks)", T_CHAIN, NBLOCKS) : @sprintf("target %.3f, wall %g s", TARGET, WALL)))
    logline("theory (selected): " * join((@sprintf("%s %+.5f", selected[i], THEORY[i]) for i in eachindex(SEL)), ",  "))
    stop = Threads.Atomic{Bool}(false)
    failed = Threads.Atomic{Bool}(false)
    store = [Float64[] for _ in 1:NCH]                             # block means of chain c, NF per block
    lk = ReentrantLock()
    t0 = time()
    tasks = [Threads.@spawn run_chain(c, stop, failed, store, lk) for c in 1:NCH]
    reason = FIXED ? "fixed_T" : "wall"
    while true
        tnext = time() + REPORT
        while time() < tnext && !all(istaskdone, tasks)
            sleep(1)
        end
        done = all(istaskdone, tasks)
        e = estimate(store, lk)
        el = time() - t0
        if e === nothing
            logline(@sprintf("[%7.0fs] no complete block yet", el))
        else
            write_summary(e)
            parts = [@sprintf("%s %+.4f±%.4f (%+.4f)", selected[i], e.bias[t] / H, e.se[t] / H, THEORY[i]) for (i, t) in enumerate(SEL)]
            logline(@sprintf("[%7.0fs] T/chain = %.3e, blocks/chain >= %d, %.2e steps/s | ", el, e.nmin * STEPS_BLOCK * H, e.nmin,
                (e.nb * STEPS_BLOCK + NCH * STEPS_BURN) / el) * join(parts, " | "))
            if !FIXED && e.nmin >= MIN_BLOCKS && all(e.se[t] / H <= TARGET * abs(THEORY[i]) for (i, t) in enumerate(SEL))
                reason = "target"; break
            end
        end
        failed[] && (reason = "diverged"; break)
        FIXED ? (done && break) : (el >= WALL && break)
    end
    stop[] = true
    nblocks = fetch.(tasks)
    e = estimate(store, lk)
    e === nothing || write_summary(e)
    failed[] && (reason = "diverged")
    wall = time() - t0
    steps = STEPS_BURN .+ nblocks .* STEPS_BLOCK                 # steps of each chain, burn-in included
    ns_step = 1e9 * wall / maximum(steps)                         # wall time per step of one chain (chains in parallel)
    open(joinpath(RUNDIR, "run.toml"), "a") do io
        println(io, "\n[result]\nstop = \"$reason\"\nfinished = \"$(now())\"\nwall_seconds = $(round(wall, digits = 1))")
        println(io, "blocks = [", join(nblocks, ", "), "]")
        println(io, "steps_per_chain = [", join(steps, ", "), "]")
        println(io, "total_steps = $(sum(steps))")
        println(io, "ns_per_step_per_chain = $(round(ns_step, digits = 2))")
        println(io, "total_steps_per_second = $(round(sum(steps) / wall, sigdigits = 4))")
    end
    tfile = joinpath(OUTROOT, "timing.csv")
    isfile(tfile) || open(io -> println(io, "scheme,noise,hexp,h,chains,stop,wall_seconds,steps_per_chain,time_per_chain,ns_per_step_per_chain,total_steps_per_second"), tfile, "w")
    open(tfile, "a") do io
        @printf(io, "%s,%s,%d,%.10g,%d,%s,%.1f,%d,%.6g,%.3f,%.6g\n", SCHEME, NOISE, HEXP, H, NCH, reason, wall,
            maximum(steps), maximum(steps) * H, ns_step, sum(steps) / wall)
    end
    logline(@sprintf("timing: %.1f s wall, %d steps per chain, %.2f ns per step per chain, %.3e steps/s in total",
        wall, maximum(steps), ns_step, sum(steps) / wall))
    logline("stopped ($reason) after $(round(time() - t0, digits = 1)) s; blocks per chain: $(nblocks)")
    if e !== nothing
        for (i, t) in enumerate(SEL)
            logline(@sprintf("  %-12s bias %+.6f ± %.6f (batch x8 %.6f)   bias/h %+.5f ± %.5f   theory %+.5f   ratio %.3f",
                selected[i], e.bias[t], e.se[t], e.se8[t], e.bias[t] / H, e.se[t] / H, THEORY[i], e.bias[t] / H / THEORY[i]))
        end
    end
end

main()
