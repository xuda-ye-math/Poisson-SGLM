# Production SGLM runs on the rotated product of rings (model.jl, model_d16.toml) at h = 1/8 and h = 1/16.
#
#   julia -t 32 sglm_runs.jl [noise=gaussian] [T=2e7]   > results/sglm_runs_NOISE.log
#
# For each step size: 32 chains, each started from an exact sample of pi, burn-in 100, then the time T; the twelve test
# functions are averaged over blocks of 100 time units. Writes
#   results/sglm_NOISE_h8.csv, results/sglm_NOISE_h16.csv   time averages pi_h(f) and their standard errors (blocks)
#   results/extrapolation_NOISE.csv                          (pi_{1/8}(f) - pi_{1/16}(f)) / (1/16) and its standard error
#   results/data_NOISE/train_x.f32, valid_x.f32, meta.toml   from the h = 1/16 run: the first 525 000 states of each chain
#                                                            after the burn-in (16 800 000 training samples, NTRAIN x d,
#                                                            row = sample) and the next 2048 states (65 536 validation)

include("model.jl")
using .HighDim
using LinearAlgebra, Random, Statistics, Printf, TOML, Base.Threads

const NOISE = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :gaussian
const T = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2e7
const NCHAIN = 32
const TBURN = 100.0
const TBLOCK = 100.0
const NTRAIN_PER = 525_000
const NVALID_PER = 2_048
const REPORT = 30.0
const OUT = joinpath(@__DIR__, "results")
const DATA = joinpath(OUT, "data_$(NOISE)")

logline(s) = (println(s); flush(stdout))
const M = load_model(joinpath(@__DIR__, "model_d16.toml"))
const D = M.d
const SAMPLER = RingSampler(M)
isdir(DATA) && error("$DATA exists; data are never overwritten")
mkpath(DATA)
logline(@sprintf("d = %d, noise = %s, sigma = %.1f, %d chains, T = %.1e per chain, threads = %d", D, NOISE, M.σ, NCHAIN, T,
                 nthreads()))

"one chain: sums of the block means and of their squares (NF each); stores states into XT, XV when given"
function chain(h, seed, c, XT, XV, report)
    rng = Xoshiro(seed)
    z, y, g, w = zeros(D), zeros(D), zeros(D), zeros(D)
    sample_exact!(z, y, rng, SAMPLER, M)
    χ, χn, fv, blk = randn(rng, D), zeros(D), zeros(NF), zeros(NF)
    a = sqrt(h / 2)
    nburn, nblock, nb = round(Int, TBURN / h), round(Int, TBLOCK / h), round(Int, T / TBLOCK)
    s1, s2 = zeros(NF), zeros(NF)
    t0 = last = time()
    ntot = nburn + nb * nblock
    for k in 1:ntot
        grad!(g, y, M, z)
        add_noise!(g, w, rng, M, z, NOISE)
        randn!(rng, χn)
        @inbounds for i in 1:D
            z[i] += -h * g[i] + a * (χ[i] + χn[i])
        end
        χ, χn = χn, χ
        k > nburn || continue
        j = k - nburn
        if XT !== nothing
            if j <= NTRAIN_PER
                @inbounds for i in 1:D
                    XT[i, (c - 1) * NTRAIN_PER + j] = z[i]
                end
            elseif j <= NTRAIN_PER + NVALID_PER
                @inbounds for i in 1:D
                    XV[i, (c - 1) * NVALID_PER + j - NTRAIN_PER] = z[i]
                end
            end
        end
        testfuns!(fv, z)
        blk .+= fv
        if j % nblock == 0
            blk ./= nblock
            s1 .+= blk
            s2 .+= blk .^ 2
            blk .= 0
        end
        isfinite(z[1]) || error("chain $c diverged at step $k")
        if report && k % 1_000_000 == 0 && time() - last > REPORT
            last = time()
            logline(@sprintf("    h = 1/%d: chain 1 at %.0f%% after %.0f s", round(Int, 1 / h), 100k / ntot, last - t0))
        end
    end
    return s1, s2, nb
end

function run(h; store=false)
    XT = store ? zeros(Float32, D, NCHAIN * NTRAIN_PER) : nothing
    XV = store ? zeros(Float32, D, NCHAIN * NVALID_PER) : nothing
    S1, S2 = zeros(NF, NCHAIN), zeros(NF, NCHAIN)
    nb = 0
    t0 = time()
    @threads for c in 1:NCHAIN
        s1, s2, n = chain(h, 71_000_000 + 1000 * round(Int, 1 / h) + c, c, XT, XV, c == 1)
        S1[:, c], S2[:, c] = s1, s2
        nb = n
    end
    ntot = NCHAIN * nb
    m = vec(sum(S1; dims=2)) ./ ntot
    v = vec(sum(S2; dims=2)) ./ ntot .- m .^ 2
    se = sqrt.(v ./ ntot)
    logline(@sprintf("SGLM h = 1/%d: %d chains x T = %.1e (%.2e samples) in %.1f s", round(Int, 1 / h), NCHAIN, T,
                     NCHAIN * T / h, time() - t0))
    open(joinpath(OUT, "sglm_$(NOISE)_h$(round(Int, 1 / h)).csv"), "w") do io
        println(io, "test,mean,standard_error")
        for j in 1:NF
            println(io, @sprintf("f%d,%.12e,%.3e", j, m[j], se[j]))
            logline(@sprintf("  f%-2d pi_h(f) = %+.6f ± %.6f", j, m[j], se[j]))
        end
    end
    return m, se, XT, XV
end

m8, se8, _, _ = run(1 / 8)
m16, se16, XT, XV = run(1 / 16; store=true)

write(joinpath(DATA, "train_x.f32"), XT)
write(joinpath(DATA, "valid_x.f32"), XV)
open(joinpath(DATA, "meta.toml"), "w") do io
    TOML.print(io, Dict("noise" => String(NOISE), "h" => 1 / 16, "sigma" => M.σ, "d" => D, "chains" => NCHAIN,
                        "n_train" => size(XT, 2), "n_valid" => size(XV, 2), "tests" => TESTNAMES,
                        "source" => "first $NTRAIN_PER and next $NVALID_PER states of each chain of the h = 1/16 run of sglm_runs.jl"))
end
logline("wrote $DATA: $(size(XT, 2)) training and $(size(XV, 2)) validation samples")

logline("extrapolation (pi_{1/8}(f) - pi_{1/16}(f)) / (1/16):")
open(joinpath(OUT, "extrapolation_$(NOISE).csv"), "w") do io
    println(io, "test,extrapolation,standard_error")
    for j in 1:NF
        ex, se = (m8[j] - m16[j]) * 16, sqrt(se8[j]^2 + se16[j]^2) * 16
        println(io, @sprintf("f%d,%.12e,%.3e", j, ex, se))
        logline(@sprintf("  f%-2d %-28s %+.5f ± %.5f", j, TESTNAMES[j], ex, se))
    end
end
