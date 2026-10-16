# Stored SGLM states at h = 1/16 and h = 1/8, for training phi_C by the extrapolated energy (train_phi.py).
#
#   julia -t 32 sglm_states.jl [noise=gaussian] [nper=525000]   > results/sglm_states_NOISE.log
#
# For each step size, a training set and a validation set from independent groups of 32 chains. Each chain starts from
# an exact sample of pi, runs a burn-in of 100 time units, then stores one state every DT = 1 time unit, nper states
# per chain. Writes results/states_NOISE/{train,valid}_h{16,8}.f32 (32 nper x d, row = sample, chain by chain) and
# meta.toml.

include("model.jl")
using .HighDim
using Random, Printf, TOML, Base.Threads

const NOISE = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :gaussian
const NPER = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 525_000
const NCHAIN = 32
const TBURN = 100.0
const DT = 1.0
const HS = (1 / 16, 1 / 8)
const SEEDS = Dict("train" => 72_000_000, "valid" => 73_000_000)
const OUT = joinpath(@__DIR__, "results", "states_$(NOISE)")

logline(s) = (println(s); flush(stdout))
const M = load_model(joinpath(@__DIR__, "model_d16.toml"))
const D = M.d
const SAMPLER = RingSampler(M)
isdir(OUT) && error("$OUT exists; data are never overwritten")
mkpath(OUT)
logline(@sprintf("d = %d, noise = %s, sigma = %.1f, %d chains per set, %d states per chain, one every %.1f time units",
                 D, NOISE, M.σ, NCHAIN, NPER, DT))

"one chain at step h: stores NPER states, one every DT, into the columns (c - 1) NPER + 1 : c NPER of X"
function chain!(X, h, seed, c)
    rng = Xoshiro(seed)
    z, y, g, w = zeros(D), zeros(D), zeros(D), zeros(D)
    sample_exact!(z, y, rng, SAMPLER, M)
    χ, χn = randn(rng, D), zeros(D)
    a = sqrt(h / 2)
    nburn, thin = round(Int, TBURN / h), round(Int, DT / h)
    for k in 1:(nburn + NPER * thin)
        grad!(g, y, M, z)
        add_noise!(g, w, rng, M, z, NOISE)
        randn!(rng, χn)
        @inbounds for i in 1:D
            z[i] += -h * g[i] + a * (χ[i] + χn[i])
        end
        χ, χn = χn, χ
        j = k - nburn
        if j > 0 && j % thin == 0
            @inbounds for i in 1:D
                X[i, (c - 1) * NPER + j ÷ thin] = z[i]
            end
        end
        isfinite(z[1]) || error("chain $c diverged at step $k")
    end
end

for set in ("train", "valid"), h in HS
    t0 = time()
    X = zeros(Float32, D, NCHAIN * NPER)
    @threads for c in 1:NCHAIN
        chain!(X, h, SEEDS[set] + 1000 * round(Int, 1 / h) + c, c)
    end
    file = joinpath(OUT, "$(set)_h$(round(Int, 1 / h)).f32")
    write(file, X)
    logline(@sprintf("%s set, h = 1/%d: %d states in %.1f s, mean |x|^2 = %.4f; wrote %s", set, round(Int, 1 / h),
                     size(X, 2), time() - t0, sum(abs2, X) / size(X, 2), file))
end
open(joinpath(OUT, "meta.toml"), "w") do io
    TOML.print(io, Dict("noise" => String(NOISE), "d" => D, "sigma" => M.σ, "chains" => NCHAIN, "n_per_chain" => NPER,
                        "n" => NCHAIN * NPER, "dt" => DT, "burn_in" => TBURN, "h" => 1 / 16, "h_coarse" => 1 / 8,
                        "source" => "sglm_states.jl: train and valid sets from independent chains"))
end
logline("wrote $(joinpath(OUT, "meta.toml"))")
