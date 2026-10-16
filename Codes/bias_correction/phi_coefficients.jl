# Coefficients and biases implied by the trained phi (train_phi.py), on the validation states at h = 1/16, which the
# training did not use; reference means pi(f) from exact_means.jl (exact samples only).
#
#   julia -t 32 phi_coefficients.jl [noise=gaussian] [epoch]   > results/phi_coefficients_NOISE.log
#
# Reads results/states_NOISE/{train,valid}_h{16,8}.f32 (sglm_states.jl), results/grad_phi_NOISE.f32 (or _eEEE,
# eval_phi.py) and results/exact_means.csv. With T(x) = x + h grad phi(x), h = 1/16, and the chain means of each set:
#   data_train      (pi_{2h}(f) - pi_h(f)) / h on the training states: the extrapolation term the training saw
#   data_valid      the same on the validation states
#   implied         -(pi_h(f o T) - pi_h(f)) / h on the validation states at h
#   bias_sglm       pi_h(f) - pi(f)                    on the validation states
#   bias_richardson 2 pi_h(f) - pi_{2h}(f) - pi(f)     on the validation states
#   bias_shift      pi_h(f o T) - pi(f)                on the validation states at h
# Standard errors over the 32 chains of each set. Writes results/phi_NOISE.csv (or _eEEE).

include("model.jl")
using .HighDim
using Statistics, Printf, TOML, Base.Threads

const NOISE = length(ARGS) >= 1 ? ARGS[1] : "gaussian"
const SUFFIX = length(ARGS) >= 2 ? @sprintf("_e%03d", parse(Int, ARGS[2])) : ""
const DATA = joinpath(@__DIR__, "results", "states_$(NOISE)")
const META = TOML.parsefile(joinpath(DATA, "meta.toml"))
const D, N, NCHAIN, H = META["d"], META["n"], META["chains"], META["h"]
const PER = N ÷ NCHAIN

function load(path)
    X = Array{Float32}(undef, D, N)
    read!(path, X)
    return X
end

"chain means of the test functions at the states x + h g (g = 0 when G is nothing): NF x NCHAIN"
function chain_means(X, G)
    S = zeros(NF, NCHAIN)
    @threads for c in 1:NCHAIN
        fv, x, acc = zeros(NF), zeros(D), zeros(NF)
        for n in ((c - 1) * PER + 1):(c * PER)
            @inbounds for i in 1:D
                x[i] = X[i, n] + (G === nothing ? 0.0 : H * G[i, n])
            end
            testfuns!(fv, x)
            acc .+= fv
        end
        S[:, c] = acc ./ PER
    end
    return S
end

mse(S) = (vec(mean(S; dims=2)), vec(std(S; dims=2)) ./ sqrt(NCHAIN))
diff_se(a, b) = sqrt.(a .^ 2 .+ b .^ 2)

ref = Dict(r[1] => parse(Float64, r[2]) for r in (split(l, ",") for l in readlines(joinpath(@__DIR__, "results",
           "exact_means.csv"))[2:end]))
piex = [ref["f$j"] for j in 1:NF]

t0 = time()
th, se_th = mse(chain_means(load(joinpath(DATA, "train_h16.f32")), nothing))
t2h, se_t2h = mse(chain_means(load(joinpath(DATA, "train_h8.f32")), nothing))
V = load(joinpath(DATA, "valid_h16.f32"))
G = load(joinpath(@__DIR__, "results", "grad_phi_$(NOISE)$(SUFFIX).f32"))
Sv, Ss = chain_means(V, nothing), chain_means(V, G)
vh, se_vh = mse(Sv)
vs, se_vs = mse(Ss)
imp, se_imp = mse(-(Ss .- Sv) ./ H)
v2h, se_v2h = mse(chain_means(load(joinpath(DATA, "valid_h8.f32")), nothing))
println(@sprintf("chain means of %d test functions on 4 sets of %d states and on the shifted validation states in %.1f s",
                 NF, N, time() - t0))

cols = [("data_train", (t2h .- th) ./ H, diff_se(se_t2h, se_th) ./ H),
        ("data_valid", (v2h .- vh) ./ H, diff_se(se_v2h, se_vh) ./ H),
        ("implied", imp, se_imp),
        ("bias_sglm", vh .- piex, se_vh),
        ("bias_richardson", 2 .* vh .- v2h .- piex, diff_se(2 .* se_vh, se_v2h)),
        ("bias_shift", vs .- piex, se_vs)]
println("coefficients (first three columns) and biases at h = 1/16 (last three), ± standard error over the chains:")
println(@sprintf("  %-4s", "") * join([@sprintf("%22s", c[1]) for c in cols]))
for j in 1:NF
    println(@sprintf("  f%-3d", j) * join([@sprintf("%+12.5f ± %.5f", c[2][j], c[3][j]) for c in cols]))
end
out = joinpath(@__DIR__, "results", "phi_$(NOISE)$(SUFFIX).csv")
open(out, "w") do io
    println(io, "test," * join(["$(c[1]),$(c[1])_se" for c in cols], ","))
    for j in 1:NF
        println(io, "f$j," * join([@sprintf("%.10e,%.3e", c[2][j], c[3][j]) for c in cols], ","))
    end
end
println("wrote $out")
