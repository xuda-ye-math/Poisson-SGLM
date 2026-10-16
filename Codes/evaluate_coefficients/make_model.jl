# Writes model.toml: the 4D Gaussian mixture pi = sum_k w_k N(mu_k, Sigma_k), the parameters
# of the two gradient noises, and the pool of test functions. model.toml is then the single
# source of the parameters for every other script (Julia and Python); this generator is kept
# only as the record of how the numbers were drawn.
#
#   julia make_model.jl

using LinearAlgebra, Random, Printf

const OUT = joinpath(@__DIR__, "model.toml")
isfile(OUT) && error("$OUT exists; it is the source of every result, delete it by hand to regenerate")

rng = Xoshiro(20261011)
d, K = 4, 5

# uneven weights
weights = [0.34, 0.25, 0.19, 0.13, 0.09]

# means: vertices of a regular simplex in R^4 with edge D, rotated at random and jittered
D = 4.5
V = (Matrix{Float64}(I, K, K) .- 1 / K) * nullspace(ones(1, K))     # K x d, pairwise distance sqrt(2)
Q = Matrix(qr(randn(rng, d, d)).Q)
means = [(D / sqrt(2)) .* (Q * V[k, :]) .+ 0.3 .* randn(rng, d) for k in 1:K]

# covariances: random orientation, eigenvalues uniform in [0.6, 1.6]
covs = map(1:K) do k
    Qk = Matrix(qr(randn(rng, d, d)).Q)
    S = Qk * Diagonal(0.6 .+ rand(rng, d)) * Qk'
    (S + S') / 2
end

unit(v) = v ./ norm(v)

# pool of test functions with closed-form Gaussian expectations and bounded derivatives
pool = Any[]
push!(pool, ("linear_1", "linear", unit(randn(rng, d)), 0.0, 0.0))
push!(pool, ("linear_2", "linear", unit(randn(rng, d)), 0.0, 0.0))
for (i, r) in enumerate((0.6, 1.0, 1.4))
    push!(pool, ("cos_$i", "cos", r .* unit(randn(rng, d)), 2pi * rand(rng), 0.0))
end
for k in 1:K
    push!(pool, ("bump_well_$k", "bump", means[k], 0.0, 1.0))
end
for (i, j) in ((1, 5), (2, 3))
    push!(pool, ("bump_mid_$(i)$(j)", "bump", (means[i] .+ means[j]) ./ 2, 0.0, 1.0))
end

vec_str(v) = "[" * join((@sprintf("%.15e", x) for x in v), ", ") * "]"

open(OUT, "w") do io
    println(io, "# 4D Gaussian mixture, gradient noises and test functions; written by make_model.jl")
    println(io, "dim = $d")
    println(io)
    println(io, "[mixture]")
    println(io, "weights = ", vec_str(weights))
    println(io, "means = [")
    for m in means
        println(io, "  ", vec_str(m), ",")
    end
    println(io, "]")
    println(io, "covariances = [")
    for S in covs
        println(io, "  [", join((vec_str(S[i, :]) for i in 1:d), ", "), "],")
    end
    println(io, "]")
    println(io)
    println(io, "# noise 1: eta = sigma1 * xi, xi ~ N(0, I); covariance sigma1^2 I")
    println(io, "# noise 2: eta = sqrt(F2(x)) R(x) zeta, zeta_j = E_j - 1, E_j ~ Exp(1) iid;")
    println(io, "#          F2 = s2 (1 + kappa - kappa exp(-(U - U_star))), covariance F2 I")
    println(io, "[noise]")
    println(io, "sigma1_sq = 1.0")
    println(io, "s2 = 1.0")
    println(io, "kappa = 1.0")
    println(io)
    println(io, "# type linear: f = a.x;  cos: f = cos(a.x + phase);  bump: f = exp(-|x - a|^2 / (2 width^2))")
    for (name, typ, a, phase, width) in pool
        println(io, "[[test]]")
        println(io, "name = \"$name\"")
        println(io, "type = \"$typ\"")
        println(io, "a = ", vec_str(a))
        println(io, "phase = ", @sprintf("%.15e", phase))
        println(io, "width = ", @sprintf("%.15e", width))
    end
end
println("wrote $OUT")
