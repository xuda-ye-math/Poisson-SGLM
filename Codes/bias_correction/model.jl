# High-dimensional test problem for the Riesz representor psi_C: a rotated product of 2D rings with Gaussian
# gradient noise.
#
#   target    pi ∝ exp(-U),  U(x) = sum_i u_i(|y^(i)|),  y = Q x,  y^(i) = (y_{2i-1}, y_{2i}),  i = 1, ..., d/2,
#             u_i(r) = r^2/2 + beta_i exp(-r^2/2), a ring of minima at r = sqrt(2 log beta_i) for beta_i > 1
#   noise     gaussian:    grad U(x) + sigma xi, xi ~ N(0, I_d), so C = sigma^2 I_d and psi_C = sigma^2 (U - pi(U))
#             nongaussian: grad U(x) + sigma R(x) zeta, zeta_k = E_k - 1 with E_k ~ Exp(1) independent (mean 0,
#                          variance 1, skewness 2), R(x) the rotation by x_{2i-1} + x_{2i} in each coordinate plane
#                          (x_{2i-1}, x_{2i}); the covariance is again C = sigma^2 I_d
#
# Q is a random orthogonal matrix, beta_i ~ U[2, 4] and v_k random unit vectors (stored, not used by the Gaussian
# noise), drawn from the seed of the dimension; make_model writes them to model_dD.toml and load_model reads them.

module HighDim

using LinearAlgebra, Random, TOML

export Model, make_model, save_model, load_model, potential, grad!, add_noise!, NF, TESTNAMES, testfuns!,
       RingSampler, sample_exact!

struct Model
    d::Int
    seed::Int
    σ::Float64
    Q::Matrix{Float64}     # y = Q x
    β::Vector{Float64}     # ring parameter of each pair
    V::Matrix{Float64}     # row k: unit vector v_k
end

function make_model(d; seed=1000 + d, σ=1.0)
    iseven(d) || throw(ArgumentError("d must be even"))
    rng = Xoshiro(seed)
    F = qr(randn(rng, d, d))
    Q = Matrix(F.Q) * Diagonal(sign.(diag(F.R)))          # Haar-distributed orthogonal matrix
    β = 2 .+ 2 .* rand(rng, d ÷ 2)
    V = randn(rng, d, d)
    V ./= sqrt.(sum(abs2, V; dims=2))
    return Model(d, seed, σ, Q, β, V)
end

function save_model(path, m::Model)
    open(path, "w") do io
        TOML.print(io, Dict("d" => m.d, "seed" => m.seed, "sigma" => m.σ, "beta" => m.β,
                            "Q" => [m.Q[i, :] for i in 1:m.d], "V" => [m.V[i, :] for i in 1:m.d]))
    end
end

function load_model(path)
    t = TOML.parsefile(path)
    rows(key) = permutedims(reduce(hcat, Vector{Float64}.(t[key])))
    return Model(t["d"], t["seed"], t["sigma"], rows("Q"), Vector{Float64}(t["beta"]), rows("V"))
end

"U(x); y is a work vector of length d"
function potential(m::Model, x, y)
    mul!(y, m.Q, x)
    U = 0.0
    @inbounds for i in eachindex(m.β)
        r2 = y[2i-1]^2 + y[2i]^2
        U += r2 / 2 + m.β[i] * exp(-r2 / 2)
    end
    return U
end

"g = ∇U(x); y is a work vector of length d"
function grad!(g, y, m::Model, x)
    mul!(y, m.Q, x)
    @inbounds for i in eachindex(m.β)
        fac = 1 - m.β[i] * exp(-(y[2i-1]^2 + y[2i]^2) / 2)
        y[2i-1] *= fac
        y[2i] *= fac
    end
    mul!(g, transpose(m.Q), y)
    return g
end

"g += the gradient noise at x; noise is :gaussian or :nongaussian (see the header); w is a work vector"
function add_noise!(g, w, rng, m::Model, x, noise::Symbol)
    if noise === :gaussian
        @inbounds for k in eachindex(g)
            g[k] += m.σ * randn(rng)
        end
    elseif noise === :nongaussian
        @inbounds for i in 1:(length(g) ÷ 2)
            ζ1, ζ2 = randexp(rng) - 1, randexp(rng) - 1
            c, s = cos(x[2i-1] + x[2i]), sin(x[2i-1] + x[2i])
            g[2i-1] += m.σ * (c * ζ1 - s * ζ2)
            g[2i] += m.σ * (s * ζ1 + c * ζ2)
        end
    else
        throw(ArgumentError("unknown noise $noise"))
    end
    return g
end

# twelve test functions, written the same way for every d >= 4
const NF = 12
const TESTNAMES = ["x1", "sum(x)/sqrt(d)", "cos(x1)", "sin(x1+x2)", "sqrt(1+x1^2)", "exp(-(x1^2+x2^2)/2)",
                   "log(1+exp(x1-x2))", "sqrt(1+|x|^2)", "cos(x1+x2)", "sqrt(1+x1^2+x2^2)",
                   "x1*x2/sqrt(1+x1^2+x2^2)", "log(1+x1^2+...+x4^2)"]

function testfuns!(fv, x)
    @inbounds begin
        x1, x2 = x[1], x[2]
        fv[1] = x1
        fv[2] = sum(x) / sqrt(length(x))
        fv[3] = cos(x1)
        fv[4] = sin(x1 + x2)
        fv[5] = sqrt(1 + x1^2)
        fv[6] = exp(-(x1^2 + x2^2) / 2)
        z = x1 - x2
        fv[7] = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
        fv[8] = sqrt(1 + sum(abs2, x))
        q2 = x1^2 + x2^2
        fv[9] = cos(x1 + x2)
        fv[10] = sqrt(1 + q2)
        fv[11] = x1 * x2 / sqrt(1 + q2)
        fv[12] = log(1 + q2 + x[3]^2 + x[4]^2)
    end
    return fv
end

"exact sampler: radius of each pair from r exp(-u_i(r)) by inverse CDF on a fine grid, uniform angle, x = Q^T y"
struct RingSampler
    r::Vector{Float64}
    cdf::Vector{Vector{Float64}}
end

function RingSampler(m::Model; rmax=10.0, n=200_001)
    r = collect(range(0.0, rmax; length=n))
    cdf = map(m.β) do β
        p = @. r * exp(-(r^2 / 2 + β * exp(-r^2 / 2)))
        c = cumsum(vcat(0.0, (p[1:end-1] .+ p[2:end]) ./ 2))
        c ./ c[end]
    end
    return RingSampler(r, cdf)
end

"x ~ pi exactly; y is a work vector"
function sample_exact!(x, y, rng, s::RingSampler, m::Model)
    @inbounds for i in eachindex(m.β)
        c = s.cdf[i]
        u = rand(rng)
        j = clamp(searchsortedfirst(c, u), 2, length(c))
        t = (u - c[j-1]) / (c[j] - c[j-1])
        ρ = s.r[j-1] + t * (s.r[j] - s.r[j-1])
        θ = 2π * rand(rng)
        y[2i-1] = ρ * cos(θ)
        y[2i] = ρ * sin(θ)
    end
    mul!(x, transpose(m.Q), y)
    return x
end

end # module
