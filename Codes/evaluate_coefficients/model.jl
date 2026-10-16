# The 4D Gaussian mixture of model.toml: potential U = -log pi and its derivatives, the test
# functions with their exact expectations, the two gradient noises and their covariances.
# Vectors are NTuple{4,Float64}, so evaluations do not allocate.

module Model

using LinearAlgebra, TOML, Random

export V4, Mixture, TestFn, load_model, potential, hessian_quad, hessian, evalf, exact_mean,
    noise1, noise2, F2, U_STAR

const V4 = NTuple{4,Float64}

@inline dot4(a::V4, b::V4) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4]
@inline mul4(M::NTuple{4,V4}, v::V4) = (dot4(M[1], v), dot4(M[2], v), dot4(M[3], v), dot4(M[4], v))
@inline sub4(a::V4, b::V4) = (a[1] - b[1], a[2] - b[2], a[3] - b[3], a[4] - b[4])
@inline add4(a::V4, b::V4) = (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
@inline scal4(s, a::V4) = (s * a[1], s * a[2], s * a[3], s * a[4])

tuple4(v) = (Float64(v[1]), Float64(v[2]), Float64(v[3]), Float64(v[4]))
rows4(M::AbstractMatrix) = ntuple(i -> tuple4(M[i, :]), 4)

struct Mixture
    w::Vector{Float64}
    mu::Vector{V4}
    Sigma::Vector{Matrix{Float64}}
    P::Vector{NTuple{4,V4}}          # precision matrices (symmetric, stored by rows)
    trP::Vector{Float64}
    logc::Vector{Float64}            # log w_k - log sqrt(det(2 pi Sigma_k))
    chol::Vector{NTuple{4,V4}}       # lower Cholesky factor of Sigma_k, by rows
end

struct TestFn
    name::String
    kind::Symbol                     # :linear, :cos, :bump
    a::V4
    phase::Float64
    width::Float64
end

struct Noise
    sigma1_sq::Float64
    s2::Float64
    kappa::Float64
end

function load_model(path::AbstractString)
    t = TOML.parsefile(path)
    mx = t["mixture"]
    w = Float64.(mx["weights"])
    mu = [tuple4(m) for m in mx["means"]]
    Sigma = [Matrix{Float64}(reduce(hcat, [Float64.(r) for r in S])') for S in mx["covariances"]]
    P = [rows4(inv(S)) for S in Sigma]
    trP = [tr(inv(S)) for S in Sigma]
    logc = [log(w[k]) - 0.5 * logdet(2pi .* Sigma[k]) for k in eachindex(w)]
    chol = [rows4(Matrix(cholesky(Symmetric(S)).L)) for S in Sigma]
    mix = Mixture(w, mu, Sigma, P, trP, logc, chol)
    tests = [TestFn(s["name"], Symbol(s["type"]), tuple4(s["a"]), Float64(s["phase"]), Float64(s["width"]))
             for s in t["test"]]
    nz = t["noise"]
    noise = Noise(Float64(nz["sigma1_sq"]), Float64(nz["s2"]), Float64(nz["kappa"]))
    selected = String.(get(t, "selected", String[]))
    return mix, tests, noise, selected
end

# One pass over the components with a running log-sum-exp. Returns U = -log pi(x), grad U,
# Laplacian of U and |grad U|^2, from
#   grad U = sum_k r_k g_k,  g_k = P_k (x - mu_k),  r_k the responsibilities,
#   Hess U = sum_k r_k P_k - ( sum_k r_k g_k g_k' - grad U grad U' ).
@inline function potential(m::Mixture, x::V4)
    lmax = -Inf
    S = 0.0
    A1 = (0.0, 0.0, 0.0, 0.0)
    A2 = 0.0
    A3 = 0.0
    @inbounds for k in eachindex(m.w)
        g = mul4(m.P[k], sub4(x, m.mu[k]))
        l = m.logc[k] - 0.5 * dot4(sub4(x, m.mu[k]), g)
        if l > lmax
            c = exp(lmax - l)
            S *= c; A1 = scal4(c, A1); A2 *= c; A3 *= c
            lmax = l
        end
        e = exp(l - lmax)
        S += e
        A1 = add4(A1, scal4(e, g))
        A2 += e * dot4(g, g)
        A3 += e * m.trP[k]
    end
    gU = scal4(1 / S, A1)
    g2 = dot4(gU, gU)
    U = -(lmax + log(S))
    lapU = A3 / S - (A2 / S - g2)
    return U, gU, lapU, g2
end

# v' Hess U(x) v, for the checks of the Riesz identities
function hessian_quad(m::Mixture, x::V4, v::V4)
    lmax = maximum(m.logc[k] - 0.5 * dot4(sub4(x, m.mu[k]), mul4(m.P[k], sub4(x, m.mu[k]))) for k in eachindex(m.w))
    S = 0.0; a = 0.0; b = 0.0; c = 0.0
    for k in eachindex(m.w)
        g = mul4(m.P[k], sub4(x, m.mu[k]))
        e = exp(m.logc[k] - 0.5 * dot4(sub4(x, m.mu[k]), g) - lmax)
        S += e
        a += e * dot4(v, mul4(m.P[k], v))
        b += e * dot4(v, g)^2
        c += e * dot4(v, g)
    end
    return a / S - (b / S - (c / S)^2)
end

# full Hessian as a Matrix, for the checks only
function hessian(m::Mixture, x::V4)
    ls = [m.logc[k] - 0.5 * dot4(sub4(x, m.mu[k]), mul4(m.P[k], sub4(x, m.mu[k]))) for k in eachindex(m.w)]
    r = exp.(ls .- maximum(ls)); r ./= sum(r)
    G = [collect(mul4(m.P[k], sub4(x, m.mu[k]))) for k in eachindex(m.w)]
    gbar = sum(r[k] .* G[k] for k in eachindex(r))
    H = sum(r[k] .* inv(m.Sigma[k]) for k in eachindex(r))
    H .-= sum(r[k] .* (G[k] * G[k]') for k in eachindex(r)) .- gbar * gbar'
    return (H + H') / 2
end

@inline function evalf(f::TestFn, x::V4)
    if f.kind === :linear
        return dot4(f.a, x)
    elseif f.kind === :cos
        return cos(dot4(f.a, x) + f.phase)
    else
        y = sub4(x, f.a)
        return exp(-dot4(y, y) / (2 * f.width^2))
    end
end

# E f(X) for X ~ N(mu, Sigma)
function exact_mean(f::TestFn, mu::V4, Sigma::Matrix{Float64})
    a = collect(f.a); m = collect(mu)
    if f.kind === :linear
        return dot(a, m)
    elseif f.kind === :cos
        return cos(dot(a, m) + f.phase) * exp(-0.5 * dot(a, Sigma * a))
    else
        B = Matrix{Float64}(I, 4, 4) ./ f.width^2
        y = m .- a
        return exp(-0.5 * dot(y, (inv(B) + Sigma) \ y)) / sqrt(det(I + Sigma * B))
    end
end
exact_mean(f::TestFn, mix::Mixture) = sum(mix.w[k] * exact_mean(f, mix.mu[k], mix.Sigma[k]) for k in eachindex(mix.w))

# Reference level of noise 2: the smallest value of U at the component means
U_STAR(m::Mixture) = minimum(potential(m, m.mu[k])[1] for k in eachindex(m.w))

# Variance factor of noise 2, F2 = s2 (1 + kappa - kappa exp(-(U - U_star))); covariance F2(x) I
@inline F2(nz::Noise, U::Float64, Ustar::Float64) = nz.s2 * (1 + nz.kappa - nz.kappa * exp(-(U - Ustar)))

# Noise 1: eta = sigma1 xi, xi ~ N(0, I); the gradient of the random potential sigma1 xi.x
@inline noise1(rng::AbstractRNG, nz::Noise) = scal4(sqrt(nz.sigma1_sq), (randn(rng), randn(rng), randn(rng), randn(rng)))

# Noise 2: eta = sqrt(F2) R(x) zeta, with zeta_j = E_j - 1, E_j ~ Exp(1) iid (mean 0, variance 1,
# skewness 2), and R(x) the rotation by x1 + x2 in the (1,2)-plane and by x3 - x4 in the
# (3,4)-plane; covariance F2(x) I
@inline function noise2(rng::AbstractRNG, nz::Noise, x::V4, U::Float64, Ustar::Float64)
    z1 = randexp(rng) - 1; z2 = randexp(rng) - 1; z3 = randexp(rng) - 1; z4 = randexp(rng) - 1
    s1, c1 = sincos(x[1] + x[2])
    s2, c2 = sincos(x[3] - x[4])
    a = sqrt(F2(nz, U, Ustar))
    return (a * (c1 * z1 - s1 * z2), a * (s1 * z1 + c1 * z2), a * (c2 * z3 - s2 * z4), a * (s2 * z3 + c2 * z4))
end

end
