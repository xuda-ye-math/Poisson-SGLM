# W_1(π̄_h, π) of SGLD and W_1(π_h, π) of SGLM at a fixed step size h, for the target π ∝ exp(-U) on R^d
# with U(x) = u(|x|), u(r) = (r^2 - r) r^2 / (r^2 + 1/4) (quadratic growth, non-convex), and the
# stochastic gradient ∇U(x) + σ ξ, ξ ~ N(0, I_d).
#
#   include("wasserstein.jl"); using .Wasserstein
#   w1_sgld(d, h, σ)    W_1(π̄_h, π), π̄_h the invariant distribution of SGLD
#   w1_sglm(d, h, σ)    W_1(π_h, π),  π_h the invariant distribution of SGLM
#
# The method is described in README.md; check_wasserstein.jl tests it.

module Wasserstein

using LinearAlgebra, SpecialFunctions, FastGaussQuadrature

export w1_sgld, w1_sglm, Potential, RING

# fixed numerical parameters; check_wasserstein.jl compares them with a refinement by the factor 2
const NODES = 12          # Gauss-Legendre nodes per panel of the radial grid
const PANEL = 0.5         # panel width in units of the narrowest kernel scale (s for SGLD, a for SGLM)
const PANEL_MAX = 0.1     # largest panel width, which resolves the radial density of π
const TAIL = 50.0         # the grid ends where the radial density of π has dropped by the factor e^-TAIL
const SPREAD = 10.0       # a Gaussian vector of scale c is taken within c (√d + SPREAD) of its mean
const ANGLE_PANELS = 4    # composite Gauss-Legendre rule of the angular integral of SGLM
const ANGLE_NODES = 16
const SUB = 0.5           # sub-interval width of the smoothing of scale hσ, in units of hσ
const SUB_NODES = 8
const H_MAX = 0.5         # largest step size accepted

# ---------------------------------------------------------------------------------------------
# radial potential: u, u', u'' and u'/r (finite at r = 0)

struct Potential{F1,F2,F3,F4}
    u::F1
    du::F2
    d2u::F3
    du_over_r::F4
end

const C4 = 0.25
ring_u(r) = (r^4 - r^3) / (r^2 + C4)
ring_du_over_r(r) = (2r^4 - r^3 + 4C4 * r^2 - 3C4 * r) / (r^2 + C4)^2
ring_du(r) = r * ring_du_over_r(r)
function ring_d2u(r)
    D = r^2 + C4
    N = 2r^5 - r^4 + 4C4 * r^3 - 3C4 * r^2
    Np = 10r^4 - 4r^3 + 12C4 * r^2 - 6C4 * r
    return Np / D^2 - 4r * N / D^3
end
const RING = Potential(ring_u, ring_du, ring_d2u, ring_du_over_r)

"W_1(π̄_h, π) for SGLD with step size h and gradient noise σ N(0, I_d)"
w1_sgld(d, h, σ) = w1_sgld(RING, d, h, σ)

"W_1(π_h, π) for SGLM with step size h and gradient noise σ N(0, I_d)"
w1_sglm(d, h, σ) = w1_sglm(RING, d, h, σ)

# ---------------------------------------------------------------------------------------------
# radial grid of composite Gauss-Legendre panels on [0, R]

struct Grid
    r::Vector{Float64}
    w::Vector{Float64}
    P::Int
    D::Float64
    x::Vector{Float64}   # reference nodes on [-1, 1]
end

function Grid(R, width; q=NODES)
    P = max(1, ceil(Int, R / width))
    D = R / P
    x, wx = gausslegendre(q)
    r = vec([D * p + D / 2 * (xi + 1) for xi in x, p in 0:P-1])
    w = repeat(D / 2 .* wx, P)
    return Grid(r, w, P, D, x)
end

# Legendre polynomials P_0, ..., P_n at x
function legendre_row(x, n)
    v = zeros(n + 1)
    v[1] = 1.0
    n >= 1 && (v[2] = x)
    for k in 1:n-1
        v[k+2] = ((2k + 1) * x * v[k+1] - k * v[k]) / (k + 1)
    end
    return v
end

legendre_vander(xs, n) = permutedims(reduce(hcat, [legendre_row(x, n) for x in xs]))

legendre_eval(c, x) = dot(c, legendre_row(x, length(c) - 1))

# coefficients of the antiderivative of Σ c_n P_n that vanishes at x = -1
function legendre_int(c)
    n = length(c)
    b = zeros(n + 1)
    b[2] += c[1]
    for k in 1:n-1          # ∫ P_k = (P_{k+1} - P_{k-1}) / (2k + 1)
        b[k+2] += c[k+1] / (2k + 1)
        b[k] -= c[k+1] / (2k + 1)
    end
    b[1] -= sum(b[k+1] * (-1)^k for k in 0:n)
    return b
end

# monomial coefficients of P_0, ..., P_n: M[k+1, j+1] = coefficient of x^j in P_k
function legendre_monomial(n)
    M = zeros(n + 1, n + 1)
    M[1, 1] = 1.0
    n >= 1 && (M[2, 2] = 1.0)
    for k in 1:n-1
        M[k+2, 2:end] .+= (2k + 1) / (k + 1) .* M[k+1, 1:end-1]
        M[k+2, :] .-= k / (k + 1) .* M[k, :]
    end
    return M
end

# real roots in (-1, 1) of Σ c_n P_n
function legendre_roots(c, Mono)
    a = Mono[1:length(c), 1:length(c)]' * c        # monomial coefficients, a[j+1] of x^j
    scale = maximum(abs, a)
    scale == 0 && return Float64[]
    m = findlast(v -> abs(v) > 1e-14 * scale, a) - 1   # degree
    m <= 0 && return Float64[]
    comp = zeros(m, m)                              # companion matrix
    comp[1, :] .= -a[m:-1:1] ./ a[m+1]
    for i in 2:m
        comp[i, i-1] = 1.0
    end
    z = eigvals(comp)
    return sort([real(v) for v in z if abs(imag(v)) < 1e-9 && abs(real(v)) < 1])
end

"integral over [0, R] of |G|, G(r) = integral of g over [0, r], with g given at the grid nodes"
function w1_from_nodes(g, grid::Grid)
    q = length(grid.x)
    Vinv = inv(legendre_vander(grid.x, q - 1))
    Mono = legendre_monomial(q + 1)
    total, G0 = 0.0, 0.0
    for p in 1:grid.P
        c = Vinv * g[(p-1)*q+1:p*q]
        cG = legendre_int(c) .* (grid.D / 2)     # G on the panel, in the reference variable
        cG[1] += G0
        H = legendre_int(cG)
        pts = vcat(-1.0, legendre_roots(cG, Mono), 1.0)
        Hv = [legendre_eval(H, t) for t in pts]
        total += grid.D / 2 * sum(abs, diff(Hv))
        G0 = legendre_eval(cG, 1.0)
    end
    return total
end

# ---------------------------------------------------------------------------------------------
# noncentral chi densities, the radial density of π, the stationary solve

"log of e^{-z} I_ν(z) for z > 0, also where besselix underflows"
function log_ive(ν, z)
    v = besselix(ν, z)
    v > 1e-280 && return log(v)
    return ν * log(z / 2) - loggamma(ν + 1) - z + log1p(z^2 / (4(ν + 1)) + z^4 / (32(ν + 1) * (ν + 2)))
end

"log density at x > 0 of |λ e_1 + s η|, η ~ N(0, I_d)"
function log_nchi(x, λ, s, d)
    λ = abs(λ)
    if d == 1      # folded normal
        return -(x - λ)^2 / (2s^2) + log1p(exp(-2x * λ / s^2)) - log(s * sqrt(2π))
    end
    ν = d / 2 - 1
    z = x * λ / s^2
    if z > 1e-8
        return log(x) - 2log(s) + ν * (log(x) - log(λ)) - (x - λ)^2 / (2s^2) + log_ive(ν, z)
    end
    # z -> 0: (x / λ)^ν I_ν(z) -> (x^2 / (2 s^2))^ν / Γ(ν + 1)
    return log(x) - 2log(s) + 2ν * log(x) - ν * log(2s^2) - loggamma(ν + 1) - (x^2 + λ^2) / (2s^2)
end

"end of the radial grid: the tail of π, widened by the variance ratio of EM for u ~ r^2"
function radius(pot, d, h, σ)
    r = range(1e-6, 100.0; length=400_001)
    lp = (d - 1) .* log.(r) .- pot.u.(r)
    i = argmax(lp)
    j = findfirst(k -> lp[k] < lp[i] - TAIL, i:length(r))
    return r[i+j-1] * sqrt((2 + h * σ^2) / (2 * (1 - h)))
end

function pi_density(pot, grid::Grid, d)
    lp = (d - 1) .* log.(grid.r) .- pot.u.(grid.r)
    ρ = exp.(lp .- maximum(lp))
    return ρ ./ dot(grid.w, ρ)
end

"""ρ at the nodes with ρ(r_j) = Σ_i w_i ρ(r_i) K[i, j] and Σ_i w_i ρ(r_i) = 1

The rows of K are normalized to unit mass, which absorbs the mass lost beyond R and the quadrature
error of the mass. The discrete kernel then conserves mass, wᵀ(I - A) = 0 with A[j, i] = w_i K[i, j],
so the solution of (I - A + 1 wᵀ) ρ = 1 satisfies wᵀρ = 1 and (I - A) ρ = 0, every node's equation kept.
"""
function stationary(K, w)
    N = length(w)
    Kn = K ./ (K * w)
    A = Matrix{Float64}(I, N, N) .- (Kn .* w)' .+ ones(N) * w'
    ρ = A \ ones(N)
    return ρ ./ dot(w, ρ)
end

function check_tail(ρ; q=NODES)
    tail = maximum(abs, ρ[end-q+1:end])
    tail > 1e-14 * maximum(ρ) && error("radial density not negligible at the end of the grid ($tail)")
end

check_step(h) = 0 < h <= H_MAX || throw(ArgumentError("step size h = $h outside (0, $H_MAX]"))

# ---------------------------------------------------------------------------------------------
# SGLD: |Z_{k+1}| given |Z_k| = r is noncentral chi with noncentrality |r - h u'(r)|, scale s

function w1_sgld(pot::Potential, d, h, σ; refine=1)
    check_step(h)
    s = sqrt(2h + (h * σ)^2)
    grid = Grid(radius(pot, d, h, σ), min(PANEL * s, PANEL_MAX) / refine)
    r, N = grid.r, length(grid.r)
    K = zeros(N, N)
    Threads.@threads for i in 1:N
        λ = r[i] - h * pot.du(r[i])
        for j in 1:N
            K[i, j] = exp(log_nchi(r[j], λ, s, d))
        end
    end
    ρh = stationary(K, grid.w)
    check_tail(ρh)
    return w1_from_nodes(ρh .- pi_density(pot, grid, d), grid)
end

# ---------------------------------------------------------------------------------------------
# SGLM: Markov form Y_{k+1} = Y_k - h ∇U(Y_k + a χ_k) - hσ ξ_k + 2a χ_k, a = √(h/2), Z_k = Y_k + a χ_k

"f(ω) = 2ω - h u'(ω), the radial profile of F(w) = 2w - h ∇U(w), with its inverse by table and Newton"
struct RadialMap{P}
    pot::P
    h::Float64
    om::Vector{Float64}
    ftab::Vector{Float64}
    lip::Float64          # Lipschitz constant of F
end

fmap(m::RadialMap, ω) = 2ω - m.h * m.pot.du(ω)
dfmap(m::RadialMap, ω) = 2 - m.h * m.pot.d2u(ω)

function RadialMap(pot, h, ρmax)
    om = collect(range(0.0, ρmax / (2 * (1 - h)) + 1.0; length=400_001))
    f = 2 .* om .- h .* pot.du.(om)
    df = 2 .- h .* pot.d2u.(om)
    tang = 2 .- h .* pot.du_over_r.(om)
    (any(df .<= 0) || any(tang .<= 0) || f[end] < ρmax) &&
        throw(ArgumentError("F(w) = 2w - h∇U(w) is not a diffeomorphism for h = $h"))
    return RadialMap(pot, h, om, f, max(maximum(df), maximum(tang)))
end

function finv(m::RadialMap, ρ)
    k = clamp(searchsortedlast(m.ftab, ρ), 1, length(m.ftab) - 1)
    ω = m.om[k] + (ρ - m.ftab[k]) * (m.om[k+1] - m.om[k]) / (m.ftab[k+1] - m.ftab[k])
    for _ in 1:2
        ω -= (fmap(m, ω) - ρ) / dfmap(m, ω)
    end
    return ω
end

function angular_rule(refine)
    x, wx = gausslegendre(ANGLE_NODES * refine)
    n = ANGLE_PANELS
    X = vec([(p + (xi + 1) / 2) / n for xi in x, p in 0:n-1])
    W = repeat(wx ./ (2n), n)
    return X, W   # rule on [0, 1]
end

"log density of V = F(W) - y, W ~ N(y, a^2 I), |y| = r, at |v| = t and angle θ between v and y"
function log_pV(m::RadialMap, d, a, r, t, θ)
    X1 = t * cos(θ) + r
    X2 = t * sin(θ)
    g = finv(m, hypot(X1, X2))
    tang = 2 - m.h * m.pot.du_over_r(g)      # f(g) / g
    e1, e2 = X1 / tang - r, X2 / tang
    return -d / 2 * log(2π * a^2) - (e1^2 + e2^2) / (2a^2) - log(dfmap(m, g)) - (d - 1) * log(tang)
end

"Q[i, k]: density at t = r_k of |V|, V = F(W) - y, W ~ N(y, a^2 I), |y| = r_i"
function radial_V(pot, d, h, a, r, refine)
    N = length(r)
    m = r .- h .* pot.du.(r)                      # F(y) - y = m e_1
    M = RadialMap(pot, h, 2r[end] + 2maximum(abs, m))
    ball = M.lip * a * (sqrt(d) + SPREAD)          # |V - m e_1| <= ball
    X, W = angular_rule(refine)
    log_sphere = d >= 2 ? log(2) + (d - 1) / 2 * log(π) - loggamma((d - 1) / 2) : 0.0
    Q = zeros(N, N)
    Threads.@threads for i in 1:N
        mi = m[i]
        mm = abs(mi)
        for k in searchsortedfirst(r, mm - ball):searchsortedlast(r, mm + ball)
            t = r[k]
            if d == 1
                Q[i, k] = exp(log_pV(M, d, a, r[i], t, 0.0)) + exp(log_pV(M, d, a, r[i], t, Float64(π)))
                continue
            end
            cap = mm > 0 ? acos(clamp((t^2 + mm^2 - ball^2) / (2t * mm), -1.0, 1.0)) : Float64(π)
            acc = 0.0
            for (xθ, wθ) in zip(X, W)
                θ = cap * xθ
                θ = mi < 0 ? π - θ : θ
                acc += cap * wθ * exp(log_pV(M, d, a, r[i], t, θ)) * sin(θ)^(d - 2)
            end
            Q[i, k] = exp(log_sphere + (d - 1) * log(t)) * acc
        end
    end
    return Q
end

"M[k, j] = ∫ l_k(t) nchi(r_j; t, b) dt, l_k the Lagrange basis of node k in its panel"
function smoothing(b, d, grid::Grid, refine)
    r, P, D, q = grid.r, grid.P, grid.D, length(grid.x)
    nsub = max(1, ceil(Int, D / (SUB * b / refine)))
    xs, ws = gausslegendre(SUB_NODES)
    Xi = vec([(2j + xi + 1) / nsub - 1 for xi in xs, j in 0:nsub-1])
    Wi = repeat(ws ./ nsub, nsub)
    Bv = legendre_vander(Xi, q - 1) * inv(legendre_vander(grid.x, q - 1))   # l_k at the sub-nodes
    reach = b * (sqrt(d) + SPREAD)
    M = zeros(length(r), length(r))
    Threads.@threads for p in 0:P-1
        t = D * p .+ D / 2 .* (Xi .+ 1)
        wt = D / 2 .* Wi
        for j in searchsortedfirst(r, D * p - reach):searchsortedlast(r, D * (p + 1) + reach)
            for (m, tm) in enumerate(t)
                c = wt[m] * exp(log_nchi(r[j], tm, b, d))
                for k in 1:q
                    M[p*q+k, j] += Bv[m, k] * c
                end
            end
        end
    end
    return M
end

function w1_sglm(pot::Potential, d, h, σ; refine=1)
    check_step(h)
    a, b = sqrt(h / 2), h * σ
    grid = Grid(radius(pot, d, h, σ), min(PANEL * a, PANEL_MAX) / refine)
    r, N = grid.r, length(grid.r)
    K = radial_V(pot, d, h, a, r, refine)
    b > 0 && (K = K * smoothing(b, d, grid, refine))
    ρy = stationary(K, grid.w)
    check_tail(ρy)
    ρz = zeros(N)
    Threads.@threads for j in 1:N
        ρz[j] = sum(grid.w[i] * ρy[i] * exp(log_nchi(r[j], r[i], a, d)) for i in 1:N)
    end
    check_tail(ρz)
    return w1_from_nodes(ρz .- pi_density(pot, grid, d), grid)
end

end # module
