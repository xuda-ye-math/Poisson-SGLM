# Wasserstein-1 coefficients of SGLD and SGLM on the 4D mixture of model.toml,
#   lim W1(pi_h, pi) / h = 1/2 ||psi pi||_KR,
# with psi = psi_C (SGLM) or psi_H + psi_C (SGLD), the Riesz representors of coefficients.jl:
#   psi_H = 1/2 |grad U|^2 - Lap U - pi(1/2 |grad U|^2 - Lap U),
#   psi_C = sigma1^2 (U - pi(U))                                          (Gaussian noise),
#   psi_C = s2 ((1 + kappa)(U - pi(U)) + 2 kappa (E - pi(E))), E = e^{-(U - U*)}  (non-Gaussian).
#
# The Kantorovich-Rubinstein norm of nu = psi pi,
#   ||nu||_KR = sup { int f dnu : f 1-Lipschitz } = min { int |m| dx : div m = -nu },
# is discretized on the box [-L, L]^4 with n cells per side: potentials f at cell centres,
# fluxes m on the faces, the constraint |forward differences of f| <= 1 at every cell, and the
# cell averages of nu. The discrete problem is solved on the GPU (w1_pdhg.jl, CUDA.jl) by the
# preconditioned primal-dual algorithm (PDHG), warm-started from the next coarser grid, until the lower bound
# <f, nu> / max|D f| of the discrete problem and the cost sum |m| of the fluxes agree within TOL
# and the residual of div m = -nu is below TOL times int |nu|; the value reported is that lower
# bound. Its accuracy is judged from the refinement of the grid and from the case
# test_gaussian (pi = N(0, I_4), psi = (|x|^2 - 4)/2), whose norm is exact, ||psi pi||_KR = E|X|/2.
#
#   julia --project=. -t 16 w1_coefficients.jl CASE N1,N2,... [theta=100] [maxit=200000] [tol=1e-4] [L=9]
#   (--project=. loads the CUDA.jl environment of this folder; the CPU threads assemble the measure)
#   CASE = test_gaussian | SGLD_gaussian | SGLM_gaussian | SGLD_nongaussian | SGLM_nongaussian
#
# Appends one row per grid to results/w1/w1_runs.csv and logs to results/w1/w1_coefficients.log;
# the potential and the fluxes of the finest grid go to results/w1/CASE_nN.f64. The final numbers are
# written to tables/w1_coefficients_gaussian.csv and tables/w1_coefficients_nongaussian.csv by w1_table.py.

include(joinpath(@__DIR__, "model.jl"))
using .Model
using LinearAlgebra, Printf, Dates, Base.Threads

const DIR = @__DIR__
const TABLES = joinpath(DIR, "tables")
const RESW1 = joinpath(DIR, "results", "w1")
mkpath(TABLES); mkpath(RESW1)

const CASE = ARGS[1]
const NS = parse.(Int, split(ARGS[2], ","))
const OPT = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS[3:end])
const THETA = parse(Float64, get(OPT, "theta", "100"))
const MAXIT = parse(Int, get(OPT, "maxit", "200000"))
const TOL = parse(Float64, get(OPT, "tol", "1e-4"))
const LBOX = parse(Float64, get(OPT, "L", "9"))
include(joinpath(@__DIR__, "w1_pdhg.jl"))
@assert CASE in ("test_gaussian", "SGLD_gaussian", "SGLM_gaussian", "SGLD_nongaussian", "SGLM_nongaussian")

const mix, tests, noise, selected = load_model(joinpath(DIR, "model.toml"))
const Ustar = U_STAR(mix)
const CASE_ID = findfirst(==(CASE), ("test_gaussian", "SGLD_gaussian", "SGLM_gaussian", "SGLD_nongaussian", "SGLM_nongaussian"))

function logline(s)
    println(s); flush(stdout)
    open(io -> println(io, s), joinpath(RESW1, "w1_coefficients.log"), "a")
end

# ---------------------------------------------------------------- the measure nu = psi pi
gauss_hermite(n) = (E = eigen(SymTridiagonal(zeros(n), sqrt.(1.0:n-1))); (E.values, E.vectors[1, :] .^ 2))

# raw representor pieces at x: (pi(x), psiH_raw, U, E) for the mixture
@inline function pieces(x::V4)
    U, gU, lapU, g2 = potential(mix, x)
    return exp(-U), 0.5 * g2 - lapU, U, exp(-(U - Ustar))
end

# pi-means of psiH_raw, U, E by Gauss-Hermite quadrature on each mixture component (mixture cases only)
function mixture_means()
    (z, w) = gauss_hermite(52)
    K = length(mix.w); nz = length(z)
    parts = zeros(3, K * nz)
    @threads for idx in 1:(K*nz)
        k = (idx - 1) ÷ nz + 1; i1 = (idx - 1) % nz + 1
        L = mix.chol[k]; s1 = 0.0; s2 = 0.0; s3 = 0.0
        for i2 in 1:nz, i3 in 1:nz, i4 in 1:nz
            zz = (z[i1], z[i2], z[i3], z[i4])
            x = ntuple(i -> mix.mu[k][i] + L[i][1] * zz[1] + L[i][2] * zz[2] + L[i][3] * zz[3] + L[i][4] * zz[4], 4)
            _, h, U, E = pieces(x)
            W = mix.w[k] * w[i1] * w[i2] * w[i3] * w[i4]
            s1 += W * h; s2 += W * U; s3 += W * E
        end
        parts[:, idx] .= (s1, s2, s3)
    end
    return vec(sum(parts, dims = 2))
end
const MEANS = CASE_ID == 1 ? zeros(3) : (logline("[$CASE] computing the pi-means of the representor pieces"); mixture_means())

# density of nu = psi pi at x
@inline function dens(x::V4)
    if CASE_ID == 1
        r2 = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
        return 0.5 * (r2 - 4) * exp(-r2 / 2) / (2pi)^2
    end
    p, h, U, E = pieces(x)
    psiC = if CASE_ID <= 3
        noise.sigma1_sq * (U - MEANS[2])
    else
        noise.s2 * ((1 + noise.kappa) * (U - MEANS[2]) + 2 * noise.kappa * (E - MEANS[3]))
    end
    psi = (CASE_ID == 2 || CASE_ID == 4) ? (h - MEANS[1]) + psiC : psiC
    return psi * p
end

# ---------------------------------------------------------------- grid
const GL = ((-sqrt(3 / 5) / 2, 5 / 18), (0.0, 8 / 18), (sqrt(3 / 5) / 2, 5 / 18))   # 3-point Gauss-Legendre on [-1/2, 1/2], weights sum 1
@inline lin(n, i1, i2, i3, i4) = i1 + n * (i2 - 1) + n * n * (i3 - 1) + n * n * n * (i4 - 1)
center(L, Δ, i) = -L + (i - 0.5) * Δ

# cell averages of nu and the integrals of |nu - average| over the cells
function assemble(n, L)
    Δ = 2L / n
    nu = zeros(n^4); dev = zeros(n)
    @threads for i4 in 1:n
        dsum = 0.0
        vals = zeros(81)
        @inbounds for i3 in 1:n, i2 in 1:n, i1 in 1:n
            c = (center(L, Δ, i1), center(L, Δ, i2), center(L, Δ, i3), center(L, Δ, i4))
            a = 0.0; q = 0
            for g1 in GL, g2 in GL, g3 in GL, g4 in GL
                x = (c[1] + g1[1] * Δ, c[2] + g2[1] * Δ, c[3] + g3[1] * Δ, c[4] + g4[1] * Δ)
                v = dens(x); W = g1[2] * g2[2] * g3[2] * g4[2]
                q += 1; vals[q] = v; a += W * v
            end
            nu[lin(n, i1, i2, i3, i4)] = a
            q = 0
            for g1 in GL, g2 in GL, g3 in GL, g4 in GL
                q += 1; dsum += g1[2] * g2[2] * g3[2] * g4[2] * abs(vals[q] - a)
            end
        end
        dev[i4] = dsum
    end
    return nu, sum(dev) * Δ^4
end

# ---------------------------------------------------------------- warm start from a coarser grid
function prolong(A, nc, L, nf, off)           # off[k] = 0.5 for centres, 1.0 for faces (in cells)
    Δc = 2L / nc; Δf = 2L / nf
    B = zeros(nf^4)
    @threads for j4 in 1:nf
        @inbounds for j3 in 1:nf, j2 in 1:nf, j1 in 1:nf
            jj = (j1, j2, j3, j4)
            x = ntuple(k -> -L + (jj[k] - 1 + off[k]) * Δf, 4)
            u = ntuple(k -> clamp((x[k] + L) / Δc - off[k] + 1, 1.0, Float64(nc)), 4)   # fractional coarse index
            i = ntuple(k -> min(floor(Int, u[k]), nc - 1), 4)
            t = ntuple(k -> u[k] - i[k], 4)
            v = 0.0
            for b in 0:15
                w = 1.0; idx = 0
                bit = ntuple(k -> (b >> (k - 1)) & 1, 4)
                for k in 1:4
                    w *= bit[k] == 1 ? t[k] : 1 - t[k]
                end
                v += w * A[lin(nc, i[1] + bit[1], i[2] + bit[2], i[3] + bit[3], i[4] + bit[4])]
            end
            B[lin(nf, j1, j2, j3, j4)] = v
        end
    end
    return B
end

# ---------------------------------------------------------------- driver
function run_grid(n, L, warm)
    Δ = 2L / n
    t0 = time()
    nu, devint = assemble(n, L)
    m0 = sum(nu) * Δ^4
    absn = sum(abs, nu)
    nu .-= (m0 / Δ^4) .* abs.(nu) ./ absn                      # exact zero mass on the grid
    scale = 1 / maximum(abs, nu)
    nuh = nu .* scale
    logline(@sprintf("[%s] n = %d, Delta = %.4f: measure assembled in %.1f s; mass before balancing %.2e, int|nu| = %.4e, int|nu - cell average| = %.4e",
        CASE, n, Δ, time() - t0, m0, absn * Δ^4, devint))
    if warm === nothing
        f = zeros(n^4); y = ntuple(_ -> zeros(n^4), 4)
    else
        (fc, yc, nc) = warm
        f = prolong(fc, nc, L, n, (0.5, 0.5, 0.5, 0.5))
        y = ntuple(k -> prolong(yc[k], nc, L, n, ntuple(j -> j == k ? 1.0 : 0.5, 4)), 4)
        for k in 1:4                                          # boundary faces carry no flux
            st = n^(k - 1)
            for I in 1:n^4
                ((I - 1) ÷ st) % n == n - 1 && (y[k][I] = 0.0)
            end
        end
    end
    d = to_gpu(f, y, nuh)
    τ = THETA * Δ / 8; σ = Δ / (2THETA)
    it = 0; lo = cost = res = NaN
    absnu = sum(abs, nu) * Δ^4
    tlast = time()
    while it < MAXIT
        pdhg_gpu!(d, n, Δ, τ, σ, 200)
        it += 200
        lo, cost, lmax, rv = discrete_bounds_gpu(d, n, Δ)
        lo *= Δ^4 / scale; cost *= Δ^4 / scale; res = rv * Δ^4 / scale
        done = abs(cost - lo) <= TOL * lo && res <= TOL * absnu
        if time() - tlast > 30 || done
            logline(@sprintf("[%s] n = %d  it %6d  %6.0f s  lower %.6f  flux cost %.6f  residual %.2e  max|Df| %.4f",
                CASE, n, it, time() - t0, lo, cost, res, lmax))
            tlast = time()
        end
        done && break
    end
    (f, y) = from_gpu(d)
    return (f = f, y = y, row = (n = n, delta = Δ, it = it, lo = lo, cost = cost, res = res, wall = time() - t0))
end

function main()
    logline("\n$(now())  $CASE  grids $(NS)  theta $THETA  maxit $MAXIT  tol $TOL  L $LBOX  threads $(nthreads())")
    CASE_ID == 1 && logline(@sprintf("exact ||psi pi||_KR = E|X|/2 = %.6f", sqrt(2) * gamma52() / 2))
    csv = joinpath(RESW1, "w1_runs.csv")
    isfile(csv) || open(io -> println(io, "case,n,L,delta,iterations,kr_norm,flux_cost,residual,coefficient,wall_seconds,date"), csv, "w")
    warm = nothing
    local res
    for n in NS
        res = run_grid(n, LBOX, warm)
        r = res.row
        open(csv, "a") do io
            @printf(io, "%s,%d,%.3f,%.6f,%d,%.8f,%.8f,%.3e,%.8f,%.1f,%s\n", CASE, r.n, LBOX, r.delta,
                r.it, r.lo, r.cost, r.res, r.lo / 2, r.wall, now())
        end
        logline(@sprintf("[%s] n = %d  ||psi pi||_KR = %.6f (flux cost %.6f, residual %.1e), coefficient 1/2 ||psi pi||_KR = %.6f  %.0f s",
            CASE, r.n, r.lo, r.cost, r.res, r.lo / 2, r.wall))
        warm = (res.f, res.y, n)
    end
    open(joinpath(RESW1, "$(CASE)_n$(NS[end]).f64"), "w") do io
        write(io, res.f); foreach(v -> write(io, v), res.y)
    end
end
gamma52() = 3 * sqrt(pi) / 4

main()
