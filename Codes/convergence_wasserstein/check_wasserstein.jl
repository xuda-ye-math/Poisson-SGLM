# Checks of wasserstein.jl, in dimensions d >= 2.
#
#   julia --project=. -t 32 check_wasserstein.jl
#
# 1. Quadratic potential u(r) = r^2/2: the invariant laws are N(0, v I) with
#    v = (2 + hσ^2) / (2 - h) (SGLD) and v = 1 + hσ^2 / (2 - h) (SGLM), so W_1 = |√v - 1| E|X|, X ~ N(0, I_d).
# 2. Ring potential of wasserstein.jl: the fixed parameters against a refinement by the factor 2
#    (panel width, angular nodes and smoothing sub-intervals).
# 3. Ring potential: W_1 / h at small h against the limits of Corollary cor: w1 limits, computed by
#    coefficients.jl.

include("coefficients.jl")
using .Wasserstein, .Coefficients
using Printf, SpecialFunctions

const QUAD = Potential(r -> r^2 / 2, r -> r, r -> 1.0, r -> 1.0)

mean_norm(d) = sqrt(2) * exp(loggamma((d + 1) / 2) - loggamma(d / 2))

timed(f, args...; kw...) = (t0 = time(); v = f(args...; kw...); (v, time() - t0))

println("threads: ", Threads.nthreads())
println("1. quadratic potential against the closed form"); flush(stdout)
for d in (2, 4, 10), j in (1, 3, 5), σ in (0.0, 1.0)
    h = 2.0^-j
    for (name, f, v) in (("SGLD", Wasserstein.w1_sgld, (2 + h * σ^2) / (2 - h)),
                         ("SGLM", Wasserstein.w1_sglm, 1 + h * σ^2 / (2 - h)))
        exact = abs(sqrt(v) - 1) * mean_norm(d)
        val, sec = timed(f, QUAD, d, h, σ)
        @printf("  d=%d h=2^-%d σ=%g %s: %.12e exact %.12e  abs err %.1e  (%.1f s)\n", d, j, σ, name, val, exact,
                abs(val - exact), sec)
        flush(stdout)
    end
end

println("2. ring potential, fixed parameters against refinement by 2"); flush(stdout)
for (d, j, σ) in ((2, 4, 1.0), (4, 2, 1.0), (4, 6, 1.0), (10, 4, 1.0))
    h = 2.0^-j
    for (name, f) in (("SGLD", Wasserstein.w1_sgld), ("SGLM", Wasserstein.w1_sglm))
        v1, s1 = timed(f, RING, d, h, σ)
        v2, s2 = timed(f, RING, d, h, σ; refine=2)
        @printf("  d=%d h=2^-%d σ=%g %s: %.12e refined %.12e  rel diff %.1e  (%.1f s, %.1f s)\n", d, j, σ, name, v1, v2,
                abs(v1 - v2) / v2, s1, s2)
        flush(stdout)
    end
end

println("3. ring potential, W_1 / h against the limit as h -> 0"); flush(stdout)
for (d, σ) in ((2, 1.0), (4, 1.0))
    lsglm, lsgld = coefficient_sglm(d, σ), coefficient_sgld(d, σ)
    @printf("  d=%d σ=%g: limits SGLM %.10f  SGLD %.10f\n", d, σ, lsglm, lsgld); flush(stdout)
    for j in 1:7
        h = 2.0^-j
        (a, sa), (b, sb) = timed(w1_sgld, d, h, σ), timed(w1_sglm, d, h, σ)
        @printf("    h=2^-%d: SGLD W1/h %.10f  SGLM W1/h %.10f  (%.1f s, %.1f s)\n", j, a / h, b / h, sa, sb)
        flush(stdout)
    end
end
