# Checks of coefficients.jl, in dimensions d >= 2.
#
#   julia --project=. -t 32 check_coefficients.jl
#
# 1. Quadratic potential u(r) = r^2/2: ψ_C = σ^2 (|x|^2 - d)/2 and ψ_{∇²U} = (|x|^2 - d)/2, and
#    ||(|x|^2 - d)/2 π||_KR = E|X|/2, X ~ N(0, I_d) (attained by f = |x| and the flux x π/2), so the
#    coefficients are σ^2 E|X|/4 (SGLM) and (1 + σ^2) E|X|/4 (SGLD).
# 2. Ring potential of wasserstein.jl: the fixed panel width against a refinement by the factor 2.

include("coefficients.jl")
using .Wasserstein, .Coefficients
using Printf, SpecialFunctions

const QUAD = Potential(r -> r^2 / 2, r -> r, r -> 1.0, r -> 1.0)

mean_norm(d) = sqrt(2) * exp(loggamma((d + 1) / 2) - loggamma(d / 2))

println("1. quadratic potential against the closed form"); flush(stdout)
for d in (2, 4, 10), σ in (0.5, 1.0, 2.0)
    for (name, f, exact) in (("SGLD", Coefficients.coefficient_sgld, (1 + σ^2) * mean_norm(d) / 4),
                             ("SGLM", Coefficients.coefficient_sglm, σ^2 * mean_norm(d) / 4))
        val = f(QUAD, d, σ)
        @printf("  d=%d σ=%g %s: %.14e exact %.14e  rel err %.1e\n", d, σ, name, val, exact, abs(val - exact) / exact)
        flush(stdout)
    end
end

println("2. ring potential, fixed panel width against refinement by 2"); flush(stdout)
for d in (2, 4, 10), σ in (0.5, 1.0, 2.0)
    for (name, f) in (("SGLD", Coefficients.coefficient_sgld), ("SGLM", Coefficients.coefficient_sglm))
        v1, v2 = f(RING, d, σ), f(RING, d, σ; refine=2)
        @printf("  d=%d σ=%g %s: %.14e refined %.14e  rel diff %.1e\n", d, σ, name, v1, v2, abs(v1 - v2) / v2)
        flush(stdout)
    end
end
