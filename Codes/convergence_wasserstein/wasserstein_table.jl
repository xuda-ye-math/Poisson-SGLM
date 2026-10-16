# W_1(π_h, π)/h (SGLM) and W_1(π̄_h, π)/h (SGLD) in dimension d = 128 for h = 2^-1, ..., 2^-8 and σ = 2, 1,
# as a 4 x 8 table in results/wasserstein_d128.csv. Before the table, two checks in d = 128 at h = 2^-4: the
# closed form for u(r) = r^2/2 (as in check_wasserstein.jl) and a refinement by 2 for σ = 1.
#
#   julia --project=. -t 32 wasserstein_table.jl > results/wasserstein_table_d128.log

include("coefficients.jl")
using .Wasserstein
using Printf, SpecialFunctions

const D = 128
const SIGMAS = (2.0, 1.0)
const EXPS = 1:8
const SCHEMES = (("SGLM", Wasserstein.w1_sglm), ("SGLD", Wasserstein.w1_sgld))
const QUAD = Potential(r -> r^2 / 2, r -> r, r -> 1.0, r -> 1.0)

timed(f, args...; kw...) = (t0 = time(); v = f(args...; kw...); (v, time() - t0))
mean_norm(d) = sqrt(2) * exp(loggamma((d + 1) / 2) - loggamma(d / 2))

println("checks in d = $D at h = 2^-4"); flush(stdout)
let h = 2.0^-4, σ = 1.0
    for (name, f, v) in (("SGLM", Wasserstein.w1_sglm, 1 + h * σ^2 / (2 - h)),
                         ("SGLD", Wasserstein.w1_sgld, (2 + h * σ^2) / (2 - h)))
        exact = abs(sqrt(v) - 1) * mean_norm(D)
        val, sec = timed(f, QUAD, D, h, σ)
        @printf("  quadratic %s: %.12e exact %.12e  rel err %.1e  (%.1f s)\n", name, val, exact, abs(val - exact) / exact, sec)
        flush(stdout)
    end
    for (name, f) in SCHEMES
        (v1, s1), (v2, s2) = timed(f, RING, D, h, σ), timed(f, RING, D, h, σ; refine=2)
        @printf("  ring %s σ=%g: %.12e refined %.12e  rel diff %.1e  (%.1f s, %.1f s)\n", name, σ, v1, v2,
                abs(v1 - v2) / v2, s1, s2)
        flush(stdout)
    end
end

println("W_1 / h in d = $D"); flush(stdout)
table = Dict{Tuple{String,Float64},Vector{Float64}}()
for σ in SIGMAS, (name, f) in SCHEMES
    row = Float64[]
    for j in EXPS
        h = 2.0^-j
        v, sec = timed(f, RING, D, h, σ)
        push!(row, v / h)
        @printf("  %s σ=%.1f h=2^-%d: W1/h %.12e  (%.1f s)\n", name, σ, j, v / h, sec)
        flush(stdout)
    end
    table[(name, σ)] = row
end

open(joinpath(@__DIR__, "results", "wasserstein_d128.csv"), "w") do io
    println(io, "scheme,sigma,", join([@sprintf("%.8f", 2.0^-j) for j in EXPS], ","))
    for σ in SIGMAS, (name, _) in SCHEMES
        println(io, name, ",", @sprintf("%.1f", σ), ",", join([@sprintf("%.10e", v) for v in table[(name, σ)]], ","))
    end
end
println("wrote results/wasserstein_d128.csv")
