# Coefficients 1/2 ||ψ π||_KR of SGLM and SGLD for a noise level σ (default 1) and d = 2, 4, ..., 256, as a
# 2 x 128 table in results/coefficients_sigmaS.csv (S = σ); the log lists each value with its change under a
# refinement by 2. Values already in the table are kept, and only the missing dimensions are computed.
#
#   julia --project=. -t 32 coefficient_table.jl >> results/coefficient_table_sigma1.log       (σ = 1)
#   julia --project=. -t 32 coefficient_table.jl 2 >> results/coefficient_table_sigma2.log     (σ = 2)

include("coefficients.jl")
using .Wasserstein, .Coefficients
using Printf

const σ = isempty(ARGS) ? 1.0 : parse(Float64, ARGS[1])
const DIMS = 2:2:256
const SCHEMES = (("SGLM", Coefficients.coefficient_sglm), ("SGLD", Coefficients.coefficient_sgld))
const TABLE = joinpath(@__DIR__, "results", "coefficients_sigma$(isinteger(σ) ? Int(σ) : σ).csv")

values = Dict(name => Dict{Int,Float64}() for (name, _) in SCHEMES)
if isfile(TABLE)
    lines = readlines(TABLE)
    dims = parse.(Int, split(lines[1], ",")[2:end])
    for line in lines[2:end]
        fields = split(line, ",")
        for (d, v) in zip(dims, fields[2:end])
            values[fields[1]][d] = parse(Float64, v)
        end
    end
end

for d in DIMS, (name, f) in SCHEMES
    haskey(values[name], d) && continue
    v, v2 = f(RING, d, σ), f(RING, d, σ; refine=2)
    values[name][d] = v
    @printf("d=%d %s: %.12e  refined %.12e  rel diff %.1e\n", d, name, v, v2, abs(v - v2) / v2)
    flush(stdout)
end

mkpath(dirname(TABLE))
open(TABLE, "w") do io
    println(io, "scheme,", join(DIMS, ","))
    for (name, _) in SCHEMES
        println(io, name, ",", join([@sprintf("%.10e", values[name][d]) for d in DIMS], ","))
    end
end
println("wrote ", TABLE)
