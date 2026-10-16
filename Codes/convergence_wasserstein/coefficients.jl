# Coefficients 1/2 ||ψ π||_KR of the Wasserstein-1 errors in Corollary cor: w1 limits, for the radial
# potential and the gradient noise σ N(0, I_d) of wasserstein.jl:
#
#   include("coefficients.jl"); using .Coefficients
#   coefficient_sgld(d, σ)    1/2 ||(ψ_{∇²U} + ψ_C) π||_KR,  the limit of W_1(π̄_h, π) / h
#   coefficient_sglm(d, σ)    1/2 ||ψ_C π||_KR,              the limit of W_1(π_h, π) / h
#
# For a radial ψ, ||ψ π||_KR = ∫_0^∞ |∫_{|x| ≤ r} ψ dπ| dr (README.md), with ψ_C = σ^2 (U - π(U)) and
# ψ_{∇²U} = 1/2 |∇U|^2 - ΔU + 1/2 π(ΔU), ΔU = u'' + (d - 1) u'/r. check_coefficients.jl tests it.

isdefined(@__MODULE__, :Wasserstein) || include("wasserstein.jl")

module Coefficients

using LinearAlgebra
using ..Wasserstein: Wasserstein, Potential, RING

export coefficient_sgld, coefficient_sglm

const PANEL = 0.02   # panel width of the radial grid, with Wasserstein.NODES Gauss-Legendre nodes per panel

"coefficient 1/2 ||(ψ_{∇²U} + ψ_C) π||_KR of SGLD for the gradient noise σ N(0, I_d)"
coefficient_sgld(d, σ) = coefficient_sgld(RING, d, σ)

"coefficient 1/2 ||ψ_C π||_KR of SGLM for the gradient noise σ N(0, I_d)"
coefficient_sglm(d, σ) = coefficient_sglm(RING, d, σ)

# radial grid, radial density of π and the representors ψ_C, ψ_{∇²U} at the nodes
function representors(pot, d, σ, refine)
    grid = Wasserstein.Grid(Wasserstein.radius(pot, d, 0.0, 0.0), PANEL / refine)
    r, w = grid.r, grid.w
    ρ = Wasserstein.pi_density(pot, grid, d)
    u = pot.u.(r)
    lap = pot.d2u.(r) .+ (d - 1) .* pot.du_over_r.(r)
    ψc = σ^2 .* (u .- dot(w, ρ .* u))
    ψh = 0.5 .* pot.du.(r) .^ 2 .- lap .+ 0.5 * dot(w, ρ .* lap)
    return grid, ρ, ψc, ψh
end

function coefficient_sgld(pot::Potential, d, σ; refine=1)
    grid, ρ, ψc, ψh = representors(pot, d, σ, refine)
    return 0.5 * Wasserstein.w1_from_nodes((ψh .+ ψc) .* ρ, grid)
end

function coefficient_sglm(pot::Potential, d, σ; refine=1)
    grid, ρ, ψc, _ = representors(pot, d, σ, refine)
    return 0.5 * Wasserstein.w1_from_nodes(ψc .* ρ, grid)
end

end # module
