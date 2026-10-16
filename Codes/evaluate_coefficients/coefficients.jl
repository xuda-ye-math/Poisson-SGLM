# Bias coefficients of SGLM and SGLD for the test functions of model.toml and the two noises,
#   SGLM:  lim (pi_h(f) - pi(f)) / h    = 1/2 pi(f psi_C),
#   SGLD:  lim (barpi_h(f) - pi(f)) / h = 1/2 pi(f (psi_H + psi_C)),
# with the Riesz representors of the paper,
#   psi_H = 1/2 |grad U|^2 - Lap U + const                       (representor of Hess U),
#   psi_C = G(U) - F(U) + const,  G' = F,  for C(x) = F(U(x)) I   (representor of C),
# so that
#   noise 1, C = sigma1^2 I:                           psi_C = sigma1^2 U + const,
#   noise 2, C = s2 (1 + kappa - kappa e^{-(U-U*)}) I:  psi_C = s2 ((1 + kappa) U + 2 kappa e^{-(U-U*)}) + const.
# Every pi-integral is a 4D tensor Gauss-Hermite rule on each mixture component; the rule is
# refined over NS and the Riesz identities pi(psi (-L omega)) = pi(Hess omega : A) are checked
# on three functions omega at every refinement.
#
#   julia -t 16 coefficients.jl > results/coefficients.log 2>&1

include(joinpath(@__DIR__, "model.jl"))
using .Model
using LinearAlgebra, Printf, Random, Base.Threads

const DIR = @__DIR__
const RES = joinpath(DIR, "results")
const NS = (28, 40, 52, 64)

const mix, tests, noise, selected = load_model(joinpath(DIR, "model.toml"))
const Ustar = U_STAR(mix)
const NT = length(tests)

println("threads: ", nthreads())
println("U_star = ", Ustar)
println("upper bound on the largest eigenvalue of Hess U: max_k lambda_max(Sigma_k^{-1}) = ",
    maximum(eigmax(inv(S)) for S in mix.Sigma))
flush(stdout)

# ---------------------------------------------------------------- checks of the potential
function model_checks()
    rng = Xoshiro(7)
    ε = 1e-4
    eg = 0.0; el = 0.0; eh = 0.0
    for _ in 1:20
        x = ntuple(_ -> 3 * randn(rng), 4)
        U, gU, lapU, g2 = potential(mix, x)
        H = hessian(mix, x)
        for i in 1:4
            e = ntuple(j -> j == i ? ε : 0.0, 4)
            xp = ntuple(j -> x[j] + e[j], 4); xm = ntuple(j -> x[j] - e[j], 4)
            Up = potential(mix, xp)[1]; Um = potential(mix, xm)[1]
            eg = max(eg, abs((Up - Um) / (2ε) - gU[i]))
            el = max(el, abs((potential(mix, xp)[2][i] - potential(mix, xm)[2][i]) / (2ε) - H[i, i]))
        end
        el = max(el, abs(tr(H) - lapU))
        v = ntuple(_ -> randn(rng), 4)
        eh = max(eh, abs(hessian_quad(mix, x, v) - dot(collect(v), H * collect(v))))
    end
    @printf("finite-difference checks at 20 points: grad U %.2e, Hess U diagonal / Laplacian %.2e, v'Hv %.2e\n", eg, el, eh)

    println("local minima of U, by gradient descent from each mean:")
    for k in eachindex(mix.w)
        x = mix.mu[k]
        for _ in 1:200000
            g = potential(mix, x)[2]
            x = ntuple(i -> x[i] - 0.05 * g[i], 4)
            sqrt(sum(abs2, g)) < 1e-13 && break
        end
        U, g, _, _ = potential(mix, x)
        @printf("  k = %d  w = %.2f  |x* - mu_k| = %.3f  U(x*) = %.4f  |grad U| = %.1e  lambda_min(Hess U) = %.3f\n",
            k, mix.w[k], sqrt(sum(abs2, x .- mix.mu[k])), U, sqrt(sum(abs2, g)), eigmin(hessian(mix, x)))
    end
    flush(stdout)
end
model_checks()

# ---------------------------------------------------------------- quadrature
function gauss_hermite(n)
    E = eigen(SymTridiagonal(zeros(n), sqrt.(1.0:n-1)))
    return E.values, E.vectors[1, :] .^ 2           # probabilists' rule, weights sum to 1
end

const OMEGA_B = ((0.7, -0.3, 0.5, 0.2), (-0.4, 0.9, 0.1, -0.6))
const OMEGA_PHASE = (0.0, 0.4)
const OMEGA_C = ntuple(i -> mix.mu[2][i] + 0.5, 4)
const NOMEGA = 3
omega_name(j) = j <= 2 ? "cos_b$(j)" : "bump_mu2"

# per test function: f, fU, f psiH, fE;  globals: U, psiH, E, LapU, |grad U|^2;
# per omega: -L omega, U(-L omega), E(-L omega), psiH(-L omega), Lap omega, E Lap omega, Hess omega : Hess U
const NACC = 4NT + 5 + 7NOMEGA

@inline function accumulate!(acc, W, x)
    U, gU, lapU, g2 = potential(mix, x)
    E = exp(-(U - Ustar))
    psiH = 0.5 * g2 - lapU
    @inbounds for t in 1:NT
        fv = evalf(tests[t], x)
        acc[4t-3] += W * fv
        acc[4t-2] += W * fv * U
        acc[4t-1] += W * fv * psiH
        acc[4t] += W * fv * E
    end
    o = 4NT
    @inbounds begin
        acc[o+1] += W * U; acc[o+2] += W * psiH; acc[o+3] += W * E; acc[o+4] += W * lapU; acc[o+5] += W * g2
    end
    o += 5
    @inbounds for j in 1:NOMEGA
        if j <= 2
            b = OMEGA_B[j]
            s, c = sincos(b[1] * x[1] + b[2] * x[2] + b[3] * x[3] + b[4] * x[4] + OMEGA_PHASE[j])
            gω = (-s * b[1], -s * b[2], -s * b[3], -s * b[4])
            lapω = -(b[1]^2 + b[2]^2 + b[3]^2 + b[4]^2) * c
            hω = -c * hessian_quad(mix, x, b)
        else
            y = ntuple(i -> x[i] - OMEGA_C[i], 4)
            y2 = y[1]^2 + y[2]^2 + y[3]^2 + y[4]^2
            ω = exp(-y2 / 2)
            gω = ntuple(i -> -ω * y[i], 4)
            lapω = ω * (y2 - 4)
            hω = ω * (hessian_quad(mix, x, y) - lapU)
        end
        mL = -lapω + gU[1] * gω[1] + gU[2] * gω[2] + gU[3] * gω[3] + gU[4] * gω[4]
        acc[o+1] += W * mL; acc[o+2] += W * U * mL; acc[o+3] += W * E * mL; acc[o+4] += W * psiH * mL
        acc[o+5] += W * lapω; acc[o+6] += W * E * lapω; acc[o+7] += W * hω
        o += 7
    end
    return nothing
end

function integrate(n)
    z, wz = gauss_hermite(n)
    K = length(mix.w)
    chunks = Iterators.partition(1:(K*n), cld(K * n, nthreads()))
    tasks = map(chunks) do idxs
        Threads.@spawn begin
            acc = zeros(NACC)
            for idx in idxs
                k = (idx - 1) ÷ n + 1
                i1 = (idx - 1) % n + 1
                L = mix.chol[k]; mu = mix.mu[k]
                for i2 in 1:n, i3 in 1:n, i4 in 1:n
                    zz = (z[i1], z[i2], z[i3], z[i4])
                    x = ntuple(i -> mu[i] + L[i][1] * zz[1] + L[i][2] * zz[2] + L[i][3] * zz[3] + L[i][4] * zz[4], 4)
                    accumulate!(acc, mix.w[k] * wz[i1] * wz[i2] * wz[i3] * wz[i4], x)
                end
            end
            acc
        end
    end
    return sum(fetch.(tasks))
end

# ---------------------------------------------------------------- coefficients from the integrals
function coefficients(acc)
    o = 4NT
    πU, πψH, πE, πlap, πg2 = acc[o+1], acc[o+2], acc[o+3], acc[o+4], acc[o+5]
    rows = NamedTuple[]
    for t in 1:NT
        πf, πfU, πfψH, πfE = acc[4t-3], acc[4t-2], acc[4t-1], acc[4t]
        covU = πfU - πf * πU
        covH = πfψH - πf * πψH
        covE = πfE - πf * πE
        h = 0.5 * covH                                                    # 1/2 pi(f psi_H)
        c1 = 0.5 * noise.sigma1_sq * covU                                 # 1/2 pi(f psi_C), noise 1
        c2 = 0.5 * noise.s2 * ((1 + noise.kappa) * covU + 2 * noise.kappa * covE)   # noise 2
        push!(rows, (test = tests[t].name, pi_f_quad = πf, cov_f_U = covU, cov_f_psiH = covH, cov_f_E = covE,
            half_pi_f_psiH = h, noise1_SGLM = c1, noise1_SGLD = c1 + h, noise2_SGLM = c2, noise2_SGLD = c2 + h))
    end
    return rows, (πU = πU, πψH = πψH, πE = πE, πlap = πlap, πg2 = πg2)
end

function riesz_rows(n, acc)
    o = 4NT
    πU, πψH, πE = acc[o+1], acc[o+2], acc[o+3]
    o += 5
    out = NamedTuple[]
    for j in 1:NOMEGA
        mL, UmL, EmL, ψHmL, lapω, Elapω, hω = acc[o+1:o+7]
        # representor of sigma1^2 I: sigma1^2 U;  of F2 I: s2((1+kappa)U + 2 kappa E);  of Hess U: psiH
        lhs1 = noise.sigma1_sq * (UmL - πU * mL)
        rhs1 = noise.sigma1_sq * lapω
        lhs2 = noise.s2 * ((1 + noise.kappa) * (UmL - πU * mL) + 2 * noise.kappa * (EmL - πE * mL))
        rhs2 = noise.s2 * ((1 + noise.kappa) * lapω - noise.kappa * Elapω)
        lhsH = ψHmL - πψH * mL
        push!(out, (n = n, omega = omega_name(j), pi_minus_L_omega = mL, C1_lhs = lhs1, C1_rhs = rhs1,
            C2_lhs = lhs2, C2_rhs = rhs2, H_lhs = lhsH, H_rhs = hω))
        o += 7
    end
    return out
end

exact = [exact_mean(f, mix) for f in tests]
results = Dict{Int,Any}()
riesz = NamedTuple[]
conv = IOBuffer()
for n in NS
    t0 = time()
    acc = integrate(n)
    rows, glob = coefficients(acc)
    results[n] = rows
    append!(riesz, riesz_rows(n, acc))
    @printf("n = %d (%d nodes per component): %.1f s;  pi(|grad U|^2) - pi(Lap U) = %.2e,  max |pi(f) quad - exact| = %.2e\n",
        n, n^4, time() - t0, glob.πg2 - glob.πlap, maximum(abs(rows[t].pi_f_quad - exact[t]) for t in 1:NT))
    flush(stdout)
end

# ---------------------------------------------------------------- output
nlast, nprev = NS[end], NS[end-1]
cols = (:noise1_SGLM, :noise1_SGLD, :noise2_SGLM, :noise2_SGLD)
qerr(t) = maximum(abs(results[nlast][t][c] - results[nprev][t][c]) for c in cols)

open(joinpath(RES, "quadrature_convergence.csv"), "w") do io
    println(io, "n,test,pi_f_quad,cov_f_U,cov_f_psiH,cov_f_E,noise1_SGLM,noise1_SGLD,noise2_SGLM,noise2_SGLD")
    for n in NS, r in results[n]
        @printf(io, "%d,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", n, r.test, r.pi_f_quad, r.cov_f_U,
            r.cov_f_psiH, r.cov_f_E, r.noise1_SGLM, r.noise1_SGLD, r.noise2_SGLM, r.noise2_SGLD)
    end
end

open(joinpath(RES, "riesz_checks.csv"), "w") do io
    println(io, "n,omega,pi_minus_L_omega,C1_lhs,C1_rhs,C2_lhs,C2_rhs,H_lhs,H_rhs")
    for r in riesz
        @printf(io, "%d,%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", r.n, r.omega, r.pi_minus_L_omega,
            r.C1_lhs, r.C1_rhs, r.C2_lhs, r.C2_rhs, r.H_lhs, r.H_rhs)
    end
end

open(joinpath(RES, "pool_coefficients.csv"), "w") do io
    println(io, "test,pi_f_exact,half_pi_f_psiH,noise1_SGLM,noise1_SGLD,noise2_SGLM,noise2_SGLD,quad_err")
    for (t, r) in enumerate(results[nlast])
        @printf(io, "%s,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.3e\n", r.test, exact[t], r.half_pi_f_psiH,
            r.noise1_SGLM, r.noise1_SGLD, r.noise2_SGLM, r.noise2_SGLD, qerr(t))
    end
end

if !isempty(selected)
    open(joinpath(RES, "coefficients.csv"), "w") do io
        println(io, "test,noise,scheme,coefficient,pi_f_exact,quad_err")
        for name in selected
            t = findfirst(f -> f.name == name, tests)
            r = results[nlast][t]
            for (nz, sgld, sglm) in (("gaussian", r.noise1_SGLD, r.noise1_SGLM), ("nongaussian", r.noise2_SGLD, r.noise2_SGLM))
                @printf(io, "%s,%s,SGLD,%.16e,%.16e,%.3e\n", name, nz, sgld, exact[t], qerr(t))
                @printf(io, "%s,%s,SGLM,%.16e,%.16e,%.3e\n", name, nz, sglm, exact[t], qerr(t))
            end
        end
    end
end
cp(joinpath(DIR, "model.toml"), joinpath(RES, "model_snapshot.toml"); force = true)

println("\nRiesz identities at n = $nlast (lhs = pi(psi (-L omega)), rhs = pi(Hess omega : A)):")
for r in filter(r -> r.n == nlast, riesz)
    @printf("  %-9s pi(-L omega) = %+.1e | sigma1^2 I: %+.10f %+.10f | F2 I: %+.10f %+.10f | Hess U: %+.10f %+.10f\n",
        r.omega, r.pi_minus_L_omega, r.C1_lhs, r.C1_rhs, r.C2_lhs, r.C2_rhs, r.H_lhs, r.H_rhs)
end
println("\nCoefficients at n = $nlast (quad_err = max change from n = $nprev):")
@printf("  %-13s %12s %12s %12s %12s %12s %10s\n", "test", "pi(f)", "G: SGLD", "G: SGLM", "NG: SGLD", "NG: SGLM", "quad_err")
for (t, r) in enumerate(results[nlast])
    @printf("  %-13s %+12.6f %+12.6f %+12.6f %+12.6f %+12.6f %10.1e%s\n", r.test, exact[t], r.noise1_SGLD,
        r.noise1_SGLM, r.noise2_SGLD, r.noise2_SGLM, qerr(t), r.test in selected ? "  *" : "")
end
println("done")
