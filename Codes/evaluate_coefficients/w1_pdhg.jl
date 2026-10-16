# The PDHG iteration and the convergence quantities of w1_coefficients.jl on the GPU (CUDA.jl),
# one GPU thread per cell; included by w1_coefficients.jl. Needs the CUDA.jl environment of this
# folder:  julia --project=. -t 16 w1_coefficients.jl CASE 25,50,100

using CUDA

@inline function _cell(n)
    I = (Int(blockIdx().x) - 1) * Int(blockDim().x) + Int(threadIdx().x)
    J = I - 1
    i1 = J % n + 1; J ÷= n
    i2 = J % n + 1; J ÷= n
    i3 = J % n + 1; i4 = J ÷ n + 1
    return I, i1, i2, i3, i4
end

function _flux_kernel!(fb, y1, y2, y3, y4, n, N, Δ, σ)
    I, i1, i2, i3, i4 = _cell(n)
    I > N && return nothing
    @inbounds begin
        fi = fb[I]
        z1 = i1 < n ? y1[I] + σ * (fb[I+1] - fi) / Δ : 0.0
        z2 = i2 < n ? y2[I] + σ * (fb[I+n] - fi) / Δ : 0.0
        z3 = i3 < n ? y3[I] + σ * (fb[I+n*n] - fi) / Δ : 0.0
        z4 = i4 < n ? y4[I] + σ * (fb[I+n*n*n] - fi) / Δ : 0.0
        nz = sqrt(z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4)
        c = nz > σ ? 1 - σ / nz : 0.0
        y1[I] = c * z1; y2[I] = c * z2; y3[I] = c * z3; y4[I] = c * z4
    end
    return nothing
end

@inline function _divergence(y1, y2, y3, y4, I, i1, i2, i3, i4, n, Δ)
    @inbounds ((i1 > 1 ? y1[I-1] : 0.0) - y1[I] + (i2 > 1 ? y2[I-n] : 0.0) - y2[I] +
               (i3 > 1 ? y3[I-n*n] : 0.0) - y3[I] + (i4 > 1 ? y4[I-n*n*n] : 0.0) - y4[I]) / Δ
end

function _potential_kernel!(f, fb, y1, y2, y3, y4, nuh, n, N, Δ, τ)
    I, i1, i2, i3, i4 = _cell(n)
    I > N && return nothing
    @inbounds begin
        g = _divergence(y1, y2, y3, y4, I, i1, i2, i3, i4, n, Δ) - nuh[I]
        fb[I] = f[I] - 2τ * g
        f[I] -= τ * g
    end
    return nothing
end

function _bounds_kernel!(t1, t2, t3, t4, f, y1, y2, y3, y4, nuh, n, N, Δ)
    I, i1, i2, i3, i4 = _cell(n)
    I > N && return nothing
    @inbounds begin
        d1 = i1 < n ? (f[I+1] - f[I]) / Δ : 0.0
        d2 = i2 < n ? (f[I+n] - f[I]) / Δ : 0.0
        d3 = i3 < n ? (f[I+n*n] - f[I]) / Δ : 0.0
        d4 = i4 < n ? (f[I+n*n*n] - f[I]) / Δ : 0.0
        t1[I] = d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4
        t2[I] = f[I] * nuh[I]
        t3[I] = sqrt(y1[I]^2 + y2[I]^2 + y3[I]^2 + y4[I]^2)
        t4[I] = abs(nuh[I] - _divergence(y1, y2, y3, y4, I, i1, i2, i3, i4, n, Δ))
    end
    return nothing
end

to_gpu(f, y, nuh) = (f = CuArray(f), fb = CuArray(f), y = map(CuArray, y), nuh = CuArray(nuh),
    t = ntuple(_ -> CUDA.zeros(Float64, length(f)), 4))

from_gpu(d) = (Array(d.f), map(Array, d.y))

const GPU_THREADS = 256

function pdhg_gpu!(d, n, Δ, τ, σ, iters)
    N = n^4; blocks = cld(N, GPU_THREADS)
    for _ in 1:iters
        @cuda threads = GPU_THREADS blocks = blocks _flux_kernel!(d.fb, d.y..., n, N, Δ, σ)
        @cuda threads = GPU_THREADS blocks = blocks _potential_kernel!(d.f, d.fb, d.y..., d.nuh, n, N, Δ, τ)
    end
    CUDA.synchronize()
end

# lower bound <f, nuh> / max |D f|, flux cost sum |y|, max |D f|, residual sum |nuh - D^T y| (density units)
function discrete_bounds_gpu(d, n, Δ)
    N = n^4
    @cuda threads = GPU_THREADS blocks = cld(N, GPU_THREADS) _bounds_kernel!(d.t..., d.f, d.y..., d.nuh, n, N, Δ)
    Lmax = sqrt(maximum(d.t[1]))
    return sum(d.t[2]) / Lmax, sum(d.t[3]), Lmax, sum(d.t[4])
end
