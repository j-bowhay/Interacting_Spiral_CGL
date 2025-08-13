using GLMakie
using FFTW
using OrdinaryDiffEq: solve, ODEProblem, Tsit5, Rosenbrock23
using ProgressLogging
using LaTeXStrings
using LinearAlgebra
using DSP

##

d = 0.1;
β = 1.0;

Lx = 20;  # size of X-dim
Ly = 20;  # size of Y-dim
n = 2^7;  # spatial grid resolution (number of Fourier modes in each dimension)
N = n*n;  # total number of grid points
x2 = range(-Lx/2, Lx/2, n+1);
y2 = range(-Ly/2, Ly/2, n+1);
kx = (2*pi/Lx)*[0:(n/2-1); -n/2:-1];
ky = (2*pi/Ly)*[0:(n/2-1); -n/2:-1];
K22 = -d*(kx'.^2 .+ ky.^2);  # Laplacian in Fourier space

## initial conditions

m = 1  # number of spirals
x = collect(x2[1:2:n]);
y = collect(y2[1:n]);

r = @. sqrt(x'^2 + y^2)
θ = @. angle(x' + im*y);

u0_half = @. tanh(r)*cos(m*θ - r);
u0 = [reverse(u0_half, dims=2) u0_half];

v0_half = @. tanh(r)*sin(m*θ - r);
v0 = [reverse(v0_half, dims=2) v0_half];

U0 = cat(fft(u0), fft(v0); dims=3);

## rhs

u3 = similar(u0)
v3 = similar(v0)
u2v = similar(u0)
uv2 = similar(u0)
u_real = similar(u0, Float64)  # for real part of ifft
v_real = similar(v0, Float64)  # for real part of ifft
temp_complex1 = similar(U0[:,:,1])  # temporary for nonlinear terms
temp_complex2 = similar(U0[:,:,1])  # temporary for nonlinear terms

fft_plan = plan_fft!(temp_complex1)
ifft_plan = plan_ifft!(temp_complex1)

function rhs!(dU, U, params, t)
    β, K22, u3, v3, u2v, uv2, u_real, v_real, temp_complex1, temp_complex2, fft_plan, ifft_plan = params

    ut = @view U[:, :, 1]
    vt = @view U[:, :, 2]

    copyto!(temp_complex1, ut)
    ifft_plan * temp_complex1
    @. u_real = real(temp_complex1)
    copyto!(temp_complex2, vt)
    ifft_plan * temp_complex2
    @. v_real = real(temp_complex2)

    @. u3 = u_real^3
    @. v3 = v_real^3
    @. u2v = (u_real^2) * v_real
    @. uv2 = u_real * (v_real^2)

    @. temp_complex1 = u_real - u3 - uv2 + β*u2v + β*v3
    fft_plan * temp_complex1

    @. temp_complex2 = v_real - u2v - v3 - β*u3 - β*uv2
    fft_plan * temp_complex2

    @. dU[:, :, 1] = K22*ut + temp_complex1
    @. dU[:, :, 2] = K22*vt + temp_complex2
end

## time-stepping
params = (β, K22, u3, v3, u2v, uv2, u_real, v_real, temp_complex1, temp_complex2, fft_plan, ifft_plan);
problem = ODEProblem(rhs!, U0, (0.0, 100.0), params);
@time sol = solve(problem, Tsit5(), saveat=0.5, progress=true, progress_steps=10,
                  abstol = 1e-4, reltol = 1e-4);

## solution plot

fps = 20;

u = [real(ifft(sol[:, :, 1, i])) ./ (n^2) for i in 1:length(sol.t)];

fig = Figure()
ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = L"x", ylabel = L"y")

heatmap_data = Observable(u[1]')
heatmap!(ax, x2, y2, heatmap_data, colormap = :viridis)

gr = GridLayout(fig[1, 2])
ax2 = Axis(gr[1, 1], xlabel = L"t", ylabel = "Norm", title = "Solution Norm")
lines!(ax2, sol.t, [norm(ut[:]) for ut in u];)

vline_pos = Observable(sol.t[1])
vlines!(ax2, vline_pos, color=:red)

ax3 = Axis(gr[2, 1], xlabel = L"k\approx\sqrt{k_1^2 + k_2^2}",
           ylabel = L"\frac{1}{n_k}\sum|û_k|", yscale=log10,
           title = "Fourier modes", limits = (nothing, (10^-25, 1)))
pxx = periodogram(u[1], radialavg = true, nfft = (n, n))

f = Observable(freq(pxx))
p = Observable(power(pxx))

scatter!(ax3, f, p)

record(fig, "heatmap_animation.mp4", 1:length(u); framerate=fps) do i
    heatmap_data[] = u[i]'
    ax.title = "t = $(round(sol.t[i], digits=3))"
    vline_pos[] = sol.t[i]
    pxx = periodogram(u[i], radialavg = true, nfft = (n, n))
    f[] = freq(pxx)
    p[] = power(pxx)
end