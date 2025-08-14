using GLMakie
using SparseArrays
using OrdinaryDiffEq: solve, ODEProblem, Tsit5
using ProgressLogging

## parameters

q = 0.2
A = 0.583189

Lx, Ly = 100, 100;
nx, ny = 500, 500;

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)


## Initial conditions

f_hat(r) = tanh(A*r)
χ(θ, r, n=1) = -tan(q*log(1/0.01))*log(r) + n*θ

function spiral(X, Y, xc, yc)
    Δx = X - xc
    Δy = Y - yc
    r = sqrt(Δx^2 + Δy^2)
    θ = atan(Δy, Δx)
    return f_hat(r) * exp(im*χ(θ,r))
end

x₁, y₁ = 20, 20

x = range(0, Lx, length=nx)
y = range(0, Ly, length=ny)
ψ₀ = spiral.(x', y, x₁, y₁);

heatmap(real.(ψ₀))

function rhs!(dψ, ψ, params, t)
    q, ny, nx = params

    @boundscheck for y = 2:ny-1, x = 2:nx-1
       dψ[x,y] = ψ[x-1,y] + ψ[x+1,y] + ψ[x,y-1] + ψ[x,y+1] - 4*ψ[x,y]
    end
    dψ[1, :] .= dψ[3, :]
    dψ[end, :] .= dψ[end-2, :]
    dψ[:, 1] .= dψ[:, 3]
    dψ[:, end] .= dψ[:, end-2]

    dψ += @. (1 + im*q)*(1 - abs2(ψ))*ψ
    return nothing
end

params = (q, ny, nx);
prob = ODEProblem(rhs!, ψ₀, (0.0, 100.0), params);
sol = solve(prob, Tsit5(), saveat=0.5, progress=true, progress_steps=10,
            abstol = 1e-6, reltol = 1e-6);

##

fig = Figure()
ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = L"x", ylabel = L"y")

heatmap_data = Observable(real.(sol[:,:,1]'))
heatmap!(ax, x, y, heatmap_data, colormap = :viridis)
record(fig, "Neumann.mp4", 1:length(sol.t); framerate=20) do i
    heatmap_data[] = real.(sol[:,:,i]')
    ax.title = "t = $(round(sol.t[i], digits=3))"
end