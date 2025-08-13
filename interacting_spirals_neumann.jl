using Plots

Lx, Ly = 200, 200;
nx, ny = 400, 400;

x = range(0, Lx, length=nx)
y = range(0, Ly, length=ny)

A = 0.583189
f(r) = tanh(A*r)
χ(θ, n=1) = n*θ

function spiral(X, Y, x, y)
    Δx = X - x
    Δy = Y - y
    r = sqrt(Δx^2 + Δy^2)
    θ = atan(Δy, Δx)
    return f(r) * exp(im*χ(θ))
end

x₁, y₁ = 20, 20

ψ₀ = spiral.(x', y, x₁, y₁)

heatmap(real(ψ₀))