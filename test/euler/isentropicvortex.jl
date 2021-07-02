using Atum
using Atum.Euler

using Test
using StaticArrays: SVector
using LinearAlgebra: norm

function vortex(law, x⃗, t)
  FT = eltype(law)
  c⃗ = SVector(FT(5), FT(0))
  u⃗₀ = SVector(FT(1), FT(0))
  β = FT(5)
  r⃗ = x⃗ - c⃗ - u⃗₀ * t
  f = β * exp(1 - r⃗' * r⃗)
  
  ρ = (1 - (γ(law) - 1) * f ^ 2 / (16 * γ(law) * π ^ 2)) ^ (1 / (γ(law) - 1))
  u⃗ = u⃗₀ + f / 2π * SVector(-r⃗[2], r⃗[1])
  ρu⃗ = ρ * u⃗
  p = ρ ^ γ(law)
  ρe = Euler.energy(law, ρ, ρu⃗, p)

  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K)
  Nq = N + 1

  law = EulerLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(20), length=K+1)
  vy = range(FT(-5), stop=FT(5), length=K+1)
  grid = brickgrid(cell, (vx, vy); periodic=(true, true))

  dg = DGSEM(; law, cell, grid, numericalflux = RusanovFlux())

  cfl = FT(1 // 4)
  dt = cfl * step(vy) / N
  timeend = FT(5)

  q = vortex.(Ref(law), points(grid), FT(0))

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  qexact = vortex.(Ref(law), points(grid), timeend)
  errf = map(components(q), components(qexact)) do f, fexact
    sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
  end
  norm(errf)
end

let
  A = Array
  FT = Float64
  N = 4

  nlevels = 4
  errors = zeros(FT, nlevels)

  for l in 1:nlevels
    K = 5 * 2 ^ (l - 1)
    errf = run(A, FT, N, K)
    errors[l] = errf
  end
  rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
  @test rates[end] ≈ N + 1 atol = 0.25
end
