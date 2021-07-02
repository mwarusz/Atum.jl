using Atum
using Atum.Euler

using Test
using StaticArrays: SVector
using LinearAlgebra: norm

function wave(law, x⃗, t)
  FT = eltype(law)
  u⃗ = SVector(FT(1), FT(1))
  ρ = 2 + sin(π * (sum(x⃗) - sum(u⃗) * t))
  ρu⃗ = ρ * u⃗
  p = FT(1)
  ρe = Euler.energy(law, ρ, ρu⃗, p)
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K)
  Nq = N + 1

  law = EulerLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d); periodic=(true, true))

  dg = DGSEM(; law, cell, grid, numericalflux = RusanovFlux())

  cfl = FT(1 // 4)
  dt = cfl * step(v1d) / N / Euler.soundspeed(law, FT(1), FT(1))
  timeend = FT(0.7)
  
  q = wave.(Ref(law), points(grid), FT(0))

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  qexact = wave.(Ref(law), points(grid), timeend)
  errf = map(components(q), components(qexact)) do f, fexact
    sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
  end
  norm(errf)
end

let
  A = Array
  FT = Float64
  N = 4

  nlevels = 3
  errors = zeros(FT, nlevels)

  for l in 1:nlevels
    K = 5 * 2 ^ (l - 1)
    errf = run(A, FT, N, K)
    errors[l] = errf
  end
  rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
  @test rates[end] ≈ N + 1 atol = 0.15
end
