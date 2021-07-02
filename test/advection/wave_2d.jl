using Atum
using Atum.Advection

using Test
using StaticArrays: SVector
using LinearAlgebra: norm

function wave(law, x⃗, t)
  ρ = 2 + sin(π * (sum(x⃗) - sum(law.u⃗) * t))
  SVector(ρ)
end

function run(A, FT, N, K)
  Nq = N + 1

  law = AdvectionLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d); periodic=(true, true))

  dg = DGSEM(; law, cell, grid, numericalflux = RusanovFlux())

  cfl = FT(1 // 4)
  dt = cfl * step(v1d) / N / norm(law.u⃗)
  timeend = FT(0.7)
  
  q = wave.(Ref(law), points(grid), FT(0))

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  qexact = wave.(Ref(law), points(grid), timeend)
  errf = map(components(q), components(qexact)) do f, fexact
    sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
  end
  errf[1]
end

let
  A = Array
  FT = Float64
  N = 4

  nlevels = 2
  errors = zeros(FT, nlevels)

  for l in 1:nlevels
    K = 5 * 2 ^ (l - 1)
    errors[l] = run(A, FT, N, K)
  end
  rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
  @test rates[end] ≈ N + 1 atol=0.1
end
