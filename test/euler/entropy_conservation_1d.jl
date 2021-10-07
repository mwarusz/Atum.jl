using Atum
using Atum.Euler

using Test
using Printf
using StaticArrays: SVector

function square(law, x⃗)
  FT = eltype(law)
  ρ = @inbounds abs(x⃗[1]) < FT(1 // 2) ? FT(2) : FT(1)
  u⃗ = SVector(FT(1))
  ρu⃗ = ρ * u⃗
  p = FT(1)
  ρe = Euler.energy(law, ρ, ρu⃗, p)
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K; volume_form=WeakForm())
  Nq = N + 1

  law = EulerLaw{FT, 1}()
  
  cell = LobattoCell{FT, A}(Nq)
  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d,); periodic=(true,))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = EntropyConservativeFlux())

  cfl = FT(1 // 2)
  dt = cfl * min_node_distance(grid) / Euler.soundspeed(law, FT(1), FT(1))
  timeend = FT(5.0)
  
  q = square.(Ref(law), points(grid))
  η0 = entropyintegral(dg, q)

  @info @sprintf """Starting
  N           = %d
  K           = %d
  volume_form = %s
  norm(q)     = %.16e
  η0          = %.16e
  """ N K volume_form weightednorm(dg, q) η0

  odesolver = RLSRK54(dg, q, dt)

  solve!(q, timeend, odesolver)
  ηf = entropyintegral(dg, q)

  Δη = (ηf - η0) / abs(η0)

  @info @sprintf """Finished
  norm(q) = %.16e
  ηf      = %.16e
  Δη      = %.16e
  """ weightednorm(dg, q) ηf Δη

  Δη
end

let
  A = Array
  FT = Float64
  N = 4
  K = 5

  volume_form = FluxDifferencingForm(EntropyConservativeFlux())
  Δη = run(A, FT, N, K; volume_form)

  @test abs(Δη) <= 20eps(FT)
end
