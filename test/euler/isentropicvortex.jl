using Atum
using Atum.Euler

using Test
using Printf
using StaticArrays: SVector
using LinearAlgebra: norm

if !@isdefined integration_testing
  const integration_testing = parse(
    Bool,
    lowercase(get(ENV, "ATUM_INTEGRATION_TESTING", "false")),
  )
end

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

function run(A, FT, N, K; esdg=false)
  Nq = N + 1

  law = EulerLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(20), length=K+1)
  vy = range(FT(-5), stop=FT(5), length=K+1)
  grid = brickgrid(cell, (vx, vy); periodic=(true, true))

  if esdg
    dg = ESDGSEM(; law, cell, grid,
                 volume_numericalflux = EntropyConservativeFlux(),
                 surface_numericalflux = RusanovFlux())
  else
    dg = DGSEM(; law, cell, grid, numericalflux = RusanovFlux())
  end

  cfl = FT(1 // 4)
  dt = cfl * step(vy) / N
  timeend = FT(5)

  q = vortex.(Ref(law), points(grid), FT(0))

  @info @sprintf """Starting
  N       = %d
  K       = %d
  esdg    = %s
  norm(q) = %.16e
  """ N K esdg weightednorm(dg, q)

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  qexact = vortex.(Ref(law), points(grid), timeend)
  errf = weightednorm(dg, q .- qexact)

  @info @sprintf """Finished
  norm(q)      = %.16e
  norm(q - qe) = %.16e
  """ weightednorm(dg, q) errf
  errf
end

let
  A = Array
  FT = Float64
  N = 4

  expected_error = Dict()

  #esdg, lev
  expected_error[false, 1] = 2.1296858506446990e+00
  expected_error[false, 2] = 5.9526168482660669e-01
  expected_error[false, 3] = 6.4358678661161267e-02
  expected_error[false, 4] = 2.3640146930597296e-03

  expected_error[true, 1] = 2.0789522602391144e+00
  expected_error[true, 2] = 3.5478518460567871e-01
  expected_error[true, 3] = 3.0653814065252447e-02
  expected_error[true, 4] = 1.3056958215461041e-03

  nlevels = integration_testing ? 4 : 1

  @testset for esdg in (false, true)
    errors = zeros(FT, nlevels)
    for l in 1:nlevels
      K = 5 * 2 ^ (l - 1)
      errf = run(A, FT, N, K; esdg)
      errors[l] = errf
      @test errors[l] ≈ expected_error[esdg, l]
    end

    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
      @test rates[end] ≈ N + 1 atol = 0.5
    end
  end
end
