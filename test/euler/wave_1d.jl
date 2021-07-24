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

function wave(law, x⃗, t)
  FT = eltype(law)
  u⃗ = SVector(FT(1))
  ρ = 2 + sin(π * (sum(x⃗) - sum(u⃗) * t))
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

  dg = DGSEM(; law, cell, grid, volume_form,
               surface_numericalflux = RusanovFlux())

  cfl = FT(1 // 2)
  dt = cfl * step(v1d) / N / Euler.soundspeed(law, FT(1), FT(1))
  timeend = FT(0.7)
  
  q = wave.(Ref(law), points(grid), FT(0))

  @info @sprintf """Starting
  N           = %d
  K           = %d
  volume_form = %s
  norm(q)     = %.16e
  """ N K volume_form weightednorm(dg, q)

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  qexact = wave.(Ref(law), points(grid), timeend)
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

  #form, lev
  expected_error[WeakForm(), 1] = 2.3064893293612415e-04
  expected_error[WeakForm(), 2] = 9.4162785098638794e-06
  expected_error[WeakForm(), 3] = 3.2238871208445972e-07
  expected_error[WeakForm(), 4] = 1.0575542508411030e-08

  expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 1] = 5.2210610664510859e-04
  expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 2] = 1.4125373916616633e-05
  expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 3] = 5.5685095409498615e-07
  expected_error[FluxDifferencingForm(EntropyConservativeFlux()), 4] = 2.1151831312278871e-08

  nlevels = integration_testing ? 4 : 1

  @testset for volume_form in (WeakForm(),
                               FluxDifferencingForm(EntropyConservativeFlux()))
    errors = zeros(FT, nlevels)
    for l in 1:nlevels
      K = 5 * 2 ^ (l - 1)
      errf = run(A, FT, N, K; volume_form)
      errors[l] = errf
      @test errors[l] ≈ expected_error[volume_form, l]
    end
    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
      @test rates[end] ≈ N + 1 atol = 0.3
    end
  end
end
