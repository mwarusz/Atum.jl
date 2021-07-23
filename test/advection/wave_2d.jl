using Atum
using Atum.Advection

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
  ρ = 2 + sin(π * (sum(x⃗) - sum(law.u⃗) * t))
  SVector(ρ)
end

function run(A, FT, N, K; esdg=false)
  Nq = N + 1

  law = AdvectionLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d); periodic=(true, true))

  if esdg
    dg = ESDGSEM(; law, cell, grid,
                 volume_numericalflux = CentralFlux(),
                 surface_numericalflux = RusanovFlux())
  else
    dg = DGSEM(; law, cell, grid, numericalflux = RusanovFlux())
  end

  cfl = FT(1 // 4)
  dt = cfl * step(v1d) / N / norm(law.u⃗)
  timeend = FT(0.7)
  
  q = wave.(Ref(law), points(grid), FT(0))

  @info @sprintf """Starting
  N       = %d
  K       = %d
  esdg    = %s
  norm(q) = %.16e
  """ N K esdg weightednorm(dg, q)

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

  #esdg, lev
  expected_error[false, 1] = 3.0423062189303111e-04
  expected_error[false, 2] = 9.6768870605952377e-06
  expected_error[false, 3] = 2.7887570054941176e-07

  expected_error[true, 1] = 3.0423062189285823e-04
  expected_error[true, 2] = 9.6768870607411476e-06
  expected_error[true, 3] = 2.7887570054607037e-07

  nlevels = integration_testing ? 3 : 1

  @testset for esdg in (false, true)
    errors = zeros(FT, nlevels)
    for l in 1:nlevels
      K = 5 * 2 ^ (l - 1)
      errors[l] = run(A, FT, N, K; esdg)
      @test errors[l] ≈ expected_error[esdg, l]
    end
    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
      @test rates[end] ≈ N + 1 atol=0.12
    end
  end
end
