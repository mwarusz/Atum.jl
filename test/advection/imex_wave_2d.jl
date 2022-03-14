using Atum
using Atum.Advection

using Test
using Printf
using StaticArrays: SVector
using LinearAlgebra: norm, dot
using WriteVTK

if !@isdefined integration_testing
  const integration_testing = parse(
    Bool,
    lowercase(get(ENV, "ATUM_INTEGRATION_TESTING", "false")),
  )
end

function Atum.boundarystate(law::AdvectionLaw, n⃗, q⁻, aux⁻, _)
    if dot(constants(law).u⃗, n⃗) ≤ 0
        T = eltype(q⁻)
        return SVector(zero(T)), aux⁻
    else
        return q⁻, aux⁻
    end
end

function wave(law, x⃗, t)
  T = eltype(x⃗)
  β = 5
  μ::T = 1 // 5
  # FIXME: Periodic in x⃗
  r⃗ = mod.(x⃗ .- constants(law).u⃗ .* t .+ 1, 2) .- 1
  ρ = β * exp(- r⃗' * r⃗ / μ^2)
  SVector(ρ)
end

function run(A, FT, N, K; volume_form=WeakForm(), split_rhs = true)
  Nq = N + 1

  u⃗ = FT.((1, 0.3))
  law = AdvectionLaw{FT, 2}(u⃗)

  cell = LobattoCell{FT, A}(Nq, Nq)
  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d); periodic=(true, false),
                   ordering = StackedOrdering{CartesianOrdering}())

  dg = DGSEM(; law, grid, volume_form,
             surface_numericalflux = RusanovFlux(),
             directions = split_rhs ? (1,) : (1,2))

  lindg = DGSEM(; law, grid, volume_form,
                surface_numericalflux = RusanovFlux(),
                directions = (2,))

  cfl = FT(1 // 4)
  dt = cfl * step(v1d) / N / norm(constants(law).u⃗)
  timeend = FT(1.0)

  q = fieldarray(undef, law, grid)
  q .= wave.(Ref(law), points(grid), FT(0))

  @info @sprintf """Starting
  N           = %d
  K           = %d
  volume_form = %s
  norm(q)     = %.16e
  """ N K volume_form weightednorm(dg, q)

  odesolver = ARK23(dg, lindg, fieldarray(q), dt; split_rhs)
  solve!(q, timeend, odesolver)

  qexact = fieldarray(undef, law, grid)
  qexact .= wave.(Ref(law), points(grid), timeend)
  errf = weightednorm(dg, q .- qexact)

  @info @sprintf """Finished
  norm(q)      = %.16e
  norm(q - qe) = %.16e
  """ weightednorm(dg, q) errf

  return errf
end

let
  A = Array
  FT = Float64
  N = 4

  expected_error = Dict()

  #form, lev
  expected_error[WeakForm(), 1] = 6.2258990105481686e-02
  expected_error[WeakForm(), 2] = 7.1435479657086494e-03
  expected_error[WeakForm(), 3] = 1.7909688936818332e-03

  expected_error[FluxDifferencingForm(CentralFlux()), 1] = 6.2258990105481887e-02
  expected_error[FluxDifferencingForm(CentralFlux()), 2] = 7.1435479657087093e-03
  expected_error[FluxDifferencingForm(CentralFlux()), 3] = 1.7909688936816061e-03

  nlevels = integration_testing ? 3 : 1

  @testset for volume_form in (WeakForm(), FluxDifferencingForm(CentralFlux()))
    for split_rhs in (true, false)
      errors = zeros(FT, nlevels)
      for l in 1:nlevels
        K = 5 * 2 ^ (l - 1)
        errors[l] = run(A, FT, N, K; volume_form, split_rhs)
        @test errors[l] ≈ expected_error[volume_form, l]
      end
      if nlevels > 1
        rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
        @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
      end
    end
  end
end

