using Atum
using Atum.Euler
using Bennu: fieldarray

using PGFPlotsX
using StaticArrays: SVector

struct Sod <: AbstractProblem end

function sod(law, x⃗)
  FT = eltype(law)
  ρ = x⃗[1] < 1 // 2 ? 1 : 1 // 8
  ρu⃗ = SVector(FT(0))
  p = x⃗[1] < 1 // 2 ? 1 : 1 // 10
  ρe = Euler.energy(law, ρ, ρu⃗, p)
  SVector(ρ, ρu⃗..., ρe)
end

import Atum: boundarystate
function boundarystate(law::EulerLaw, ::Sod, n⃗, q⁻, aux⁻, bctag)
  FT = eltype(law)
  bctag == 1 ? sod(law, SVector(FT(0))) : sod(law, SVector(FT(1))), aux⁻
end

function run(A, FT, N, K; volume_form=WeakForm())
  Nq = N + 1

  law = EulerLaw{FT, 1}(problem = Sod())

  cell = LobattoCell{FT, A}(Nq)
  v1d = range(FT(0), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d,); periodic=(false,))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux())

  cfl = FT(1 // 4)
  dt = cfl * step(v1d) / N / Euler.soundspeed(law, FT(1), FT(1))
  timeend = FT(2 // 10)

  q = fieldarray(undef, law, grid)
  q .= sod.(Ref(law), points(grid))

  odesolver = LSRK54(dg, q, dt)
  solve!(q, timeend, odesolver)

  @pgf if !@isdefined(_testing)
    ρ, ρu, ρe = components(q)
    p = Euler.pressure.(Ref(law), ρ, ρu, ρe)
    u = ρu ./ ρ
    x = vec(first(components(points(grid))))

    fig = @pgf GroupPlot({group_style= {group_size="2 by 2"}})
    ρ_plot = Plot({no_marks}, Coordinates(x, vec(ρ)))
    u_plot = Plot({no_marks}, Coordinates(x, vec(u)))
    E_plot = Plot({no_marks}, Coordinates(x, vec(ρe)))
    p_plot = Plot({no_marks}, Coordinates(x, vec(p)))

    push!(fig, {}, ρ_plot)
    push!(fig, {}, u_plot)
    push!(fig, {}, E_plot)
    push!(fig, {}, p_plot)

    path = mkpath(joinpath("output", "euler", "sod"))
    pgfsave(joinpath(path, "sod.pdf"), fig)
  end
end

let
  A = Array
  FT = Float64
  N = 4
  K = 32
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())

  errf = run(A, FT, N, K; volume_form)
end
