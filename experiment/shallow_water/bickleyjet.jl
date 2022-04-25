using Atum
using Atum.ShallowWater
using Bennu: fieldarray

using StaticArrays: SVector
using WriteVTK

struct BickleyJet <: AbstractProblem end

import Atum: boundarystate
function boundarystate(law::ShallowWaterLaw, ::BickleyJet, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρθ⁻ = ShallowWater.unpackstate(law, q⁻)
  ρ⁺, ρθ⁺ = ρ⁻, ρθ⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρθ⁺), aux⁻
end

function bickleyjet(law, x⃗)
  FT = eltype(law)
  x, y = x⃗

  ϵ = FT(1 / 10)
  l = FT(1 / 2)
  k = FT(1 / 2)

  U = cosh(y)^(-2)

  Ψ = exp(-(y + l / 10)^2 / (2 * (l^2))) * cos(k * x) * cos(k * y)

  u = Ψ * (k * tan(k * y) + y / (l^2))
  v = -Ψ * k * tan(k * x)

  ρ = FT(1)
  ρu = ρ * (U + ϵ * u)
  ρv = ρ * (ϵ * v)
  ρθ = ρ * sin(k * y)

  SVector(ρ, ρu, ρv, ρθ)
end

function run(A, FT, N, K; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = ShallowWaterLaw{FT, 2}(problem=BickleyJet())
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  v1d = range(FT(-2π), stop=FT(2π), length=K+1)
  grid = brickgrid(cell, (v1d, v1d); periodic = (true, false))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RoeFlux())

  cfl = FT(1 // 8)
  dt = cfl * step(v1d) / N / sqrt(constants(law).grav)
  timeend = @isdefined(_testing) ? 10dt : FT(200)
 
  q = fieldarray(undef, law, grid)
  q .= bickleyjet.(Ref(law), points(grid))

  if outputvtk
    vtkdir = joinpath("output", "shallow_water", "bickleyjet")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0 
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρθ = last(components(q))
      vtkfile["ρθ"] = vec(Array(P * ρθ))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = LSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output)
  outputvtk && vtk_save(pvd)
end

let
  A = Array
  FT = Float64
  N = 3

  K = 16
  run(A, FT, N, K)
end
