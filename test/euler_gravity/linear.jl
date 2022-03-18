using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using WriteVTK

import Atum: boundarystate
function boundarystate(law::Union{LinearEulerGravityLaw, EulerGravityLaw},
                       n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, x⃗)
  FT = eltype(law)
  x, z = x⃗

  Φ = constants(law).grav * z

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  θref = FT(300)
  p0 = FT(1e5)
  xc = FT(500)
  zc = FT(350)
  rc = FT(250)
  δθc = FT(1 / 2)

  r = sqrt((x - xc) ^ 2 + (z - zc) ^ 2)
  δθ = r <= rc ? δθc : zero(FT)

  θ = θref
  π_exner = 1 - constants(law).grav / (cp_d * θ) * z
  ρ = p0 / (R_d * θ) * π_exner ^ (cv_d / R_d)

  ρu⃗ = SVector(FT(0), FT(0))

  T = θ * π_exner
  ρe = ρ * (cv_d * T + Φ)

  SVector(ρ, EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ))
end


function initialcondition(law, x⃗)
  FT = eltype(law)
  x, z = x⃗

  Φ = constants(law).grav * z

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  θref = FT(300)
  p0 = FT(1e5)
  xc = FT(500)
  zc = FT(350)
  rc = FT(250)
  δθc = FT(1 / 2)

  r = sqrt((x - xc) ^ 2 + (z - zc) ^ 2)
  δθ = r <= rc ? δθc : zero(FT)

  θ = θref
  π_exner = 1 - constants(law).grav / (cp_d * θ) * z
  ρ = p0 / (R_d * θ) * π_exner ^ (cv_d / R_d)

  ρu = FT(0)
  ρv = FT(0)

  T = θ * π_exner
  ρe = ρ * (cv_d * T + Φ)

  SVector(ρ, ρu, ρv, ρe)
end

function run(A, FT, N, Kh, Kv; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  dim = 2
  law = LinearEulerGravityLaw(EulerGravityLaw{FT, dim}())
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(1e3), length=Kh+1)
  vz = range(FT(0), stop=FT(1e3), length=Kv+1)
  grid = brickgrid(cell, (vx, vz); periodic = (true, false),
                   ordering = StackedOrdering{CartesianOrdering}())


  dg_nonlinear = DGSEM(; law=parent(law), grid, volume_form,
                       surface_numericalflux = RusanovFlux())
  dg_linear = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux(),
                    auxstate=dg_nonlinear.auxstate,
                    directions = (dim,))

  cfl_h = FT(1 // 4)
  cfl_v = FT(0.775)
  # cfl_v = FT(0.8)
  dt_h = cfl_h * step(vx) / N / 330
  dt_v = cfl_v * step(vz) / N / 330
  dt = min(dt_h, dt_v)
  timeend = @isdefined(_testing) ? 10dt : FT(500)
  timeend = 100dt
 
  q = fieldarray(undef, law, grid)
  q .= initialcondition.(Ref(law), points(grid))
  qref = fieldarray(q)
  qref .= q

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "linear")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0 
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρe_ref = components(qref)
      vtkfile["ρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["ρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["ρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["ρe"] = vec(Array(P * (ρe - ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = ARK23(dg_nonlinear, dg_linear, fieldarray(q), dt;
                    split_rhs = false,
                    paperversion = false)
  #=
  odesolver = ARK23(nothing, dg_linear, fieldarray(q), dt;
                    split_rhs = false,
                    paperversion = false)
  odesolver = ARK23(dg_nonlinear, nothing, fieldarray(q), dt;
                    split_rhs = false,
                    paperversion = false)
  =#

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output)
  outputvtk && vtk_save(pvd)
end

let
  A = Array
  FT = Float64
  N = 4
  Kh = 10
  Kv = 100
  run(A, FT, N, Kh, Kv)
end
