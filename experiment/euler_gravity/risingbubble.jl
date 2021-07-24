using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using WriteVTK

import Atum: boundarystate
function boundarystate(law::EulerGravityLaw, n⃗, x⃗, q⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺)
end

function risingbubble(law, x⃗, add_perturbation=true)
  FT = eltype(law)
  x, z = x⃗

  Φ = grav(law) * z

  cv_d = FT(719)
  cp_d = γ(law) * cv_d
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
  if add_perturbation 
    θ += δθ * (1 + cos(π * r / rc)) / 2
  end
  π_exner = 1 - grav(law) / (cp_d * θ) * z
  ρ = p0 / (R_d * θ) * π_exner ^ (cv_d / R_d)
  
  ρu = FT(0)
  ρv = FT(0)

  T = θ * π_exner
  ρe = ρ * (cv_d * T + Φ)
  
  SVector(ρ, ρu, ρv, ρe)
end

function run(A, FT, N, K; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = EulerGravityLaw{FT, 2}()
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(1e3), length=K+1)
  vz = range(FT(0), stop=FT(1e3), length=K+1)
  grid = brickgrid(cell, (vx, vz); periodic = (true, false))

  dg = DGSEM(; law, cell, grid, volume_form,
               surface_numericalflux = RoeFlux())

  cfl = FT(1 // 3)
  dt = cfl * step(vz) / N / 330
  timeend = @isdefined(_testing) ? 10dt : FT(500)
 
  q = risingbubble.(Ref(law), points(grid))
  qref = risingbubble.(Ref(law), points(grid), false)

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "risingbubble")
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

  odesolver = LSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output)
  outputvtk && vtk_save(pvd)
end

let
  A = Array
  FT = Float64
  N = 4
  K = 10
  run(A, FT, N, K)
end
