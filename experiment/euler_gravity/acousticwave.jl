using Atum
using Atum.EulerGravity

#using CUDA
using LinearAlgebra: norm
using StaticArrays: SVector
using WriteVTK

import Atum: boundarystate
function boundarystate(law::Union{EulerGravityLaw, LinearEulerGravityLaw},
        n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, x⃗)
  FT = eltype(law)
  x, y, z = x⃗

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  p_s = FT(1e5)
  T_ref = FT(250)

  δ = constants(law).grav / (R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)
  ρ_ref = ρ_s * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_ref

  SVector(ρ_ref, p_ref)
end

function acousticwave(law, x⃗, t, add_perturbation=true)
    ρ_ref, _ = referencestate(law, x⃗)
end

function run(A, FT, N, Kh, Kv; volume_form=WeakForm(), outputvtk=true,
        useark = true, earth_radius = 6_371_220, height = 10_000)
  Nq = N + 1
  dim = 3

  law = EulerGravityLaw{FT, dim}(pde_level_balance=volume_form isa WeakForm)
  linlaw = LinearEulerGravityLaw(law)

  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  vert_coord = range(FT(earth_radius), stop=FT(earth_radius + height),
                     length=Kv+1)
  grid = cubedspheregrid(cell, vert_coord, Kh)

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux())
  dg_linear = DGSEM(; law = linlaw, grid,
                    volume_form = volume_form isa FluxDifferencingForm ?
                    FluxDifferencingForm(CentralFlux()) : volume_form,
                    surface_numericalflux = RusanovFlux(),
                    auxstate=dg.auxstate,
                    directions = (dim,))

  #=
  cfl = FT(0.75)
  dt = cfl * step(vz) / N / 330
  timeend = @isdefined(_testing) ? 10dt : FT(30 * 60)
  timeend = FT(100)
  nsteps = ceil(Int, timeend / dt)
  dt = timeend / nsteps
  =#

  q = fieldarray(undef, law, grid)
  # q .= acousticwave.(Ref(law), points(grid), FT(0), false)
  qref = fieldarray(undef, law, grid)
  qref .= referencestate.(Ref(law), points(grid))
  return nothing

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "acousticwave")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  count = 0
  do_output = function(step, time, q)
    if outputvtk # && step % ceil(Int, timeend / 100 / dt) == 0
      filename = "Kv_$(lpad(Kv, 6, '0'))_Kh_$(lpad(Kh, 6, '0'))_useark_$(useark)_step$(lpad(count, 6, '0'))"
      count += 1
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv_ref)))
      vtkfile["δρe"] = vec(Array(P * (ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end
  outputvtk && do_output(0, FT(0), q)

  #=
  # odesolver = LSRK54(dg, q, dt)
  odesolver = ARK23(dg, useark ? dg_linear : nothing, fieldarray(q), dt;
                    split_rhs = false,
                    paperversion = false)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output, adjust_final = false)
  outputvtk && vtk_save(pvd)
  =#

  normf = map(components(q)) do f
    sqrt(sum(dg.MJ .* f .^ 2))
  end
  norm(normf)
end

let
  A = Array
  FT = Float64
  N = 4

  ndof_x = 60
  ndof_y = 15
  KX_base = round(Int, ndof_x / N)
  KY_base = round(Int, ndof_y / N)

  nlevels = @isdefined(_testing) ? 1 : 2
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())
  volume_form = WeakForm()
  for useark in (true, false)
    for l in 1:nlevels
      KX = KX_base * 2 ^ (l - 1)
      KY = KY_base * 2 ^ (l - 1)
      normf = run(A, FT, N, KX, KY; useark, volume_form)
      @show normf
    end
  end
end
