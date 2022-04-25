using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross
using WriteVTK

const _X = 125
const _a = 6.371229e6 / _X

longitude(x⃗) = @inbounds atan(x⃗[2], x⃗[1])
latitude(x⃗) = @inbounds asin(x⃗[3] / norm(x⃗))

struct IsothermalZonalFlow <: AbstractProblem end

import Atum: boundarystate, source!
function boundarystate(law::Union{LinearEulerGravityLaw, EulerGravityLaw}, ::IsothermalZonalFlow, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function cartesian(v⃗, x⃗)
  u, v, w = v⃗
  r = norm(x⃗)
  λ = longitude(x⃗)
  φ = latitude(x⃗)

  uc = -sin(λ) * u - sin(φ) * cos(λ) * v + cos(φ) * cos(λ) * w
  vc =  cos(λ) * u - sin(φ) * sin(λ) * v + cos(φ) * sin(λ) * w
  wc =  cos(φ) * v + sin(φ) * w

  SVector(uc, vc, wc)
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, ::IsothermalZonalFlow, x⃗)
  FT = eltype(law)
  r = norm(x⃗)
  z = r - _a

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  p_s = FT(1e5)
  T_ref = FT(300)

  δ = constants(law).grav / (R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)
  ρ_ref = ρ_s * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_ref

  SVector(ρ_ref, p_ref)
end

function zonalflow(law, x⃗, aux)
  FT = eltype(law)

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  r = norm(x⃗)
  z = r - _a
  λ = longitude(x⃗)
  φ = latitude(x⃗)

  g = constants(law).grav
  Φ = EulerGravity.geopotential(law, aux)

  p₀ = FT(1e5)
  u₀ = FT(20)
  T₀ = FT(300)

  f1 = z
  f2 = z / _a + z^2 / (2 * _a^2)
  shear = 1 + z / _a

  u_sphere = SVector{3, FT}(u₀ * shear * cos(φ), 0, 0)
  u_cart = cartesian(u_sphere, x⃗)

  prefac = u₀^2 / (R_d * T₀)
  fac1 = prefac * f2 * cos(φ)^2
  fac2 = prefac * sin(φ)^2 / 2
  fac3 = g * f1 / (R_d * T₀)
  exparg = fac1 - fac2 - fac3
  p = p₀ * exp(exparg)

  ρ = p / (R_d * T₀)
  ρu⃗ = ρ * u_cart
  ρe = ρ * (cv_d * T₀ + u_cart' * u_cart / 2 + Φ)
  
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, KH, KV; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = EulerGravityLaw{FT, 3}(sphere=true, problem=IsothermalZonalFlow())
  lin_law = LinearEulerGravityLaw(law)
  
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  modeltop = 10e3
  vr = range(FT(_a), stop=FT(_a + modeltop), length=KV+1)
  grid = cubedspheregrid(cell, vr, KH)

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux())

  dg_linear = DGSEM(; law=lin_law, grid, volume_form = WeakForm(),
                    surface_numericalflux = RusanovFlux(),
                    auxstate=dg.auxstate,
                    directions = (3,))

  cfl = FT(3)
  dz = min_node_distance(grid, dims = (3,))
  dt = cfl * dz / FT(330)

  timeend = @isdefined(_testing) ? 10dt : FT(24 * 3600)
  @show ceil(Int, timeend / dt)

  q = fieldarray(undef, law, grid)
  q .= zonalflow.(Ref(law), points(grid), dg.auxstate)
  qref = fieldarray(undef, law, grid)
  qref .= q

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "isothermal_zonal_flow")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end
  count = 0
  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      filename = "KH_$(lpad(KH, 6, '0'))_KV_$(lpad(KV, 6, '0'))_step$(lpad(count, 6, '0'))"
      count += 1
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρw, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρw_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      (vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref))))
      (vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref))))
      (vtkfile["δρw"] = vec(Array(P * (ρw - ρw_ref))))
      (vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref))))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end


  outputvtk && do_output(0, FT(0), q)
  odesolver = ARK23(dg, dg_linear, fieldarray(q), dt;
                    split_rhs = false,
                    paperversion = false)

  solve!(q, timeend, odesolver; after_step=do_output, adjust_final = false)
  outputvtk && vtk_save(pvd)

  errf = weightednorm(dg, q .- qref)
end

let
  A = Array
  FT = Float64
  N = 3
  #volume_form = WeakForm()
  volume_form=FluxDifferencingForm(EntropyConservativeFlux())
  for l in 1:1
    KH = 5 * 2 ^ (l - 1)
    KV = KH
    errf = run(A, FT, N, KH, KV; volume_form)
    @show l, errf
  end
end
