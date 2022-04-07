using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross
using WriteVTK

const _X = 1
const _a = 6.371229e6 / _X
const _Ω = 7.29212e-5 * _X
const _p_0 = 1e5
const _grav = 9.80616

longitude(x⃗) = @inbounds atan(x⃗[2], x⃗[1])
latitude(x⃗) = @inbounds asin(x⃗[3] / norm(x⃗))

import Atum: boundarystate, source!
function boundarystate(law::Union{LinearEulerGravityLaw, EulerGravityLaw}, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

function source!(law::EulerGravityLaw, dq, q, aux, dim, directions)
  if dim ∈ directions
    FT = eltype(law)
    _, ix_ρu⃗, _ = EulerGravity.varsindices(law)
    @inbounds ρu⃗ = q[ix_ρu⃗]
    @inbounds dq[ix_ρu⃗] .-= cross(SVector{3, FT}(0, 0, 2_Ω), ρu⃗)
  end
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
function referencestate(law::EulerGravityLaw, x⃗)
  FT = eltype(law)
  r = norm(x⃗)
  z = r - _a

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  p_s = FT(_p_0)
  T_ref = FT(300)

  δ = constants(law).grav / (R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)
  ρ_ref = ρ_s * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_ref

  SVector(ρ_ref, p_ref)
end

function baroclinicwave(law, x⃗, aux, add_perturbation=true)
  FT = eltype(law)
  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d
  grav = constants(law).grav
  p_0 = FT(_p_0)
  Ω = FT(_Ω)
  a = FT(_a)
  Φ = EulerGravity.geopotential(law, aux)

  k = FT(3)
  T_E = FT(310)
  T_P = FT(240)
  T_0 = (T_E + T_P) / 2
  Γ = FT(0.005)
  A = 1 / Γ
  B = (T_0 - T_P) / T_0 / T_P
  C = (k + 2) / 2 * (T_E - T_P) / T_E / T_P
  b = 2
  H = R_d * T_0 / grav
  z_t = FT(15e3)
  λ_c = FT(π / 9)
  φ_c = FT(2 * π / 9)
  d_0 = FT(a / 6)
  V_p = FT(1)
  
  r = norm(x⃗)
  z = r - a
  λ = longitude(x⃗)
  φ = latitude(x⃗)

  γ = FT(1) # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

  # convenience functions for temperature and pressure
  τ_z_1 = exp(Γ * z / T_0)
  τ_z_2 = 1 - 2 * (z / b / H)^2
  τ_z_3 = exp(-(z / b / H)^2)
  τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
  τ_2 = C * τ_z_2 * τ_z_3
  τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
  τ_int_2 = C * z * τ_z_3
  I_T = (cos(φ) * (1 + γ * z / _a))^k -
      k / (k + 2) * (cos(φ) * (1 + γ * z / a))^(k + 2)

  # base state virtual temperature, pressure, specific humidity, density
  T = (τ_1 - τ_2 * I_T)^(-1)
  p = p_0 * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

  # base state velocity
  U =
      grav * k / a *
      τ_int_2 *
      T *
      (
          (cos(φ) * (1 + γ * z / a))^(k - 1) -
          (cos(φ) * (1 + γ * z / a))^(k + 1)
      )
  u_ref =
      -Ω * (a + γ * z) * cos(φ) +
      sqrt((Ω * (a + γ * z) * cos(φ))^2 + (a + γ * z) * cos(φ) * U)
  v_ref = 0
  w_ref = 0

  # velocity perturbations
  F_z = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
  if z > z_t
      F_z = FT(0)
  end
  d = a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
  c3 = cos(π * d / 2 / d_0)^3
  s1 = sin(π * d / 2 / d_0)
  if 0 < d < d_0 && d != FT(a * π)
      u′ =
          -16 * V_p / 3 / sqrt(3) *
          F_z *
          c3 *
          s1 *
          (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
          sin(d / a)
      v′ =
          16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
          sin(d / a)
  else
      u′ = FT(0)
      v′ = FT(0)
  end
  w′ = FT(0)
 
  if add_perturbation
    u⃗_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
  else
    u⃗_sphere = SVector{3, FT}(u_ref, v_ref, w_ref)
  end
  u⃗ = cartesian(u⃗_sphere, x⃗)

  ρ = p / (R_d * T)
  ρu⃗ = ρ * u⃗
  ρe = ρ * (cv_d * T + u⃗' * u⃗ / 2 + Φ)
  
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, KH, KV; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = EulerGravityLaw{FT, 3}(sphere=true, grav=_grav)
  lin_law = LinearEulerGravityLaw(law)
  
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  modeltop = 30e3
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

  #cfl = FT(0.1)
  #dt = cfl * min_node_distance(grid, dims=(1, 2)) / FT(330)
  timeend = @isdefined(_testing) ? 10dt : FT(1 * 24 * 3600)
  @show ceil(Int, timeend / dt)

  q = fieldarray(undef, law, grid)
  q .= baroclinicwave.(Ref(law), points(grid), dg.auxstate, true)

  qref = similar(q)
  qref .= baroclinicwave.(Ref(law), points(grid), dg.auxstate, false)

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "baroclinicwave_test")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end
  count = 0
  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      @show step, time
      filename = "KH_$(lpad(KH, 6, '0'))_KV_$(lpad(KV, 6, '0'))_step$(lpad(count, 6, '0'))"
      count += 1
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρw, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρw_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["δρw"] = vec(Array(P * (ρw - ρw_ref)))
      vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref)))
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
end

let
  A = Array
  FT = Float64
  N = 3
  
  #volume_form = WeakForm()
  volume_form=FluxDifferencingForm(EntropyConservativeFlux())

  KH = 8
  KV = 4
  run(A, FT, N, KH, KV; volume_form)
end
