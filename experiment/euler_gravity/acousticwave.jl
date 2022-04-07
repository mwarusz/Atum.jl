using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross
using WriteVTK

const _X = 1
const _a = 6.371229e6 / _X
const _H = 10e3

longitude(x⃗) = @inbounds atan(x⃗[2], x⃗[1])
latitude(x⃗) = @inbounds asin(x⃗[3] / norm(x⃗))

import Atum: boundarystate, source!
function boundarystate(law::Union{EulerGravityLaw, LinearEulerGravityLaw}, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, x⃗)
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

function acousticwave(law, x⃗, aux, add_perturbation=true)
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

  T₀ = FT(300)
  H = FT(_H)

  α = FT(3)
  γ = FT(100)
  nv = 1
  β = min(FT(1), α * acos(cos(φ) * cos(λ)))
  f = (1 + cos(FT(π) * β)) / 2
  g = sin(nv * FT(π) * z / H)
  Δp = γ * f * g
  
  p = EulerGravity.reference_p(law, aux)
  if add_perturbation
    p += Δp
  end

  ρ = p / (R_d * T₀)
  ρu⃗ = SVector{3, FT}(0, 0, 0)
  ρe = ρ * (cv_d * T₀ + Φ)
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, KH, KV; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = EulerGravityLaw{FT, 3}(sphere=true)
  lin_law = LinearEulerGravityLaw(law)
  
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  vr = range(FT(_a), stop=FT(_a + _H), length=KV+1)
  grid = cubedspheregrid(cell, vr, KH)

  dg = DGSEM(; law, grid,
               volume_form,
               surface_numericalflux = RusanovFlux())

  dg_linear = DGSEM(; law=lin_law, grid,
                    volume_form = WeakForm(),
                    surface_numericalflux = RusanovFlux(),
                    auxstate=dg.auxstate,
                    directions = (3,))


  element_size = (_H / KV)
  acoustic_speed = FT(330)
  dt_factor = 100
  dt = dt_factor * element_size / acoustic_speed / N^2

  timeend = @isdefined(_testing) ? 10dt : FT(33 * 60 * 60)
  @show ceil(Int, timeend / dt)

  q = fieldarray(undef, law, grid)
  q .= acousticwave.(Ref(law), points(grid), dg.auxstate)
  qref = fieldarray(undef, law, grid)
  qref .= acousticwave.(Ref(law), points(grid), dg.auxstate, false)

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "acousticwave")
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
end

let
  A = Array
  FT = Float64
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())
  
  N = 5
  KH = 10
  KV = 5
  run(A, FT, N, KH, KV; volume_form)
end
