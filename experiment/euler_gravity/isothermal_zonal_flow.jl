using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using LinearAlgebra: norm, cross
using WriteVTK

const _X = 125
const _a = 6.371229e6 / _X

longitude(x⃗) = @inbounds atan(x⃗[2], x⃗[1])
latitude(x⃗) = @inbounds asin(x⃗[3] / norm(x⃗))

import Atum: boundarystate, source!
function boundarystate(law::EulerGravityLaw, n⃗, q⁻, aux⁻, _)
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
  u₀ = FT(0)
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

  law = EulerGravityLaw{FT, 3}(sphere=true)
  
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)
  modeltop = 10e3
  vr = range(FT(_a), stop=FT(_a + modeltop), length=KV)
  grid = cubedspheregrid(cell, vr, KH)

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux())

  cfl = FT(1 // 4)
  dt = cfl * min_node_distance(grid) / FT(330)
  timeend = @isdefined(_testing) ? 10dt : FT(600)
  @show ceil(Int, timeend / dt)

  q = zonalflow.(Ref(law), points(grid), dg.auxstate)
  qref = similar(q)
  qref .= q

  check_pulse = function(step, time, q)
    if step % 10 == 0
      δρ, δρu, δρv, δρw, δρe = Array.(components(q .- qref))
      @show extrema(δρ)
      @show extrema(δρu)
      @show extrema(δρv)
      @show extrema(δρw)
      @show extrema(δρe)
    end
  end

  odesolver = LSRK54(dg, q, dt)
  #solve!(q, timeend, odesolver; after_step=check_pulse)
  solve!(q, timeend, odesolver)

  errf = weightednorm(dg, q .- qref)
end

let
  A = Array
  FT = Float64
  N = 3
  volume_form = WeakForm()
  for l in 1:2
    KH = 5 * 2 ^ (l - 1)
    KV = KH
    errf = run(A, FT, N, KH, KV; volume_form)
    @show l, errf
  end
end
