using Atum
using Atum.EulerGravity

using LinearAlgebra: norm
using StaticArrays: SVector

const _ΔT = 1e-3
const _L = 300e3
const _H = 10e3

function gas_constant(law)
  FT = eltype(law)
  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d
end

import Atum: boundarystate
function boundarystate(law::EulerGravityLaw, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, x⃗)
  FT = eltype(law)
  x, z = x⃗

  R_d = gas_constant(law)

  p_s = FT(1e5)
  T_ref = FT(250)

  δ = constants(law).grav / (R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)
  ρ_ref = ρ_s * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_ref

  SVector(ρ_ref, p_ref)
end

function calculate_diagnostics(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = constants(law).grav * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  ρ_ref, p_ref = referencestate(law, x⃗)
  T_ref = p_ref / (R_d * ρ_ref)

  w = ρw / ρ
  δT = T - T_ref

  SVector(w, δT)
end

function gravitywave(law, x⃗, t, add_perturbation=true)
  FT = eltype(law)
  x, z = x⃗

  Φ = constants(law).grav * z

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  ΔT = FT(_ΔT)
  L = FT(_L)
  H = FT(_H)
  f = FT(0)
  d = FT(5e3)
  x_c = FT(100e3)
  u_0 = FT(20)
  p_s = FT(1e5)
  T_ref = FT(250)

  g = constants(law).grav
  δ = g / (R_d * T_ref)
  c_s = sqrt(cp_d / cv_d * R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)

  if t == 0
    δT_b = ΔT * exp(-(x - x_c) ^ 2 / d ^ 2) * sin(π * z / H)
    δT = exp(δ * z / 2) * δT_b
    δρ_b = -ρ_s * δT_b / T_ref
    δρ = exp(-δ * z / 2) * δρ_b
    δu, δv, δw = 0, 0, 0
  else
    xp = x - u_0 * t

    δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5, Complex{FT}})
    for m in (-1, 1)
      for n in -100:100
        k_x = 2π * n / L
        k_z = π * m / H

        p_1 = c_s ^ 2 * (k_x ^ 2 + k_z ^ 2 + δ ^ 2 / 4) + f ^ 2
        q_1 = g * k_x ^ 2 * (c_s ^ 2 * δ - g) + c_s ^ 2 * f ^ 2 * (k_z ^ 2 + δ ^ 2 / 4)
        
        α = sqrt(p_1 / 2 - sqrt(p_1 ^ 2 / 4 - q_1))
        β = sqrt(p_1 / 2 + sqrt(p_1 ^ 2 / 4 - q_1))

        fac1 = 1 / (β ^ 2 - α ^ 2) 
        L_m1 = (-cos(α * t) / α ^ 2 + cos(β * t) / β ^ 2) * fac1 + 1 / (α ^ 2 * β ^ 2)
        L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
        L_1 = (cos(α * t) - cos(β * t)) * fac1
        L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
        L_3 = (-α ^ 2 * cos(α * t) + β ^ 2 * cos(β * t)) * fac1
        
        if α == 0
          L_m1 = (β ^ 2 * t ^ 2 - 1 + cos(β * t)) / β ^ 4
          L_0 = (β * t - sin(β * t)) / β ^ 3
        end
    
        δρ̃_b0 = -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                exp(-d ^ 2 * k_x ^ 2 / 4) * exp(-im * k_x * x_c) * k_z * H / 2im

        δρ̃_b = (L_3 + (p_1 + g * (im * k_z - δ / 2)) * L_1 +
              (c_s ^ 2 * (k_z ^ 2 + δ ^ 2 / 4) + g * (im * k_z - δ / 2)) * f ^ 2 * L_m1) * δρ̃_b0

        δp̃_b = -(g - c_s ^ 2 * (im * k_z + δ / 2)) * (L_1 + f ^ 2 * L_m1) * g * δρ̃_b0

        δũ_b = im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

        δṽ_b = -f * im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 / ρ_s 

        δw̃_b = -(L_2 + (f ^ 2 + c_s ^ 2 * k_x ^ 2) * L_0) * g * δρ̃_b0 / ρ_s 

        expfac = exp(im * (k_x * xp + k_z * z)) 
        
        δρ_b += δρ̃_b * expfac
        δp_b += δp̃_b * expfac

        δu_b += δũ_b * expfac
        δv_b += δṽ_b * expfac
        δw_b += δw̃_b * expfac
      end
    end

    δρ = exp(-δ * z / 2) * real(δρ_b)
    δp = exp(-δ * z / 2) * real(δp_b)

    δu = exp(δ * z / 2) * real(δu_b)
    δv = exp(δ * z / 2) * real(δv_b)
    δw = exp(δ * z / 2) * real(δw_b)

    δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
    δT = exp(δ * z / 2) * real(δT_b)
  end
  
  ρ_ref = ρ_s * exp(-δ * z)
  
  ρ = ρ_ref
  T = T_ref
  u = u_0
  w = FT(0)
  
  if add_perturbation
    ρ += δρ
    T += δT
    u += δu
    w += δw
  end

  e_kin = (u ^ 2 + w ^ 2) / 2
  e_int = cv_d * T
  ρe = ρ * (e_int + e_kin + Φ)

  return SVector(ρ, ρ * u, ρ * w, ρe)
end
