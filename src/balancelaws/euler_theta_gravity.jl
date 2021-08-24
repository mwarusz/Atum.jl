module EulerThetaGravity
  export EulerThetaGravityLaw, γ, grav

  import ..Atum
  using ..Atum: avg, logavg, roe_avg
  using StaticArrays
  using LinearAlgebra: I

  struct EulerThetaGravityLaw{γ, grav, R_d, p_ref, pde, FT, D, S} <: Atum.AbstractBalanceLaw{FT, D, S}
    function EulerThetaGravityLaw{FT, D}(; γ = 7 // 5,
                                           cv_d = 719,
                                           p_ref = 10 ^ 5,
                                           grav = 981 // 100,
                                           pde_level_balance = false) where {FT, D}
      S = 2 + D
      R_d = γ * (cv_d - 1)
      new{FT(γ), FT(R_d), FT(p_ref), FT(grav), pde_level_balance, FT, D, S}()
    end
  end

  γ(::EulerThetaGravityLaw{_γ}) where {_γ} = _γ
  R_d(::EulerThetaGravityLaw{_γ, _R_d}) where {_γ, _R_d} = _R_d
  p_ref(::EulerThetaGravityLaw{_γ, _R_d, _p_ref}) where {_γ, _R_d, _p_ref} = _p_ref
  grav(::EulerThetaGravityLaw{_γ, _R_d, _p_ref, _grav}) where {_γ, _R_d, _p_ref, _grav} = _grav
  pde_level_balance(::EulerThetaGravityLaw{_γ, _R_d, _p_ref, _grav, _pde}) where {_γ, _R_d, _p_ref, _grav, _pde} = _pde

  function varsindices(law::EulerThetaGravityLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρθ = S
    return ix_ρ, ix_ρu⃗, ix_ρθ
  end

  function unpackstate(law::EulerThetaGravityLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρθ = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρθ]
  end

  referencestate(law::EulerThetaGravityLaw, x⃗) = SVector{0, eltype(law)}()
  reference_ρ(law::EulerThetaGravityLaw, aux) = @inbounds aux[ndims(law) + 1]
  reference_p(law::EulerThetaGravityLaw, aux) = @inbounds aux[ndims(law) + 2]

  function Atum.auxiliary(law::EulerThetaGravityLaw, x⃗)
    vcat(x⃗, referencestate(law, x⃗))
  end

  function coordinates(law::EulerThetaGravityLaw, aux)
    aux[SOneTo(ndims(law))]
  end

  function pressure(law::EulerThetaGravityLaw, ρθ)
    α = R_d(law) ^ γ(law) / p_ref(law) ^ (γ(law) - 1)
    α * ρθ ^ γ(law)
  end
  function theta(law::EulerThetaGravityLaw, p)
    α = R_d(law) ^ γ(law) / p_ref(law) ^ (γ(law) - 1)
    (p / α) ^ (1 / γ(law))
  end

  function soundspeed(law::EulerThetaGravityLaw, ρ, ρθ)
    sqrt(γ(law) * pressure(law, ρθ) / ρ)
  end

  function Atum.flux(law::EulerThetaGravityLaw, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρθ)

    δp = p
    if pde_level_balance(law)
      δp -= reference_p(law, aux)
    end

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + δp * I
    fρθ = u⃗ * ρθ

    hcat(fρ, fρu⃗, fρθ)
  end

  function Atum.nonconservative_term!(law::EulerThetaGravityLaw, dq, q, aux)
    ix_ρ, ix_ρu⃗, _ = varsindices(law)

    @inbounds ρ = q[ix_ρ]

    if pde_level_balance(law)
      ρ -= reference_ρ(law, aux)
    end

    @inbounds dq[ix_ρu⃗[end]] -= ρ * grav(law)
  end

  function Atum.wavespeed(law::EulerThetaGravityLaw, n⃗, q, aux)
    ρ, ρu⃗, ρθ = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + soundspeed(law, ρ, ρθ)
  end

  #function Atum.surfaceflux(::Atum.RoeFlux, law::EulerThetaGravityLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
  #  Φ = geopotential(law, aux⁻)

  #  f⁻ = Atum.flux(law, q⁻, aux⁻)
  #  f⁺ = Atum.flux(law, q⁺, aux⁺)

  #  ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
  #  u⃗⁻ = ρu⃗⁻ / ρ⁻
  #  e⁻ = ρe⁻ / ρ⁻
  #  p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ)
  #  h⁻ = e⁻ + p⁻ / ρ⁻
  #  c⁻ = soundspeed(law, ρ⁻, p⁻)

  #  ρ⁺, ρu⃗⁺, ρe⁺ = unpackstate(law, q⁺)
  #  u⃗⁺ = ρu⃗⁺ / ρ⁺
  #  e⁺ = ρe⁺ / ρ⁺
  #  p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ)
  #  h⁺ = e⁺ + p⁺ / ρ⁺
  #  c⁺ = soundspeed(law, ρ⁺, p⁺)

  #  ρ = sqrt(ρ⁻ * ρ⁺)
  #  u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
  #  h = roe_avg(ρ⁻, ρ⁺, h⁻, h⁺)
  #  c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

  #  uₙ = u⃗' * n⃗

  #  Δρ = ρ⁺ - ρ⁻
  #  Δp = p⁺ - p⁻
  #  Δu⃗ = u⃗⁺ - u⃗⁻
  #  Δuₙ = Δu⃗' * n⃗

  #  c⁻² = 1 / c^2
  #  w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
  #  w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
  #  w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
  #  w4 = abs(uₙ) * ρ

  #  fp_ρ = (w1 + w2 + w3) / 2
  #  fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) +
  #           w2 * (u⃗ + c * n⃗) +
  #           w3 * u⃗ +
  #           w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
  #  fp_ρe = (w1 * (h - c * uₙ) +
  #           w2 * (h + c * uₙ) +
  #           w3 * (u⃗' * u⃗ / 2 + Φ) +
  #           w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) / 2

  #  (f⁻ + f⁺)' * n⃗ / 2 - SVector(fp_ρ, fp_ρu⃗..., fp_ρe)
  #end

  #function Atum.twopointflux(::Atum.EntropyConservativeFlux,
  #                           law::EulerThetaGravityLaw,
  #                           q₁, aux₁, q₂, aux₂)
  #    FT = eltype(law)
  #    ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
  #    ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

  #    Φ₁ = geopotential(law, aux₁)
  #    u⃗₁ = ρu⃗₁ / ρ₁
  #    p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)
  #    b₁ = ρ₁ / 2p₁

  #    Φ₂ = geopotential(law, aux₂)
  #    u⃗₂ = ρu⃗₂ / ρ₂
  #    p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)
  #    b₂ = ρ₂ / 2p₂

  #    ρ_avg = avg(ρ₁, ρ₂)
  #    u⃗_avg = avg(u⃗₁, u⃗₂)
  #    b_avg = avg(b₁, b₂)
  #    Φ_avg = avg(Φ₁, Φ₂)

  #    u²_avg = avg(u⃗₁' * u⃗₁, u⃗₂' * u⃗₂)
  #    ρ_log = logavg(ρ₁, ρ₂)
  #    b_log = logavg(b₁, b₂)

  #    fρ = u⃗_avg * ρ_log
  #    fρu⃗ = u⃗_avg * fρ' + ρ_avg / 2b_avg * I
  #    fρe = (1 / (2 * (γ(law) - 1) * b_log) - u²_avg / 2 + Φ_avg) * fρ + fρu⃗ * u⃗_avg

  #    # fluctuation
  #    α = b_avg * ρ_log / 2b₁
  #    fρu⃗ -= α * (Φ₁ - Φ₂) * I

  #    hcat(fρ, fρu⃗, fρe)
  #end

  #function Atum.twopointflux(::Atum.KennedyGruberFlux,
  #                           law::EulerThetaGravityLaw,
  #                           q₁, aux₁, q₂, aux₂)
  #    FT = eltype(law)
  #    ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
  #    ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

  #    Φ₁ = geopotential(law, aux₁)
  #    u⃗₁ = ρu⃗₁ / ρ₁
  #    e₁ = ρe₁ / ρ₁
  #    p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)

  #    Φ₂ = geopotential(law, aux₂)
  #    u⃗₂ = ρu⃗₂ / ρ₂
  #    e₂ = ρe₂ / ρ₂
  #    p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)

  #    ρ_avg = avg(ρ₁, ρ₂)
  #    u⃗_avg = avg(u⃗₁, u⃗₂)
  #    e_avg = avg(e₁, e₂)
  #    p_avg = avg(p₁, p₂)

  #    fρ = u⃗_avg * ρ_avg
  #    fρu⃗ = u⃗_avg * fρ' + p_avg * I
  #    fρe = u⃗_avg * (ρ_avg * e_avg + p_avg)

  #    # fluctuation
  #    α = ρ_avg / 2
  #    fρu⃗ -= α * (Φ₁ - Φ₂) * I

  #    hcat(fρ, fρu⃗, fρe)
  #end
end
