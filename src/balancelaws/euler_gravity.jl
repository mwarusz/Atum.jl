module EulerGravity
  export EulerGravityLaw, γ, grav

  import ..Atum
  using ..Atum: avg, logavg, roe_avg
  using StaticArrays
  using LinearAlgebra: I

  struct EulerGravityLaw{γ, grav, pde, FT, D, S} <: Atum.AbstractBalanceLaw{FT, D, S}
    function EulerGravityLaw{FT, D}(; γ = 7 // 5,
                                      grav = 981 // 100,
                                      pde_level_balance = false) where {FT, D}
      S = 2 + D
      new{FT(γ), FT(grav), pde_level_balance, FT, D, S}()
    end
  end

  γ(::EulerGravityLaw{_γ}) where {_γ} = _γ
  grav(::EulerGravityLaw{_γ, _grav}) where {_γ, _grav} = _grav
  pde_level_balance(::EulerGravityLaw{_γ, _grav, _pde}) where {_γ, _grav, _pde} = _pde

  function varsindices(law::EulerGravityLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
  end

  function unpackstate(law::EulerGravityLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
  end

  referencestate(law::EulerGravityLaw, x⃗) = SVector{0, eltype(law)}()
  reference_ρ(law::EulerGravityLaw, aux) = @inbounds aux[ndims(law) + 1]
  reference_p(law::EulerGravityLaw, aux) = @inbounds aux[ndims(law) + 2]

  function Atum.auxiliary(law::EulerGravityLaw, x⃗)
    vcat(x⃗, referencestate(law, x⃗))
  end

  function coordinates(law::EulerGravityLaw, aux)
    aux[SOneTo(ndims(law))]
  end

  function geopotential(law, aux)
    x⃗ = coordinates(law, aux)
    z = last(x⃗)
    grav(law) * z
  end

  function pressure(law::EulerGravityLaw, ρ, ρu⃗, ρe, Φ)
    (γ(law) - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
  end
  function energy(law::EulerGravityLaw, ρ, ρu⃗, p, Φ)
    p / (γ(law) - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ
  end
  function soundspeed(law::EulerGravityLaw, ρ, p)
    sqrt(γ(law) * p / ρ)
  end
  function soundspeed(law::EulerGravityLaw, ρ, ρu⃗, ρe, Φ)
    soundspeed(law, ρ, pressure(law, ρ, ρu⃗, ρe, Φ))
  end

  function Atum.flux(law::EulerGravityLaw, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    Φ = geopotential(law, aux)

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρ, ρu⃗, ρe, Φ)

    δp = p
    if pde_level_balance(law)
      δp -= reference_p(law, aux)
    end

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + δp * I
    fρe = u⃗ * (ρe + p)

    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.nonconservative_term!(law::EulerGravityLaw, dq, q, aux)
    ix_ρ, ix_ρu⃗, _ = varsindices(law)

    @inbounds ρ = q[ix_ρ]

    if pde_level_balance(law)
      ρ -= reference_ρ(law, aux)
    end

    @inbounds dq[ix_ρu⃗[end]] -= ρ * grav(law)
  end

  function Atum.wavespeed(law::EulerGravityLaw, n⃗, q, aux)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    Φ = geopotential(law, aux)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + soundspeed(law, ρ, ρu⃗, ρe, Φ)
  end

  function Atum.surfaceflux(::Atum.RoeFlux, law::EulerGravityLaw, n⃗, q⁻, aux⁻, q⁺, aux⁺)
    Φ = geopotential(law, aux⁻)

    f⁻ = Atum.flux(law, q⁻, aux⁻)
    f⁺ = Atum.flux(law, q⁺, aux⁺)

    ρ⁻, ρu⃗⁻, ρe⁻ = unpackstate(law, q⁻)
    u⃗⁻ = ρu⃗⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = pressure(law, ρ⁻, ρu⃗⁻, ρe⁻, Φ)
    h⁻ = e⁻ + p⁻ / ρ⁻
    c⁻ = soundspeed(law, ρ⁻, p⁻)

    ρ⁺, ρu⃗⁺, ρe⁺ = unpackstate(law, q⁺)
    u⃗⁺ = ρu⃗⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = pressure(law, ρ⁺, ρu⃗⁺, ρe⁺, Φ)
    h⁺ = e⁺ + p⁺ / ρ⁺
    c⁺ = soundspeed(law, ρ⁺, p⁺)

    ρ = sqrt(ρ⁻ * ρ⁺)
    u⃗ = roe_avg(ρ⁻, ρ⁺, u⃗⁻, u⃗⁺)
    h = roe_avg(ρ⁻, ρ⁺, h⁻, h⁺)
    c = roe_avg(ρ⁻, ρ⁺, c⁻, c⁺)

    uₙ = u⃗' * n⃗

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu⃗ = u⃗⁺ - u⃗⁻
    Δuₙ = Δu⃗' * n⃗

    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * c⁻² / 2
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * c⁻² / 2
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ

    fp_ρ = (w1 + w2 + w3) / 2
    fp_ρu⃗ = (w1 * (u⃗ - c * n⃗) +
             w2 * (u⃗ + c * n⃗) +
             w3 * u⃗ +
             w4 * (Δu⃗ - Δuₙ * n⃗)) / 2
    fp_ρe = (w1 * (h - c * uₙ) +
             w2 * (h + c * uₙ) +
             w3 * (u⃗' * u⃗ / 2 + Φ) +
             w4 * (u⃗' * Δu⃗ - uₙ * Δuₙ)) / 2

    (f⁻ + f⁺)' * n⃗ / 2 - SVector(fp_ρ, fp_ρu⃗..., fp_ρe)
  end

  function Atum.twopointflux(::Atum.EntropyConservativeFlux,
                             law::EulerGravityLaw,
                             q₁, aux₁, q₂, aux₂)
      FT = eltype(law)
      ρ₁, ρu⃗₁, ρe₁ = unpackstate(law, q₁)
      ρ₂, ρu⃗₂, ρe₂ = unpackstate(law, q₂)

      Φ₁ = geopotential(law, aux₁)
      u⃗₁ = ρu⃗₁ / ρ₁
      p₁ = pressure(law, ρ₁, ρu⃗₁, ρe₁, Φ₁)
      b₁ = ρ₁ / 2p₁

      Φ₂ = geopotential(law, aux₂)
      u⃗₂ = ρu⃗₂ / ρ₂
      p₂ = pressure(law, ρ₂, ρu⃗₂, ρe₂, Φ₂)
      b₂ = ρ₂ / 2p₂

      ρ_avg = avg(ρ₁, ρ₂)
      u⃗_avg = avg(u⃗₁, u⃗₂)
      b_avg = avg(b₁, b₂)
      Φ_avg = avg(Φ₁, Φ₂)

      u²_avg = avg(u⃗₁' * u⃗₁, u⃗₂' * u⃗₂)
      ρ_log = logavg(ρ₁, ρ₂)
      b_log = logavg(b₁, b₂)

      fρ = u⃗_avg * ρ_log
      fρu⃗ = u⃗_avg * fρ' + ρ_avg / 2b_avg * I
      fρe = (1 / (2 * (γ(law) - 1) * b_log) - u²_avg / 2 + Φ_avg) * fρ + fρu⃗ * u⃗_avg

      # fluctuation
      α = b_avg * ρ_log / 2b₁
      fρu⃗ -= α * (Φ₁ - Φ₂) * I

      hcat(fρ, fρu⃗, fρe)
  end
end
