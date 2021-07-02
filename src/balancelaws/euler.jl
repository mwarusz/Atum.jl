module Euler
  export EulerLaw, γ

  import ..Atum
  using StaticArrays
  using LinearAlgebra: I

  struct EulerLaw{γ, FT, D, S} <: Atum.AbstractBalanceLaw{FT, D, S}
    function EulerLaw{FT, D}(; γ = 7 // 5) where {FT, D}
      S = 2 + D
      new{FT(γ), FT, D, S}()
    end
  end

  γ(::EulerLaw{_γ}) where {_γ} = _γ

  function varsindices(law::EulerLaw)
    S = Atum.numberofstates(law)
    ix_ρ = 1
    ix_ρu⃗ = StaticArrays.SUnitRange(2, S - 1)
    ix_ρe = S
    return ix_ρ, ix_ρu⃗, ix_ρe
  end

  function unpackstate(law::EulerLaw, q)
    ix_ρ, ix_ρu⃗, ix_ρe = varsindices(law)
    @inbounds q[ix_ρ], q[ix_ρu⃗], q[ix_ρe]
  end

  function pressure(law::EulerLaw, ρ, ρu⃗, ρe)
    (γ(law) - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ)
  end
  function energy(law::EulerLaw, ρ, ρu⃗, p)
    p / (γ(law) - 1) + ρu⃗' * ρu⃗ / 2ρ
  end
  function soundspeed(law::EulerLaw, ρ, p)
    sqrt(γ(law) * p / ρ)
  end
  function soundspeed(law::EulerLaw, ρ, ρu⃗, ρe)
    soundspeed(law, ρ, pressure(law, ρ, ρu⃗, ρe))
  end

  function Atum.flux(law::EulerLaw, q, _)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    p = pressure(law, ρ, ρu⃗, ρe)

    fρ = ρu⃗
    fρu⃗ = ρu⃗ * u⃗' + p * I
    fρe = u⃗ * (ρe + p)

    hcat(fρ, fρu⃗, fρe)
  end

  function Atum.wavespeed(law::EulerLaw, n⃗, q, x⃗)
    ρ, ρu⃗, ρe = unpackstate(law, q)

    u⃗ = ρu⃗ / ρ
    abs(n⃗' * u⃗) + soundspeed(law, ρ, ρu⃗, ρe)
  end
end
