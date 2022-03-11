export LinearEulerGravityLaw

struct LinearEulerGravityLaw{FT, D, S, C, P} <: Atum.AbstractBalanceLaw{FT, D, S, C}
  parent::P
  function LinearEulerGravityLaw(parent::EulerGravityLaw{FT, D, S, C}) where {FT, D, S, C}
    new{FT, D, S, C, typeof(parent)}(parent)
  end
end
Base.parent(law::LinearEulerGravity) = law.parent

varsindices(law::LinearEulerGravityLaw) = varsindices(parent(law))
unpackstate(law::LinearEulerGravityLaw, q) = unpackstate(parent(law), q)

referencestate(law::LinearEulerGravityLaw, x⃗) = referencestate(parent(law), x⃗)
reference_ρ(law::LinearEulerGravityLaw, aux) = reference_ρ(parent(law), aux)
reference_p(law::LinearEulerGravityLaw, aux) = reference_p(parent(law), aux)
function reference_ρe(law::LinearEulerGravityLaw, p, ρ, Φ)
  γ = constants(law).γ
  p / (γ - 1) + ρ + Φ
end

Atum.auxiliary(law::LinearEulerGravityLaw, x⃗) = Atum.auxiliary(parent(law), x⃗)

coordinates(law::LinearEulerGravityLaw, aux) = coordinates(parent(law), aux)
geopotential(law::LinearEulerGravityLaw, aux) = geopotential(parent(law), aux)

function pressure(law::LinearEulerGravityLaw, ρ, ρu⃗, ρe, Φ)
  γ = constants(law).γ
  (γ - 1) * (ρe - ρ * Φ)
end
function soundspeed(law::LinearEulerGravityLaw, aux)
  soundspeed(parent(law), reference_p(law, aux) reference_ρ(law, aux))
end

function Atum.flux(law::LinearEulerGravityLaw, q, aux)
  ρ, ρu⃗, ρe = unpackstate(law, q)

  Φ = geopotential(law, aux)

  u⃗ = ρu⃗ / ρ
  p = pressure(law, ρ, ρu⃗, ρe, Φ)

  ref_p = reference_p(law, aux)
  ref_ρ = reference_ρ(law, aux)
  ref_ρe = reference_ρe(law, ref_p, ref_ρ, Φ)

  fρ = ρu⃗
  fρu⃗ = p * I
  fρe = ((ref_ρe + ref_p) / ref_ρ) * ρu⃗

  hcat(fρ, fρu⃗, fρe)
end

Atum.wavespeed(law::LinearEulerGravityLaw, n⃗, q, aux) = soundspeed(law, aux)

function Atum.source!(law::LinearEulerGravityLaw, dq, q, aux, dim, directions)
  if dim ∈ directions
    ix_ρ, ix_ρu⃗, _ = varsindices(law)

    @inbounds ρ = q[ix_ρ]
    @inbounds dq[ix_ρu⃗[end]] -= ρ * constants(law).grav
  end
end
