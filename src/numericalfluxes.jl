export CentralFlux
export RusanovFlux
export EntropyConservativeFlux
export RoeFlux

avg(s⁻, s⁺) = (s⁻ + s⁺) / 2
function logavg(a, b)
    ζ = a / b
    f = (ζ - 1) / (ζ + 1)
    u = f^2
    ϵ = eps(eltype(u))

    if u < ϵ
        F = @evalpoly(u, one(u), one(u) / 3, one(u) / 5, one(u) / 7, one(u) / 9)
    else
        F = log(ζ) / 2f
    end

    (a + b) / 2F
end
roe_avg(ρ⁻, ρ⁺, s⁻, s⁺) = (sqrt(ρ⁻) * s⁻ + sqrt(ρ⁺) * s⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

abstract type AbstractNumericalFlux end
function twopointflux end
function surfaceflux(flux::AbstractNumericalFlux, law::AbstractBalanceLaw, n⃗, x⃗, q⁻, q⁺)
  twopointflux(flux, law, q⁻, x⃗, q⁺, x⃗)' * n⃗
end

struct CentralFlux <: AbstractNumericalFlux end
function twopointflux(::CentralFlux, law::AbstractBalanceLaw, q₁, x⃗₁, q₂, x⃗₂)
  (flux(law, q₁, x⃗₁) + flux(law, q₂, x⃗₂)) / 2
end

struct RusanovFlux <: AbstractNumericalFlux end
function surfaceflux(::RusanovFlux, law::AbstractBalanceLaw, n⃗, x⃗, q⁻, q⁺)
  fc = surfaceflux(CentralFlux(), law, n⃗, x⃗, q⁻, q⁺)
  ws⁻ = wavespeed(law, n⃗, q⁻, x⃗)
  ws⁺ = wavespeed(law, n⃗, q⁺, x⃗)
  fc - max(ws⁻, ws⁺) * (q⁺ - q⁻) / 2
end

struct EntropyConservativeFlux <: AbstractNumericalFlux end
struct RoeFlux <: AbstractNumericalFlux end
