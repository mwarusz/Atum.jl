export CentralFlux
export RusanovFlux
export RoeFlux

struct CentralFlux end
function (::CentralFlux)(law::AbstractBalanceLaw, n⃗, x⃗, q⁻, q⁺)
  (flux(law, q⁻, x⃗) + flux(law, q⁺, x⃗))' * n⃗ / 2
end

struct RusanovFlux end
function (::RusanovFlux)(law::AbstractBalanceLaw, n⃗, x⃗, q⁻, q⁺)
  fc = CentralFlux()(law, n⃗, x⃗, q⁻, q⁺)
  ws⁻ = wavespeed(law, n⃗, q⁻, x⃗)
  ws⁺ = wavespeed(law, n⃗, q⁺, x⃗)
  fc - max(ws⁻, ws⁺) * (q⁺ - q⁻) / 2
end

roe_avg(ρ⁻, ρ⁺, s⁻, s⁺) = (sqrt(ρ⁻) * s⁻ + sqrt(ρ⁺) * s⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))
struct RoeFlux end
