module Advection
  export AdvectionLaw

  import ..Atum
  using ..Atum: constants
  using StaticArrays: SVector

  struct AdvectionLaw{FT, D, C} <: Atum.AbstractBalanceLaw{FT, D, 1, C}
    function AdvectionLaw{FT, D}(u⃗ = ones(SVector{D, FT})) where {FT, D}
      new{FT, D, (u⃗ = SVector{D, FT}(u⃗),)}()
    end
  end

  Atum.flux(law::AdvectionLaw, q, aux) = constants(law).u⃗ * q'
  Atum.wavespeed(law::AdvectionLaw, n⃗, q, aux) = abs(n⃗' * constants(law).u⃗)
end
