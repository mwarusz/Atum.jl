module Advection
  export AdvectionLaw

  import ..Atum
  using ..Atum: constants
  using StaticArrays: SVector

  struct AdvectionLaw{FT, D, C} <: Atum.AbstractBalanceLaw{FT, D, 1, C}
    function AdvectionLaw{FT, D}(u⃗ = ntuple(d->FT(1), D)) where {FT, D}
      new{FT, D, (;u⃗)}()
    end
  end

  Atum.flux(law::AdvectionLaw, q, aux) = SVector(constants(law).u⃗) * q'
  Atum.wavespeed(law::AdvectionLaw, n⃗, q, aux) = abs(n⃗' * SVector(constants(law).u⃗))
  Atum.entropy(law::AdvectionLaw, q, aux) = q' * q / 2
  Atum.entropyvariables(law::AdvectionLaw, q, aux) = q
end
