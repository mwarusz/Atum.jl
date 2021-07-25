module Advection
  export AdvectionLaw

  import ..Atum
  using StaticArrays: SVector

  @Base.kwdef struct AdvectionLaw{FT, D} <: Atum.AbstractBalanceLaw{FT, D, 1}
    u⃗::SVector{D, FT} = ones(SVector{D, FT})
  end

  Atum.flux(law::AdvectionLaw, q, aux) = law.u⃗ * q'
  Atum.wavespeed(law::AdvectionLaw, n⃗, q, aux) = abs(n⃗' * law.u⃗)
end
