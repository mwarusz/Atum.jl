export numberofstates

abstract type AbstractBalanceLaw{FT, D, S} end

Base.eltype(::AbstractBalanceLaw{FT}) where {FT} = FT
Base.ndims(::AbstractBalanceLaw{FT, D}) where {FT, D} = D
numberofstates(::AbstractBalanceLaw{FT, D, S}) where {FT, D, S} = S

function flux end
function wavespeed end
boundarystate(::AbstractBalanceLaw, n⃗, x⃗, q⁻, tag) = q⁻
source!(::AbstractBalanceLaw, dq, q, x⃗) = nothing
nonconservative_term!(::AbstractBalanceLaw, dq, q, x⃗) = nothing
