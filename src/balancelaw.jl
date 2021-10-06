export numberofstates, constants

abstract type AbstractBalanceLaw{FT, D, S, C} end

Base.eltype(::AbstractBalanceLaw{FT}) where {FT} = FT
Base.ndims(::AbstractBalanceLaw{FT, D}) where {FT, D} = D
numberofstates(::AbstractBalanceLaw{FT, D, S}) where {FT, D, S} = S
constants(::AbstractBalanceLaw{FT, D, S, C}) where {FT, D, S, C} = C

auxiliary(law::AbstractBalanceLaw, x⃗) = SVector(nothing)

function flux end
function wavespeed end
boundarystate(::AbstractBalanceLaw, n⃗, q⁻, aux⁻, tag) = q⁻, aux⁻
source!(::AbstractBalanceLaw, dq, q, aux) = nothing
nonconservative_term!(::AbstractBalanceLaw, dq, q, aux) = nothing

function entropy end
function entropyvariables end
