module Atum

using Reexport
using StaticArrays
@reexport using Bennu

using Adapt
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using CUDAKernels
using UnPack

include("balancelaw.jl")
include("numericalfluxes.jl")
include("dgsem.jl")
include("measures.jl")
include("odesolvers.jl")
include("bandedsystem.jl")

include("balancelaws/advection.jl")
include("balancelaws/shallow_water.jl")
include("balancelaws/euler.jl")
include("balancelaws/euler_gravity.jl")

end # module
