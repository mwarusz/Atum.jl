using Test

@testset "advection" begin
  include("wave_1d.jl")
  include("wave_2d.jl")
  include("wave_3d.jl")
end
