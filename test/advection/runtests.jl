using Test
using SafeTestsets

@testset "advection" begin
  @safetestset "wave_1d" begin include("wave_1d.jl") end
  @safetestset "wave_2d" begin include("wave_2d.jl") end
  @safetestset "wave_3d" begin include("wave_3d.jl") end
end
