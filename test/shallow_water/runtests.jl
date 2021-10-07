using Test
using SafeTestsets

@testset "shallow_water" begin
  @safetestset "entropy_conservation_1d" begin include("entropy_conservation_1d.jl") end
end
