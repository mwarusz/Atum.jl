using Test
using SafeTestsets

@testset "euler" begin
  @safetestset "linear" begin include("linear.jl") end
end
