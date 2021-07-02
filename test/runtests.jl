using Test

@testset "tests" begin
  include(joinpath("advection", "runtests.jl"))
  include(joinpath("euler", "runtests.jl"))
end
