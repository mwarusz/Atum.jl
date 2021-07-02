using Test

@testset "euler" begin
  include("wave_1d.jl")
  include("wave_2d.jl")
  include("wave_3d.jl")
  include("isentropicvortex.jl")
end
