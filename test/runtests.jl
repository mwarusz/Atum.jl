using Test

@testset "tests" begin
  include(joinpath("advection", "runtests.jl"))
  include(joinpath("shallow_water", "runtests.jl"))
  include(joinpath("euler", "runtests.jl"))
end

@testset "experiments" begin
  experiments_dir = joinpath(@__DIR__, "..", "experiment")
  for (root, dir, files) in walkdir(experiments_dir)
    jlfiles = filter(s -> endswith(s, ".jl"), files)
    for jlfile in jlfiles
      path = joinpath(root, jlfile)
      modname = Symbol(jlfile)
      @eval module $modname
        using Test
        const _testing = true
        @testset $jlfile begin
          include($path)
        end
      end
    end
  end
end
