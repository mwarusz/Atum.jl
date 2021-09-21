include("risingbubble.jl")

using WriteVTK
using JLD2
using CUDA
using Adapt

function run(A, FT, N, KX, KY;
             volume_form=WeakForm(),
             surface_flux=RoeFlux(),
             outputjld=true,
             outputvtk=false)
  Nq = N + 1

  pde_level_balance = volume_form isa WeakForm
  law = EulerGravityLaw{FT, 2}(;pde_level_balance)
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(_L), length=KX+1)
  vz = range(FT(0), stop=FT(_H), length=KY+1)
  grid = brickgrid(cell, (vx, vz); periodic = (true, false))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = surface_flux)

  cfl = FT(1 // 10)
  dt = cfl * min_node_distance(grid) / 330
  timeend = FT(1000)
 
  q = risingbubble.(Ref(law), points(grid))
  qref = risingbubble.(Ref(law), points(grid), false)

  if outputvtk
    vtkdir = joinpath("paper_output", "risingbubble")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρe = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρe_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["δρe"] = vec(Array(P * (ρe - ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = LSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output)
  outputvtk && vtk_save(pvd)

  if outputjld
    jlddir = joinpath("paper_output",
                      "risingbubble",
                      "$N",
                      "$(KX)x$(KY)")

    mkpath(jlddir)
    dg = adapt(Array, dg)
    q = adapt(Array, q)
    qref = adapt(Array, qref)

    @save(joinpath(jlddir, "finalstep.jld2"),
          law,
          dg,
          q,
          qref)
  end
end

let
  A = CuArray
  FT = Float64
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())
  surface_flux = MatrixFlux()

  N = 4
  KX = 10
  KY = 10
  run(A, FT, N, KX, KY; volume_form, surface_flux)
end
