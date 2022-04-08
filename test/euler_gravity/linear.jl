using Atum
using Atum.EulerGravity

using StaticArrays: SVector
using WriteVTK
using Printf
using Test

import Atum: boundarystate
function boundarystate(law::LinearEulerGravityLaw, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρe⁻ = EulerGravity.unpackstate(law, q⁻)
  ρ⁺, ρe⁺ = ρ⁻, ρe⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρe⁺), aux⁻
end

import Atum.EulerGravity: referencestate
function referencestate(law::EulerGravityLaw, x⃗)
  FT = eltype(law)
  x, z = x⃗

  Φ = constants(law).grav * z

  cv_d = FT(719)
  cp_d = constants(law).γ * cv_d
  R_d = cp_d - cv_d

  θref = FT(300)
  p0 = FT(1e5)

  θ = θref
  π_exner = 1 - constants(law).grav / (cp_d * θ) * z
  ρ = p0 / (R_d * θ) * π_exner ^ (cv_d / R_d)

  ρu⃗ = SVector(FT(0), FT(0))

  T = θ * π_exner
  ρe = ρ * (cv_d * T + Φ)

  SVector(ρ, EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ))
end

function initialcondition(law, x⃗, aux)
  FT = eltype(law)
  ρ = EulerGravity.reference_ρ(law, aux)
  p = EulerGravity.reference_p(law, aux)
  ρu⃗ = SVector{2, FT}(0, 0)
  Φ = EulerGravity.geopotential(law, aux)
  ρe = EulerGravity.reference_ρe(law, p, ρ, Φ)
  SVector(ρ, ρu⃗..., ρe)
end

function run(A, FT, N, K; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = LinearEulerGravityLaw(EulerGravityLaw{FT, 2}())
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(10e3), length=K+1)
  vz = range(FT(0), stop=FT(10e3), length=K+1)
  grid = brickgrid(cell, (vx, vz); periodic = (true, false),
                   ordering = StackedOrdering{CartesianOrdering}())

  dg_nonlinear = DGSEM(; law=parent(law), grid, volume_form,
                       surface_numericalflux = RusanovFlux())
  dg = DGSEM(; law, grid, volume_form, surface_numericalflux = RusanovFlux(),
             auxstate=dg_nonlinear.auxstate,
             directions=(2,))

  cfl = FT(100)
  dz = min_node_distance(grid, dims=(2,))
  dt = cfl * dz / 330
  timeend = FT(5 * 24 * 60 * 60)
  
  #numberofsteps = ceil(Int, timeend / dt)
  #@show numberofsteps
 
  q = fieldarray(undef, law, grid)
  q .= initialcondition.(Ref(law), points(grid), dg.auxstate)
  qref = similar(q)
  qref .= q

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity", "linear")
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
      vtkfile["ρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["ρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["ρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["ρe"] = vec(Array(P * (ρe - ρe_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = ARK23(nothing, dg, fieldarray(q), dt; split_rhs = true)

  @info @sprintf """Starting
  N           = %d
  K           = %d
  volume_form = %s
  norm(q)     = %.16e
  """ N K volume_form weightednorm(dg, q)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output, adjust_final=false)
  outputvtk && vtk_save(pvd)


  errf = weightednorm(dg, q .- qref)
  @info @sprintf """Finished
  norm(q)      = %.16e
  norm(q - qe) = %.16e
  """ weightednorm(dg, q) errf
  
  errf
end

let
  A = Array
  FT = Float64
  N = 4

  nlevels = 2
  expected_error = Dict()

  expected_error[1] = 4.5574748976998530e-01
  expected_error[2] = 1.4283626820403727e-02

  @testset "linear model" begin
    errors = zeros(FT, nlevels)
    for l in 1:nlevels
      K = 5 * 2 ^ (l - 1)
      errf = run(A, FT, N, K)
      errors[l] = errf
      @test errors[l] ≈ expected_error[l] rtol=1e-3
    end

    if nlevels > 1
      rates = log2.(errors[1:(nlevels-1)] ./ errors[2:nlevels])
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
      @test rates[end] ≈ N + 1 atol = 0.1
    end
  end
end
