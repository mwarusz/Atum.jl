using Atum
using Atum.EulerThetaGravity

#using CUDA
using LinearAlgebra: norm
using StaticArrays: SVector
using WriteVTK

import Atum: boundarystate
function boundarystate(law::EulerThetaGravityLaw, n⃗, q⁻, aux⁻, _)
  ρ⁻, ρu⃗⁻, ρθ⁻ = EulerThetaGravity.unpackstate(law, q⁻)
  ρ⁺, ρθ⁺ = ρ⁻, ρθ⁻
  ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
  SVector(ρ⁺, ρu⃗⁺..., ρθ⁺), aux⁻
end

import Atum.EulerThetaGravity: referencestate
function referencestate(law::EulerThetaGravityLaw, x⃗)
  FT = eltype(law)
  x, z = x⃗

  cv_d = FT(719)
  cp_d = γ(law) * cv_d
  R_d = cp_d - cv_d

  p_s = FT(1e5)
  T_ref = FT(250)

  δ = grav(law) / (R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)
  ρ_ref = ρ_s * exp(-δ * z)

  p_ref = ρ_ref * R_d * T_ref

  SVector(ρ_ref, p_ref)
end

function gravitywave(law, x⃗, t, add_perturbation=true)
  FT = eltype(law)
  x, z = x⃗

  Φ = grav(law) * z

  cv_d = FT(719)
  cp_d = γ(law) * cv_d
  R_d = cp_d - cv_d

  ΔT = FT(0.0001)
  H = FT(10e3)
  f = FT(0)
  L = FT(300e3)
  d = FT(5e3)
  x_c = FT(100e3)
  u_0 = FT(20)
  p_s = FT(1e5)
  T_ref = FT(250)

  g = grav(law)
  δ = g / (R_d * T_ref)
  c_s = sqrt(cp_d / cv_d * R_d * T_ref)
  ρ_s = p_s / (T_ref * R_d)

  if t == 0
    δT_b = ΔT * exp(-(x - x_c) ^ 2 / d ^ 2) * sin(π * z / H)
    δT = exp(δ * z / 2) * δT_b
    δρ_b = -ρ_s * δT_b / T_ref
    δρ = exp(-δ * z / 2) * δρ_b
    δu, δv, δw = 0, 0, 0
  else
    xp = x - u_0 * t

    δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5, Complex{FT}})
    for m in (-1, 1)
      for n in -100:100
        k_x = 2π * n / L
        k_z = π * m / H

        p_1 = c_s ^ 2 * (k_x ^ 2 + k_z ^ 2 + δ ^ 2 / 4) + f ^ 2
        q_1 = g * k_x ^ 2 * (c_s ^ 2 * δ - g) + c_s ^ 2 * f ^ 2 * (k_z ^ 2 + δ ^ 2 / 4)

        α = sqrt(p_1 / 2 - sqrt(p_1 ^ 2 / 4 - q_1))
        β = sqrt(p_1 / 2 + sqrt(p_1 ^ 2 / 4 - q_1))

        fac1 = 1 / (β ^ 2 - α ^ 2)
        L_m1 = (-cos(α * t) / α ^ 2 + cos(β * t) / β ^ 2) * fac1 + 1 / (α ^ 2 * β ^ 2)
        L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
        L_1 = (cos(α * t) - cos(β * t)) * fac1
        L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
        L_3 = (-α ^ 2 * cos(α * t) + β ^ 2 * cos(β * t)) * fac1

        if α == 0
          L_m1 = (β ^ 2 * t ^ 2 - 1 + cos(β * t)) / β ^ 4
          L_0 = (β * t - sin(β * t)) / β ^ 3
        end

        δρ̃_b0 = -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                exp(-d ^ 2 * k_x ^ 2 / 4) * exp(-im * k_x * x_c) * k_z * H / 2im

        δρ̃_b = (L_3 + (p_1 + g * (im * k_z - δ / 2)) * L_1 +
              (c_s ^ 2 * (k_z ^ 2 + δ ^ 2 / 4) + g * (im * k_z - δ / 2)) * f ^ 2 * L_m1) * δρ̃_b0

        δp̃_b = -(g - c_s ^ 2 * (im * k_z + δ / 2)) * (L_1 + f ^ 2 * L_m1) * g * δρ̃_b0

        δũ_b = im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

        δṽ_b = -f * im * k_x * (g - c_s ^ 2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 / ρ_s

        δw̃_b = -(L_2 + (f ^ 2 + c_s ^ 2 * k_x ^ 2) * L_0) * g * δρ̃_b0 / ρ_s

        expfac = exp(im * (k_x * xp + k_z * z))

        δρ_b += δρ̃_b * expfac
        δp_b += δp̃_b * expfac

        δu_b += δũ_b * expfac
        δv_b += δṽ_b * expfac
        δw_b += δw̃_b * expfac
      end
    end

    δρ = exp(-δ * z / 2) * real(δρ_b)
    δp = exp(-δ * z / 2) * real(δp_b)

    δu = exp(δ * z / 2) * real(δu_b)
    δv = exp(δ * z / 2) * real(δv_b)
    δw = exp(δ * z / 2) * real(δw_b)

    δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
    δT = exp(δ * z / 2) * real(δT_b)
  end

  ρ_ref = ρ_s * exp(-δ * z)

  ρ = ρ_ref
  T = T_ref
  u = u_0
  w = FT(0)

  if add_perturbation
    ρ += δρ
    T += δT
    u += δu
    w += δw
  end

  p = R_d * ρ * T
  ρθ = EulerThetaGravity.theta(law, p)

  return SVector(ρ, ρ * u, ρ * w, ρθ)
end

function run(A, FT, N, KX, KY; volume_form=WeakForm(), outputvtk=true)
  Nq = N + 1

  law = EulerThetaGravityLaw{FT, 2}(pde_level_balance=true)
  
  cell = LobattoCell{FT, A}(Nq, Nq)
  vx = range(FT(0), stop=FT(300e3), length=KX+1)
  vz = range(FT(0), stop=FT(10e3), length=KY+1)
  grid = brickgrid(cell, (vx, vz); periodic = (true, false))

  dg = DGSEM(; law, cell, grid, volume_form,
               surface_numericalflux = RusanovFlux())

  cfl = FT(1 // 3)
  dt = cfl * step(vz) / N / 330
  timeend = @isdefined(_testing) ? 10dt : FT(30 * 60)
 
  q = gravitywave.(Ref(law), points(grid), FT(0))
  qref = gravitywave.(Ref(law), points(grid), FT(0), false)

  if outputvtk
    vtkdir = joinpath("output", "euler_gravity_2c", "gravitywave")
    mkpath(vtkdir)
    pvd = paraview_collection(joinpath(vtkdir, "timesteps"))
  end

  do_output = function(step, time, q)
    if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0 
      filename = "step$(lpad(step, 6, '0'))"
      vtkfile = vtk_grid(joinpath(vtkdir, filename), grid)
      P = Bennu.toequallyspaced(cell)
      ρ, ρu, ρv, ρθ = components(q)
      ρ_ref, ρu_ref, ρv_ref, ρθ_ref = components(qref)
      vtkfile["δρ"] = vec(Array(P * (ρ - ρ_ref)))
      vtkfile["δρu"] = vec(Array(P * (ρu - ρu_ref)))
      vtkfile["δρv"] = vec(Array(P * (ρv - ρv_ref)))
      vtkfile["δρθ"] = vec(Array(P * (ρθ - ρθ_ref)))
      vtk_save(vtkfile)
      pvd[time] = vtkfile
    end
  end

  odesolver = LSRK54(dg, q, dt)

  outputvtk && do_output(0, FT(0), q)
  solve!(q, timeend, odesolver; after_step=do_output)
  outputvtk && vtk_save(pvd)

  qexact = gravitywave.(Ref(law), points(grid), timeend)
  errf = map(components(q), components(qexact)) do f, fexact
    sqrt(sum(dg.MJ .* (f .- fexact) .^ 2))
  end

  ρ, ρu, ρw, ρθ = components(q)
  p = EulerThetaGravity.pressure.(Ref(law), ρθ)
  T =  p ./ (EulerThetaGravity.R_d(law) .* ρ)
  w = @. ρw / ρ

  ρ_exact, ρu_exact, ρw_exact, ρθ_exact = components(qexact)
  p_exact = EulerThetaGravity.pressure.(Ref(law), ρθ_exact)
  T_exact = p_exact ./ (EulerThetaGravity.R_d(law) .* ρ_exact)
  w_exact = @. ρw_exact / ρ_exact

  T_err = sqrt(sum(dg.MJ .* (T .- T_exact) .^ 2))
  w_err = sqrt(sum(dg.MJ .* (w .- w_exact) .^ 2))

  T_err, w_err
end

let
  A = Array
  FT = Float64
  N = 4

  ndof_x = 60
  ndof_y = 15
  KX_base = round(Int, ndof_x / N)
  KY_base = round(Int, ndof_y / N)

  nlevels = @isdefined(_testing) ? 1 : 2
  T_errors = zeros(FT, nlevels)
  w_errors = zeros(FT, nlevels)
  for l in 1:nlevels
    KX = KX_base * 2 ^ (l - 1)
    KY = KY_base * 2 ^ (l - 1)
    T_errors[l], w_errors[l] = run(A, FT, N, KX, KY)
  end
  if nlevels > 1
    T_rates = log2.(T_errors[1:(nlevels-1)] ./ T_errors[2:nlevels])
    w_rates = log2.(w_errors[1:(nlevels-1)] ./ w_errors[2:nlevels])
    @show T_errors
    @show T_rates
    @show w_errors
    @show w_rates
  end
end
