include("gravitywave.jl")
include("../helpers.jl")

using FileIO
using JLD2: @load
using PyPlot
using LaTeXStrings

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function compute_errors(dg, diag, diag_exact)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  err_w = weightednorm(dg, w .- w_exact) / sqrt(sum(dg.MJ))
  err_δT = weightednorm(dg, δT .- δT_exact) / sqrt(sum(dg.MJ))

  err_w, err_δT
end

function plot(root, diag_points, diag, diag_exact)
  x, z = components(diag_points)
  FT = eltype(x)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  # convert coordiantes to km
  x ./= 1e3
  z ./= 1e3

  ioff()
  xticks = range(0, 300, length = 7)
  fig, ax = subplots(2, 1, figsize=(14, 14))

  for a in ax
    a.set_xlim([xticks[1], xticks[end]])
    a.set_xticks(xticks)
    a.set_xlabel(L"x" * " [km]")
    a.set_ylabel(L"z" * " [km]")
    a.set_aspect(15)
  end

  ΔT = FT(0.0001)
  scaling = 1e2 * _ΔT
  ll = 0.0036 * scaling
  sl = 0.0006 * scaling
  levels = vcat(-ll:sl:-sl, sl:sl:ll)
  ax[1].set_title("T perturbation [K]")
  norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[1].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
  ax[1].contour(x', z', δT_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[1])

  ax[2].set_title("w [m/s]")
  #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[2].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
  ax[2].contour(x', z', w_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[2])

  tight_layout()
  savefig(joinpath(root, "gw_T_perturbation.pdf"))
  close(fig)
end

function calculate_diagnostics(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = grav(law) * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  ρ_ref, p_ref = referencestate(law, x⃗)
  T_ref = p_ref / (R_d * ρ_ref)

  w = ρw / ρ
  δT = T - T_ref

  SVector(w, δT)
end

let
  outputdir = joinpath("paper_output", "gravitywave")

  errors_w = Dict()
  errors_T = Dict()
  for (root, dir, files) in walkdir(outputdir)
    jldfiles = filter(s->endswith(s, "jld2"), files)
    length(jldfiles) == 0 && continue
    @assert length(jldfiles) == 1
    data = load(joinpath(root, jldfiles[1]))

    law = data["law"]
    dg = data["dg"]
    grid = dg.grid
    q = data["q"]
    qexact = data["qexact"]

    diag = calculate_diagnostics.(Ref(law), q, points(grid))
    diag_exact = calculate_diagnostics.(Ref(law), qexact, points(grid))

    dx = _L / size(grid)[1]
    errors_w[dx], errors_T[dx] = compute_errors(dg, diag, diag_exact)

    @show dx, errors_w[dx], errors_T[dx]

    diag_points, diag = interpolate_equidistant(diag, grid)
    _, diag_exact = interpolate_equidistant(diag_exact, grid)

    plot(root, diag_points, diag, diag_exact)
  end
end
