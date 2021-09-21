include("risingbubble.jl")
include("../helpers.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function potential_temperature(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = grav(law) * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  p0 = 1e5
  T = p / (R_d * ρ)
  θ = T * (p0 / p) ^ (1 - 1 / γ(law))
end

function diagnostics(law, q, qref, x⃗)
  θ = potential_temperature(law, q, x⃗)
  θ_ref = potential_temperature(law, qref, x⃗)
  δθ = θ - θ_ref
  SVector(δθ)
end

function calculate_diagnostics(outputdir)
  diagnostic_data = Dict()

  for (root, dir, files) in walkdir(outputdir)
    jldfiles = filter(s->endswith(s, "jld2"), files)
    length(jldfiles) == 0 && continue
    @assert length(jldfiles) == 1
    data = load(joinpath(root, jldfiles[1]))

    law = data["law"]
    dg = data["dg"]
    grid = dg.grid
    cell = referencecell(grid)
    q = data["q"]
    qref = data["qref"]
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = diagnostics.(Ref(law), q, qref, points(grid))
    diag_points, diag = interpolate_equidistant(diag, grid)
    diagnostic_data[(N, KX)] = (diag_points, diag)
  end
  diagnostic_data
end

function contour_plot(root, diagnostic_data, N, KX)
  diag_points, diag = diagnostic_data[(N, KX)]
  x, z = components(diag_points)
  FT = eltype(x)
  δθ = first(components(diag))

  ioff()
  levels = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  fig = figure(figsize=(14, 12))
  ax = gca()
  xticks = range(0, 2000, length = 5)
  ax.set_title("Potential temperature perturbation [K]")
  ax.set_xlim([xticks[1], xticks[end]])
  ax.set_ylim([xticks[1], xticks[end]])
  ax.set_xticks(xticks)
  ax.set_yticks(xticks)
  ax.set_xlabel(L"x" * " [m]")
  ax.set_ylabel(L"z" * " [m]")
  norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
  ax.contour(x', z', δθ', levels=levels, colors=("k",))
  ax.set_aspect(1)
  cbar = colorbar(cset)
  tight_layout()
  savefig(joinpath(root, "rtb_tht_perturbation_$(N)_$(KX).pdf"))
end

let
  outputdir = joinpath("paper_output", "risingbubble")
  diagnostic_data = calculate_diagnostics(outputdir)

  plotdir = joinpath("paper_plots", "risingbubble")
  mkpath(plotdir)

  for (N, KX) in keys(diagnostic_data)
    contour_plot(plotdir, diagnostic_data, N, KX)
  end
end
