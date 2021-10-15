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
  Φ = constants(law).grav * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  p0 = 1e5
  T = p / (R_d * ρ)
  θ = T * (p0 / p) ^ (1 - 1 / constants(law).γ)
end

function diagnostics(law, q, qref, x⃗)
  θ = potential_temperature(law, q, x⃗)
  θ_ref = potential_temperature(law, qref, x⃗)
  δθ = θ - θ_ref
  SVector(δθ)
end

function calculate_diagnostics(outputfile)
  diagnostic_data = Dict()

  data = load(outputfile)

  law = data["law"]
  experiments = data["experiments"]
  for exp_key in keys(experiments)
    exp = experiments[exp_key]
    dg = exp.dg
    q = exp.q
    qref = exp.qref
    dη_timeseries = exp.dη_timeseries

    grid = dg.grid
    cell = referencecell(grid)
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = diagnostics.(Ref(law), q, qref, points(grid))
    diag_points, diag = interpolate_equidistant(diag, grid)
    diagnostic_data[(exp_key, N, KX)] = (;diag_points, diag, dη_timeseries)
  end

  diagnostic_data
end

function contour_plot(root, diagnostic_data, exp, N, KX)
  diag_points, diag, _ = diagnostic_data[(exp, N, KX)]
  x, z = components(diag_points)
  FT = eltype(x)
  δθ = first(components(diag))

  ioff()
  levels = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  fig = figure(figsize=(14, 12))
  ax = gca()
  xticks = range(-1000, 1000, length = 5)
  yticks = range(0, 2000, length = 5)
  ax.set_title("Potential temperature perturbation [K]")
  ax.set_xlim([xticks[1], xticks[end]])
  ax.set_ylim([yticks[1], yticks[end]])
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel(L"x" * " [m]")
  ax.set_ylabel(L"z" * " [m]")
  norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
  ax.contour(x', z', δθ', levels=levels, colors=("k",))
  ax.set_aspect(1)
  cbar = colorbar(cset)
  tight_layout()
  savefig(joinpath(root, "rtb_tht_perturbation_$(exp)_$(N)_$(KX).pdf"))
end

function entropy_conservation_plot(root, diagnostic_data, N, KX)
  dη_ts_ec = diagnostic_data[("lowres_ec", N, KX)].dη_timeseries
  dη_ts_mat = diagnostic_data[("lowres_matrix", N, KX)].dη_timeseries

  t_ec = first.(dη_ts_ec)
  dη_ec = last.(dη_ts_ec)

  t_mat = first.(dη_ts_mat)
  dη_mat = last.(dη_ts_mat)

  t_ec = t_ec[1:10:end]
  dη_ec = dη_ec[1:10:end]

  t_mat = t_mat[1:10:end]
  dη_mat = dη_mat[1:10:end]

  @pgf begin
    plot_ec = Plot({mark="o", color="red"}, Coordinates(t_ec, dη_ec))
    plot_matrix = Plot({mark="x", color="blue"}, Coordinates(t_mat, dη_mat))
    legend = Legend("Entropy conservative flux", "Matrix dissipation flux")
    axis = Axis({
                 ylabel=L"(\eta - \eta_0) / |\eta_0|",
                 xlabel="time [s]",
                 legend_pos="south west",
                },
                #L"\node[] at (320,-0.5e-8) {vanilla DGSEM};",
                #L"\node[] at (270,-0.6e-8) {breaks here};",
                plot_ec,
                plot_matrix,
                #plot300,
               legend)
    pgfsave(joinpath(root, "rtb_entropy.pdf"), axis)
  end
end

let
  outputfile = joinpath("paper_output", "risingbubble", "jld2", "risingbubble.jld2")
  diagnostic_data = calculate_diagnostics(outputfile)

  plotdir = joinpath("paper_plots", "risingbubble")
  mkpath(plotdir)

  for (exp, N, KX) in keys(diagnostic_data)
    contour_plot(plotdir, diagnostic_data, exp, N, KX)
  end
  entropy_conservation_plot(plotdir, diagnostic_data, 4, 5)
end
