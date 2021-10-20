include("gravitywave.jl")
include("../helpers.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings

const _scaling = 1e2 * _ΔT

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function diagnostics(law, q, x⃗)
  ρ, ρu, ρw, ρe = q
  ρu⃗ = SVector(ρu, ρw)

  x, z = x⃗
  Φ = constants(law).grav * z

  R_d = gas_constant(law)
  p = EulerGravity.pressure(law, ρ, ρu⃗, ρe, Φ)
  T = p / (R_d * ρ)

  ρ_ref, p_ref = referencestate(law, x⃗)
  T_ref = p_ref / (R_d * ρ_ref)

  w = ρw / ρ
  δT = T - T_ref

  SVector(w, δT)
end

function compute_errors(dg, diag, diag_exact)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  err_w = weightednorm(dg, w .- w_exact) / sqrt(sum(dg.MJ))
  err_T = weightednorm(dg, δT .- δT_exact) / sqrt(sum(dg.MJ))

  err_w, err_T
end

function calculate_convergence(data)
  convergence_data = Dict()
  law = data["law"]
  experiments = data["experiments"]
  conv = experiments["conv"]

  for N in keys(conv)
    for exp in conv[N]
      dg = exp.dg
      q = exp.q
      qexact = exp.qexact

      grid = dg.grid
      cell = referencecell(grid)
      N = size(cell)[1] - 1
      KX = size(grid)[1]

      diag = diagnostics.(Ref(law), q, points(grid))
      diag_exact = diagnostics.(Ref(law), qexact, points(grid))

      dx = _L / KX
      err_w, err_T = compute_errors(dg, diag, diag_exact)
      @show dx, err_w, err_T

      if N in keys(convergence_data)
        push!(convergence_data[N].w_errors, err_w)
        push!(convergence_data[N].T_errors, err_T)
        push!(convergence_data[N].dxs, dx)
      else
        convergence_data[N] = (dxs = Float64[], w_errors=Float64[], T_errors=Float64[])
        push!(convergence_data[N].w_errors, err_w)
        push!(convergence_data[N].T_errors, err_T)
        push!(convergence_data[N].dxs, dx)
      end
    end
  end
  convergence_data
end

function calculate_diagnostics(data)
  diagnostic_data = Dict()

  law = data["law"]
  experiments = data["experiments"]

  for exp_key in keys(experiments)
    exp_key == "conv" && continue
    exp = experiments[exp_key]
    dg = exp.dg
    q = exp.q
    timeend = exp.timeend

    grid = dg.grid
    cell = referencecell(grid)
    N = size(cell)[1] - 1
    KX = size(grid)[1]

    diag = diagnostics.(Ref(law), q, points(grid))
    diag_points, diag = interpolate_equidistant(diag, grid)
    qexact = gravitywave.(Ref(law), diag_points, timeend)
    diag_exact = diagnostics.(Ref(law), qexact, diag_points)

    x, z = components(diag_points)
    # convert coordiantes to km
    x ./= 1e3
    z ./= 1e3

    diagnostic_data[(N, KX)] = (; diag_points, diag, diag_exact)
  end

  diagnostic_data
end

function contour_plot(root, diagnostic_data, N, KX)
  diag_points, diag, diag_exact = diagnostic_data[(N, KX)]

  x, z = components(diag_points)
  FT = eltype(x)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  ioff()
  xticks = range(0, 300, length = 7)
  yticks = range(0, 10, length = 5)
  fig, ax = subplots(2, 1, figsize=(14, 14))

  for a in ax
    a.set_xlim([0, 300])
    a.set_xticks(xticks)
    a.set_yticks(yticks)
    a.set_ylim([0, 10])
    a.set_xlabel(L"x" * " [km]")
    a.set_ylabel(L"z" * " [km]")
    a.set_aspect(15)
  end

  ll = 0.0036 * _scaling
  sl = 0.0006 * _scaling
  levels = vcat(-ll:sl:-sl, sl:sl:ll)
  ax[1].set_title("T perturbation [K]")
  norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[1].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
  ax[1].contour(x', z', δT_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[1], ticks=levels)

  ax[2].set_title("w [m/s]")
  #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
  cset = ax[2].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
  ax[2].contour(x', z', w_exact', levels=levels, colors=("k",))
  cbar = colorbar(cset, ax = ax[2], ticks=levels)

  tight_layout()
  savefig(joinpath(root, "gw_contour_$(N)_$(KX).pdf"))
  close(fig)
end

function line_plot(root, diagnostic_data, N, KX)
  diag_points, diag, diag_exact = diagnostic_data[(N, KX)]

  x, z = components(diag_points)
  FT = eltype(x)
  w, δT = components(diag)
  w_exact, δT_exact = components(diag_exact)

  k = findfirst(z[1, :] .>= 5)

  @show z[1, k]
  x_k = x[:, k]
  w_k = w[:, k]
  w_exact_k = w_exact[:, k]
  @pgf begin
    fig = @pgf GroupPlot({group_style= {group_size="1 by 1", vertical_sep="1.5cm"},
                         xmin=0,
                         xmax= 300})
    ytick = [_scaling * (-3 + i) * 1e-3 for i in 0:7]
    xtick = [50 * i for i in 0:6]
    p1 = Plot({dashed}, Coordinates(x_k, w_k))
    p2 = Plot({}, Coordinates(x_k, w_exact_k))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                ytick = ytick,
                xtick = xtick,
                width="10cm",
                height="5cm"},
               p1, p2)
    pgfsave(joinpath(root, "gw_line_$(N)_$(KX).pdf"), fig)
  end
end

function convergence_plot(outputdir, convergence_data)
  dxs = convergence_data[2].dxs
  @pgf begin
    plotsetup = {
              xlabel = "Δx [km]",
              grid = "major",
              xmode = "log",
              ymode = "log",
              xticklabel="{\\pgfmathparse{exp(\\tick)/1000}\\pgfmathprintnumber[fixed,precision=3]{\\pgfmathresult}}",
              #xmax = 1,
              xtick = dxs,
              #ymin = 10. ^ -10 / 5,
              #ymax = 5,
              #ytick = 10. .^ -(0:2:10),
              legend_pos="south east",
              group_style= {group_size="2 by 2",
                            vertical_sep="2cm",
                            horizontal_sep="2cm"},
            }

    Ns = sort(collect(keys(convergence_data)))
    fig = GroupPlot(plotsetup)
    for s in ('T', 'w')
        unit = s == 'T' ? "K" : "m/s"
        ylabel = L"L_{2}" * " error of $s [$unit]"
        labels = []
        plots = []
        title = "Convergence of $s"
        for N in Ns
          @show N
          dxs = convergence_data[N].dxs
          if N == 2
            Tcoeff = 2e-4
            wcoeff = 3e-4
          elseif N == 3
            Tcoeff = 2e-5
            wcoeff = 3e-5
          elseif N == 4
            Tcoeff = 5e-6
            wcoeff = 8e-6
          end
          ordl = (dxs ./ dxs[1]) .^ (N + 1)
          if s === 'T'
            errs = convergence_data[N].T_errors
            ordl *= Tcoeff
          else
            errs = convergence_data[N].w_errors
            ordl *= wcoeff
          end
          coords = Coordinates(dxs, errs)
          ordc = Coordinates(dxs, ordl)
          plot = PlotInc({}, coords)
          plotc = Plot({dashed}, ordc)
          push!(plots, plot, plotc)
          #push!(labels, "N$N " * @sprintf("(%.2f)", rates[end]))
          push!(labels, "N$N")
          push!(labels, "order $(N+1)")
        end
        legend = Legend(labels)
        push!(fig, {title=title, ylabel=ylabel}, plots..., legend)
      end
      savepath = joinpath(outputdir,
                          "gw_convergence.pdf")
      pgfsave(savepath, fig)
  end
end


function compare_contour_plot(root, diagnostics, N, (KX1, KX2))
  ioff()

  fig, ax = subplots(2, 2, figsize=(28, 10), sharex="col", sharey="row")

  ll = 0.0036 * _scaling
  sl = 0.0006 * _scaling
  levels = vcat(-ll:sl:-sl, sl:sl:ll)
  xticks = range(0, 300, length = 7)
  yticks = range(0, 10, length = 5)

  m = 1
  cset = nothing
  for KX in (KX1, KX2)
    diag_points, diag, diag_exact = diagnostics[(N, KX)]
    x, z = components(diag_points)
    w, δT = components(diag)
    w_exact, δT_exact = components(diag_exact)


    dx = _L / KX

    ax[1, m].set_title(L"\Delta x" * " = $(dx) m\n\n w [m/s]")
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[1, m].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
    ax[1, m].contour(x', z', w_exact', levels=levels, colors=("k",))

    ax[2, m].set_title("T perturbation [K]")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[2, m].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    ax[2, m].contour(x', z', δT_exact', levels=levels, colors=("k",))


    m += 1
  end

  for a in vec(ax)
      a.set_xlim([0, 300])
      a.set_xticks(xticks)
      a.set_yticks(yticks)
      a.set_ylim([0, 10])
      a.set_aspect(10)
  end

  ax[2, 1].set_xlabel(L"x" * " [km]")
  ax[2, 2].set_xlabel(L"x" * " [km]")
  ax[1, 1].set_ylabel(L"z" * " [km]")
  ax[2, 1].set_ylabel(L"z" * " [km]")


  tight_layout()
  cbar = colorbar(cset, ax = vec(ax), shrink=1.0,
                  ticks=levels)
  savefig(joinpath(root, "gw_compare_contour_$(N)_$(KX1)_vs_$(KX2).pdf"))
  close(fig)
end

function compare_line_plot(root, diagnostics, N, (KX1, KX2))
  fig = @pgf GroupPlot({group_style= {group_size="2 by 1", horizontal_sep="1.5cm"},
                        xmin=0,
                        xmax= 300})
  for KX in (KX1, KX2)
    diag_points, diag, diag_exact = diagnostics[(N, KX)]
    x, z = components(diag_points)
    w, δT = components(diag)
    w_exact, δT_exact = components(diag_exact)

    k = findfirst(z[1, :] .>= 5)
    @show z[1, k]
    x_k = x[:, k]
    w_k = w[:, k]
    w_exact_k = w_exact[:, k]

    ytick = [_scaling * (-3 + i) * 1e-3 for i in 0:7]
    xtick = [50 * i for i in 0:6]

    dx = _L / KX

    @pgf begin
      p1 = Plot({dashed}, Coordinates(x_k, w_k))
      p2 = Plot({}, Coordinates(x_k, w_exact_k))
      push!(fig, {xlabel="x [km]",
                  ylabel="w [m/s]",
                  ytick = ytick,
                  xtick = xtick,
                  title = L"\Delta x" * " = $dx m",
                  width="10cm",
                  height="5cm"},
                 p1, p2)
    end
  end
  pgfsave(joinpath(root, "gw_compare_line_$(N)_$(KX1)_vs_$(KX2).pdf"), fig)
end



let
  outputfile = joinpath("paper_output", "gravitywave", "jld2", "gravitywave.jld2")
  data = load(outputfile)
  diagnostic_data = calculate_diagnostics(data)
  convergence_data = calculate_convergence(data)

  plotdir = joinpath("paper_plots", "gravitywave")
  mkpath(plotdir)

  for (N, KX) in keys(diagnostic_data)
    contour_plot(plotdir, diagnostic_data, N, KX)
    line_plot(plotdir, diagnostic_data, N, KX)
  end

  compare_contour_plot(plotdir, diagnostic_data, 3, (50, 100))
  compare_line_plot(plotdir, diagnostic_data, 3, (50, 100))
  convergence_plot(plotdir, convergence_data)
end
