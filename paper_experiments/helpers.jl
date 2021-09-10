using Bennu

interleave(a, b) = collect(Iterators.flatten(zip(a, b)))
function reshape_global_cartesian(a, Nq, K)
  dim = length(Nq)
  @assert length(K) == dim
  a = reshape(a, (Nq..., K...))
  a = permutedims(a, interleave(1:dim, dim+1:2dim))
  reshape(a, Nq .* K)
end

function interpolate_equidistant(data, grid)
  cell = referencecell(grid)
  dim = ndims(cell)

  ξsrc = vec.(cell.points_1d)
  Nq = size(cell)
  K = size(grid)

  Nqi = 4 .* Nq
  dξi = 2 ./ Nqi
  ξdst = ntuple(d -> [-1 + (j - 1 / 2) * dξi[d] for j in 1:Nqi[d]], dim)
  I = kron(ntuple(d->spectralinterpolation(ξsrc[d], ξdst[d]), dim)...)

  points_i = I * points(grid)
  points_i = reshape_global_cartesian(points_i, Nqi, K)

  data_i = I * data
  data_i = reshape_global_cartesian(data_i, Nqi, K)

  points_i, data_i
end

function rcParams!(rcParams)
  fs = 22
  bigfs = 25
  rcParams["font.size"] = fs
  rcParams["text.usetex"] = true
  rcParams["font.family"] = "serif"
  rcParams["font.serif"] = "Computer Modern"
  rcParams["xtick.labelsize"] = fs
  rcParams["ytick.labelsize"] = fs

  tw = 1
  ts = 8
  rcParams["xtick.major.size"] = ts
  rcParams["xtick.major.width"] = tw
  rcParams["ytick.major.size"] = ts
  rcParams["ytick.major.width"] = tw

  mtw = 1
  mts = 4
  rcParams["xtick.minor.size"] = mts
  rcParams["xtick.minor.width"] = mtw
  rcParams["ytick.minor.size"] = mts
  rcParams["ytick.minor.width"] = mtw

  rcParams["legend.fontsize"] = fs
  rcParams["figure.titlesize"] = bigfs
  rcParams["axes.titlepad"] = 10
  rcParams["axes.labelpad"] = 10
  rcParams["axes.linewidth"] = 1
end
