using Atum
using Atum.EulerGravity
using CUDA
using Adapt
using StaticArrays: SVector, SMatrix
using KernelAbstractions
using Printf

mysum(a) = foldl(+, a)

@kernel function set(dq, q, aux, D, MJ, MJI, g, ::Val{Nq}) where {Nq}
  e = @index(Group, Linear)
  i, j, k = @index(Local, NTuple)

  ijk = i + Nq * (j - 1 + Nq * (k - 1))

  if e == 1 && i <= Nq && j <= Nq
    D[i, j] = i - 5 * j
  end

  ρ = 1 + (i + j + k)
  ρu = -1 + (i - j + k)
  ρv = 0 + (i + j - k)
  ρw = 1 + (i - j + k)
  ρe = 10000 - (i + j + k)

  _MJ = 2 + (i * j * k)
  #_MJ = 1

  g11 = 1 + (-2 * i + j + k)
  g21 = 1 + (i - 2 * j + k)
  g31 = 1 + (i + j - 2 * k)

  g12 = 2 + (i + 3 * j + k)
  g22 = 2 + (i + j + 3 * k)
  g32 = 2 + (3 * i + j + k)

  g13 = 3 + (i + j - k)
  g23 = 3 + (i - j + k)
  g33 = 3 + (-i + j + k)

  aux[ijk, e] = SVector(0, 0, k)
  dq[ijk, e] = SVector(0, 0, 0, 0, 0)
  q[ijk, e] = SVector(ρ, ρu, ρv, ρw, ρe)
  MJ[ijk, e] = _MJ
  MJI[ijk, e] = 1 / _MJ
  g[ijk, e] = SMatrix{3, 3}(g11, g21, g31,
                            g12, g22, g32,
                            g13, g23, g33)
end

function ic(law, x)
  FT = eltype(law)
  ρ = FT(1)
  ρu = 0
  ρv = 0
  ρw = 0
  ρe = FT(10000)
  SVector(ρ, ρu, ρv, ρw, ρe)
end

function run_naive(dg, dq, q)
  increment = false
  cell = referencecell(dg)
  device = Atum.getdevice(dg)
  form = dg.volume_form
  Nq = size(cell)[1]
  dim = ndims(cell)
  Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

  workgroup = ntuple(i -> i <= min(2, dim) ? Nq : 1, 3)
  ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)

  event = Event(device)

  comp_stream = Atum.esvolumeterm!(device, workgroup)(
    dg.law,
    dq,
    q,
    derivatives_1d(cell)[1],
    form.volume_numericalflux,
    metrics(dg.grid),
    dg.MJ,
    dg.MJI,
    dg.auxstate,
    Val(dim),
    Val(Nq),
    Val(numberofstates(dg.law)),
    Val(Naux),
    Val(increment);
    ndrange,
    dependencies = event
  )
  wait(comp_stream)
end

function run_perdim(dg, dq, q)
  increment = false
  cell = referencecell(dg)
  device = Atum.getdevice(dg)
  form = dg.volume_form
  Nq = size(cell)[1]
  dim = ndims(cell)
  Naux = eltype(eltype(dg.auxstate)) === Nothing ? 0 : length(eltype(dg.auxstate))

  workgroup = ntuple(i -> i <= dim ? Nq : 1, 3)
  ndrange = (length(dg.grid) * workgroup[1], Base.tail(workgroup)...)

  comp_stream = Event(device)
  for dir in 1:3
    comp_stream = Atum.esvolumeterm_dir!(device, workgroup)(
      dg.law,
      dq,
      q,
      derivatives_1d(cell)[1],
      form.volume_numericalflux,
      metrics(dg.grid),
      dg.MJ,
      dg.MJI,
      dg.auxstate,
      dir == 1, # add_source
      Val(dir),
      Val(dim),
      Val(Nq),
      Val(numberofstates(dg.law)),
      Val(Naux),
      Val(dir == 1 ? increment : true);
      ndrange,
      dependencies = comp_stream
    )
  end
  wait(comp_stream)
end

function check()
  A = CuArray
  FT = Float64
  law = EulerGravityLaw{FT, 3}(;grav=1)
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())

  N = 4
  K = 10

  Nq = N + 1
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)

  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d, v1d); periodic = (true, true, true))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux)


  q = ic.(Ref(law), points(grid))
  dq = similar(q)
  g, J = components(metrics(grid))

  D = derivatives_1d(cell)[1]

  device = Atum.getdevice(dg)
  event = set(device, (Nq, Nq, Nq))(dq, q, dg.auxstate, D, dg.MJ, dg.MJI,
                                    g, Val(Nq); ndrange = (Nq * length(grid), Nq, Nq))
  wait(event)

  expected_state_min = [ 4.0000000000000000e+00,
                        -4.0000000000000000e+00,
                        -3.0000000000000000e+00,
                        -2.0000000000000000e+00,
                         9.9850000000000000e+03]
  expected_state_max = [ 1.6000000000000000e+01,
                         8.0000000000000000e+00,
                         9.0000000000000000e+00,
                         1.0000000000000000e+01,
                         9.9970000000000000e+03]
  expected_state_sum = [ 1.2500000000000000e+06,
                         2.5000000000000000e+05,
                         3.7500000000000000e+05,
                         5.0000000000000000e+05,
                         1.2488750000000000e+09]

  hq = adapt(Array, q)
  ρ, ρu, ρv, ρw, ρe = components(hq)
  states = components(hq)
  #println("State")
  for s in 1:length(states)
    min_s, max_s = extrema(states[s])
    sum_s =  mysum(states[s])
    #@printf("%d %.16e %.16e %.16e\n", s, min_s, max_s, sum_s)
    @assert min_s == expected_state_min[s]
    @assert max_s == expected_state_max[s]
    @assert sum_s == expected_state_sum[s]
  end

  run_perdim(dg, dq, q)
  hdq = adapt(Array, dq)
  states = components(hdq)

  expected_tend_min = [ -8.0117671762591508e+03,
                        -2.3679217315230197e+06,
                        -3.6301053645583731e+06,
                        -1.0381257284802594e+06,
                        -7.7002708445811123e+06]
  expected_tend_max = [ 1.0724989336573461e+04,
                        2.2249376427549031e+06,
                        5.4765574022255857e+06,
                        2.7774765115051647e+06,
                        1.5120347669426484e+07]

  expected_tend_sum = [ 2.9012775536328048e+08,
                        1.0343814475884972e+11,
                        2.0882071181425778e+11,
                        7.4236989961421112e+10,
                        3.8842405050425000e+11]

  #println("Tendency")
  for s in 1:length(states)
    min_s, max_s = extrema(states[s])
    sum_s =  mysum(states[s])
    #@printf("%d %.16e %.16e %.16e\n", s, min_s, max_s, sum_s)
    @assert min_s ≈ expected_tend_min[s]
    @assert max_s ≈ expected_tend_max[s]
    @assert sum_s ≈ expected_tend_sum[s]
  end
end

function benchmark(::Type{FT}, ::Val{N}, K) where {N, FT}
  A = CuArray
  law = EulerGravityLaw{FT, 3}(;grav=1)
  volume_form = FluxDifferencingForm(EntropyConservativeFlux())

  Nq = N + 1
  cell = LobattoCell{FT, A}(Nq, Nq, Nq)

  v1d = range(FT(-1), stop=FT(1), length=K+1)
  grid = brickgrid(cell, (v1d, v1d, v1d); periodic = (true, true, true))

  dg = DGSEM(; law, grid, volume_form,
               surface_numericalflux = RusanovFlux)


  q = ic.(Ref(law), points(grid))
  dq = similar(q)
  g, J = components(metrics(grid))

  D = derivatives_1d(cell)[1]

  device = Atum.getdevice(dg)
  event = set(device, (Nq, Nq, Nq))(dq, q, dg.auxstate, D, dg.MJ, dg.MJI,
                                    g, Val(Nq); ndrange = (Nq * length(grid), Nq, Nq))
  wait(event)
  @CUDA.profile for m in 1:2
    run_perdim(dg, dq, q)
  end
end

check()
benchmark(Float64, Val(4), 40)
