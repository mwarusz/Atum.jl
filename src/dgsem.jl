export DGSEM

struct DGSEM{L, C, G, A1, A2, A3, NF}
  law::L
  cell::C
  grid::G
  MJ::A1
  MJI::A2
  faceMJ::A3
  numericalflux::NF

  function DGSEM(; law, cell, grid, numericalflux)
    M = mass(cell)
    _, J = components(metrics(grid))
    MJ = M * J
    MJI = 1 ./ MJ
    
    faceM = facemass(cell)
    _, faceJ = components(facemetrics(grid))

    faceMJ = faceM * faceJ

    args = (law, cell, grid, MJ, MJI, faceMJ, numericalflux)
    new{typeof.(args)...}(args...)
  end
end

function (dg::DGSEM)(dq, q, time)
  cell = dg.cell
  grid = dg.grid
  device = Bennu.device(arraytype(cell))
  dim = ndims(cell)
  Nq = size(cell)[1]
  @assert all(size(cell) .== Nq)

  comp_stream = Event(device)

  workgroup = ntuple(i -> i <= dim ? Nq : 1, 3)
  ndrange = (length(grid) * workgroup[1], Base.tail(workgroup)...)
  comp_stream = volumeterm!(device, workgroup)(
    dg.law,
    dq,
    q,
    derivatives_1d(cell)[1],
    metrics(grid),
    dg.MJ,
    dg.MJI,
    points(grid),
    Val(dim),
    Val(Nq),
    Val(numberofstates(dg.law));
    ndrange,
    dependencies = comp_stream
  )

  Nfp = Nq ^ (dim  - 1)
  workgroup_face = (Nfp,)
  ndrange = (Nfp * length(grid),)
  faceix⁻, faceix⁺ = faceindices(grid)
  facenormal, _ = components(facemetrics(grid))
  comp_stream = surfaceterm!(device, workgroup_face)(
    dg.law,
    dq,
    q,
    Val(Bennu.connectivityoffsets(cell, Val(2))),
    dg.numericalflux,
    dg.MJI,
    faceix⁻,
    faceix⁺,
    dg.faceMJ,
    facenormal,
    boundaryfaces(grid),
    points(grid),
    Val(dim);
    ndrange,
    dependencies = comp_stream
  )

  wait(comp_stream)
end

@kernel function volumeterm!(law,
                             dq,
                             q,
                             D,
                             metrics,
                             MJ,
                             MJI,
                             points,
                             ::Val{dim},
                             ::Val{Nq},
                             ::Val{Ns}) where {dim, Nq, Ns}
  @uniform begin
    FT = eltype(law)
    Nq1 = Nq
    Nq2 = dim > 1 ? Nq : 1
    Nq3 = dim > 2 ? Nq : 1
  end

  l_F = @localmem FT (Nq1, Nq2, Nq3, dim, Ns)
  dqijk = @private FT (Ns,)
  p_MJ = @private FT (1,)

  e = @index(Group, Linear)
  i, j, k = @index(Local, NTuple)

  @inbounds begin
    ijk = i + Nq * (j - 1 + Nq * (k - 1))

    g = metrics[ijk, e].g
    
    qijk = q[ijk, e]
    x⃗ijk = points[ijk, e]
    MJijk = MJ[ijk, e]

    fijk = flux(law, qijk, x⃗ijk)

    @unroll for s in 1:Ns
      @unroll for d in 1:dim
        l_F[i, j, k, d, s] = g[d, 1] * fijk[1, s]
        if dim > 1
          l_F[i, j, k, d, s] += g[d, 2] * fijk[2, s]
        end
        if dim > 2
          l_F[i, j, k, d, s] += g[d, 3] * fijk[3, s]
        end
        l_F[i, j, k, d, s] *= MJijk
      end
    end
    
    fill!(dqijk, -zero(FT))
    source!(law, dqijk, qijk, x⃗ijk)

    @synchronize
    
    ijk = i + Nq * (j - 1 + Nq * (k - 1))
    MJIijk = MJI[ijk, e]

    @unroll for n in 1:Nq
      Dni = D[n, i] * MJIijk 
      Dnj = D[n, j] * MJIijk 
      Dnk = D[n, k] * MJIijk 
      @unroll for s in 1:Ns
        dqijk[s] += Dni * l_F[n, j, k, 1, s]
        if dim > 1
          dqijk[s] += Dnj * l_F[i, n, k, 2, s]
        end
        if dim > 2
          dqijk[s] += Dnk * l_F[i, j, n, 3, s]
        end
      end
    end

    dq[ijk, e] += dqijk
  end
end

@kernel function surfaceterm!(law,
                              dq,
                              q,
                              ::Val{faceoffsets},
                              numflux,
                              MJI,
                              faceix⁻,
                              faceix⁺,
                              faceMJ,
                              facenormal,
                              boundaryfaces,
                              points,
                              ::Val{dim}) where {faceoffsets, dim}
  @uniform begin
    FT = eltype(q)
    nfaces = 2 * dim
  end
  
  e⁻ = @index(Group, Linear)
  i = @index(Local, Linear)

  @inbounds begin
    @unroll for face in 1:nfaces
        j = i + faceoffsets[face]
        id⁻ = faceix⁻[j, e⁻]

        n⃗ = facenormal[j, e⁻]
        fMJ = faceMJ[j, e⁻]
       
        x⃗⁻ = points[id⁻]
        q⁻ = q[id⁻]
        boundarytag = boundaryfaces[face, e⁻] 
        if boundarytag == 0
          id⁺ = faceix⁺[j, e⁻]
          q⁺ = q[id⁺]
        else
          q⁺ = boundarystate(law, n⃗, x⃗⁻, q⁻, boundarytag)
        end

        nf = numflux(law, n⃗, x⃗⁻, q⁻, q⁺)
        dq[id⁻] -= fMJ * nf * MJI[id⁻]

        @synchronize(mod(face, 2) == 0)
    end
  end
end
