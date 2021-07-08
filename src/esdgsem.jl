export ESDGSEM

struct ESDGSEM{L, C, G, A1, A2, A3, NFV, NFS}
  law::L
  cell::C
  grid::G
  MJ::A1
  MJI::A2
  faceMJ::A3
  volume_numericalflux::NFV
  surface_numericalflux::NFS

  function ESDGSEM(; law, cell, grid, volume_numericalflux, surface_numericalflux)
    M = mass(cell)
    _, J = components(metrics(grid))
    MJ = M * J
    MJI = 1 ./ MJ

    faceM = facemass(cell)
    _, faceJ = components(facemetrics(grid))

    faceMJ = faceM * faceJ

    args = (law, cell, grid, MJ, MJI, faceMJ, volume_numericalflux, surface_numericalflux)
    new{typeof.(args)...}(args...)
  end
end

function (dg::ESDGSEM)(dq, q, time)
  cell = dg.cell
  grid = dg.grid
  device = Bennu.device(arraytype(cell))
  dim = ndims(cell)
  Nq = size(cell)[1]
  @assert all(size(cell) .== Nq)
  @assert(length(eltype(q)) == numberofstates(dg.law))

  comp_stream = Event(device)

  workgroup = ntuple(i -> i <= min(2, dim) ? Nq : 1, 3)
  ndrange = (length(grid) * workgroup[1], Base.tail(workgroup)...)
  comp_stream = esvolumeterm!(device, workgroup)(
    dg.law,
    dq,
    q,
    derivatives_1d(cell)[1],
    dg.volume_numericalflux,
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
    dg.surface_numericalflux,
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

@kernel function esvolumeterm!(law,
                               dq,
                               q,
                               D,
                               volume_numericalflux,
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

    q1 = MVector{Ns, FT}(undef)
    q2 = MVector{Ns, FT}(undef)
    x⃗1 = MVector{dim, FT}(undef)
    x⃗2 = MVector{dim, FT}(undef)
  end

  dqijk = @private FT (Ns,)

  pencil_q = @private FT (Ns, Nq3)
  pencil_x⃗ = @private FT (dim, Nq3)
  pencil_g3 = @private FT (3, Nq3)
  pencil_MJ = @private FT (Nq3,)

  l_q = @localmem FT (Nq1, Nq2, Ns)
  l_x⃗ = @localmem FT (Nq1, Nq2, dim)
  l_g = @localmem FT (Nq1, Nq2, min(2, dim), dim)

  e = @index(Group, Linear)
  i, j = @index(Local, NTuple)

  @inbounds begin
    @unroll for k in 1:Nq3
      ijk = i + Nq * (j - 1 + Nq * (k - 1))

      pencil_MJ[k] = MJ[ijk, e]
      @unroll for s in 1:Ns
        pencil_q[s, k] = q[ijk, e][s]
      end
      @unroll for d in 1:dim
        pencil_x⃗[d, k] = points[ijk, e][d]
      end
      if dim > 2
        @unroll for d in 1:dim
          pencil_g3[d, k] = metrics[ijk, e].g[3, d]
          pencil_g3[d, k] *= pencil_MJ[k]
        end
      end
    end

    @unroll for k in 1:Nq3
      @synchronize
      ijk = i + Nq * (j - 1 + Nq * (k - 1))

      @unroll for s in 1:Ns
        l_q[i, j, s] = pencil_q[s, k]
      end
      @unroll for d in 1:dim
        l_x⃗[i, j, d] = pencil_x⃗[d, k]
      end

      MJk = pencil_MJ[k]
      @unroll for d in 1:dim
        l_g[i, j, 1, d] = MJk * metrics[ijk, e].g[1, d]
        if dim > 1
          l_g[i, j, 2, d] = MJk * metrics[ijk, e].g[2, d]
        end
      end

      @synchronize

      fill!(dqijk, -zero(FT))

      @unroll for s in 1:Ns
        q1[s] = l_q[i, j, s]
      end
      @unroll for d in 1:dim
        x⃗1[d] = l_x⃗[i, j, d]
      end

      source!(law, dqijk, q1, x⃗1)

      MJIijk = 1 / pencil_MJ[k]
      @unroll for n in 1:Nq
        @unroll for s in 1:Ns
          q2[s] = l_q[n, j, s]
        end
        @unroll for d in 1:dim
          x⃗2[d] = l_x⃗[n, j, d]
        end

        f = twopointflux(volume_numericalflux, law, q1, x⃗1, q2, x⃗2)
        @unroll for s in 1:Ns
          Din = MJIijk * D[i, n]
          Dni = MJIijk * D[n, i]
          @unroll for d in 1:dim
            dqijk[s] -= Din * l_g[i, j, 1, d] * f[d, s]
            dqijk[s] += f[d, s] * l_g[n, j, 1, d] * Dni
          end
        end

        if dim > 1
          @unroll for s in 1:Ns
            q2[s] = l_q[i, n, s]
          end
          @unroll for d in 1:dim
            x⃗2[d] = l_x⃗[i, n, d]
          end
          f = twopointflux(volume_numericalflux, law, q1, x⃗1, q2, x⃗2)
          @unroll for s in 1:Ns
            Djn = MJIijk * D[j, n]
            Dnj = MJIijk * D[n, j]
            @unroll for d in 1:dim
              dqijk[s] -= Djn * l_g[i, j, 2, d] * f[d, s]
              dqijk[s] += f[d, s] * l_g[i, n, 2, d] * Dnj
            end
          end
        end

        if dim > 2
          @unroll for s in 1:Ns
            q2[s] = pencil_q[s, n]
          end
          @unroll for d in 1:dim
            x⃗2[d] = pencil_x⃗[d, n]
          end
          f = twopointflux(volume_numericalflux, law, q1, x⃗1, q2, x⃗2)
          @unroll for s in 1:Ns
            Dkn = MJIijk * D[k, n]
            Dnk = MJIijk * D[n, k]
            @unroll for d in 1:dim
              dqijk[s] -= Dkn * pencil_g3[d, k] * f[d, s]
              dqijk[s] += f[d, s] * pencil_g3[d, n] * Dnk
            end
          end
        end
      end

      ijk = i + Nq * (j - 1 + Nq * (k - 1))
      dq[ijk, e] += dqijk
    end
  end
end
