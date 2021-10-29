using NLsolve: nlsolve

export solve!
export LSRK144, LSRK54
export RLSRK144, RLSRK54

function solve!(q, timeend, solver;
                after_step::Function = (x...) -> nothing,
                after_stage::Function = (x...) -> nothing)
  finalstep = false
  step = 0
  while true
    step += 1
    time = solver.time
    if time + solver.dt >= timeend
      solver.dt = timeend - time
      finalstep = true
    end
    dostep!(q, solver, after_stage)
    after_step(step, time, q)
    finalstep && break
  end
end

mutable struct LSRK{FT, AT, NS, RHS}
  dt::FT
  time::FT
  rhs!::RHS
  dq::AT
  rka::NTuple{NS, FT}
  rkb::NTuple{NS, FT}
  rkc::NTuple{NS, FT}

  function LSRK(rhs!, rka, rkb, rkc, q, dt, t0)
      FT = eltype(eltype(q))
      dq = similar(q)
      fill!(dq, zero(eltype(q)))
      AT = typeof(q)
      RHS = typeof(rhs!)
      new{FT, AT, length(rka), RHS}(FT(dt), FT(t0), rhs!, dq, rka, rkb, rkc)
  end
end

function dostep!(q, lsrk::LSRK, after_stage)
  @unpack rhs!, dq, rka, rkb, rkc, dt, time = lsrk
  for stage = 1:1
    stagetime = time + rkc[stage] * dt
    dq .*= rka[stage]
    rhs!(dq, q, stagetime)
    @. q += rkb[stage] * dt * dq
    after_stage(stagetime, q)
  end
  lsrk.time += dt
end

function LSRK54(rhs!, q, dt; t0 = 0)
  rka, rkb, rkc = coefficients_lsrk54()
  LSRK(rhs!, rka, rkb, rkc, q, dt, t0)
end

function LSRK144(rhs!, q, dt; t0 = 0)
  rka, rkb, rkc = coefficients_lsrk144()
  LSRK(rhs!, rka, rkb, rkc, q, dt, t0)
end

mutable struct RLSRK{FT, AT, NS, RHS}
  dt::FT
  time::FT
  γ::FT
  rhs!::RHS
  dq::AT
  q0::AT
  k::AT
  rka::NTuple{NS, FT}
  rkb::NTuple{NS, FT}
  rkb_full::NTuple{NS, FT}
  rkc::NTuple{NS, FT}

  function RLSRK(rhs!, rka, rkb, rkc, q, dt, t0)
      FT = eltype(eltype(q))
      dq = similar(q)
      fill!(dq, zero(eltype(q)))
      q0 = similar(q)
      k = similar(q)
      # construct standard RK b coefficients
      rkb_full = zeros(FT, length(rkb))
      rkb_full[end] = rkb[end]
      for i in length(rkb)-1:-1:1
        rkb_full[i] = rka[i+1] * rkb_full[i+1] + rkb[i]
      end
      γ = FT(1)

      AT = typeof(q)
      RHS = typeof(rhs!)
      new{FT, AT, length(rka), RHS}(FT(dt), FT(t0), γ, rhs!, dq, q0, k,
                                    rka, rkb, Tuple(rkb_full), rkc)
  end
end

function dostep!(q, rlsrk::RLSRK{FT}, after_stage) where {FT}
  @unpack rhs!, dq, q0, k, dt, time = rlsrk
  @unpack rka, rkb, rkb_full, rkc = rlsrk

  q0 .= q
  η0 = entropyintegral(rhs!, q0)
  dη = -zero(FT)

  for stage = 1:length(rka)
    dq .*= rka[stage]
    stagetime = time + rkc[stage] * dt
    rhs!(k, q, stagetime; increment = false)
    dq .+= k
    dη += rkb_full[stage] * dt * entropyproduct(rhs!, q, k)

    @. q += rkb[stage] * dt * dq
    after_stage(stagetime, q)
  end

  @. k = q - q0
  function r(F, γ)
    @. q = q0 + γ[1] * k
    F[1] = entropyintegral(rhs!, q) - η0 - γ[1] * dη
  end
  function dr(J, γ)
    @. q = q0 + γ[1] * k
    J[1] = entropyproduct(rhs!, q, k) - dη
  end
  sol = nlsolve(r, dr, [rlsrk.γ]; ftol=10eps(FT))
  rlsrk.γ = sol.zero[1]
  @assert 0.7 < rlsrk.γ < 1.3
  q .= q0 .+ rlsrk.γ * k

  rlsrk.time += dt
end

function RLSRK54(rhs!, q, dt; t0 = 0)
  rka, rkb, rkc = coefficients_lsrk54()
  RLSRK(rhs!, rka, rkb, rkc, q, dt, t0)
end

function RLSRK144(rhs!, q, dt; t0 = 0)
  rka, rkb, rkc = coefficients_lsrk144()
  RLSRK(rhs!, rka, rkb, rkc, q, dt, t0)
end

function coefficients_lsrk54()
  rka = (
    (0),
    (-567301805773 // 1357537059087),
    (-2404267990393 // 2016746695238),
    (-3550918686646 // 2091501179385),
    (-1275806237668 // 842570457699),
  )

  rkb = (
    (1432997174477 // 9575080441755),
    (5161836677717 // 13612068292357),
    (1720146321549 // 2090206949498),
    (3134564353537 // 4481467310338),
    (2277821191437 // 14882151754819),
  )

  rkc = (
    (0),
    (1432997174477 // 9575080441755),
    (2526269341429 // 6820363962896),
    (2006345519317 // 3224310063776),
    (2802321613138 // 2924317926251),
  )

  rka, rkb, rkc
end

function coefficients_lsrk144()
  rka = (
     0,
    -0.7188012108672410,
    -0.7785331173421570,
    -0.0053282796654044,
    -0.8552979934029281,
    -3.9564138245774565,
    -1.5780575380587385,
    -2.0837094552574054,
    -0.7483334182761610,
    -0.7032861106563359,
     0.0013917096117681,
    -0.0932075369637460,
    -0.9514200470875948,
    -7.1151571693922548,
  )

  rkb = (
    0.0367762454319673,
    0.3136296607553959,
    0.1531848691869027,
    0.0030097086818182,
    0.3326293790646110,
    0.2440251405350864,
    0.3718879239592277,
    0.6204126221582444,
    0.1524043173028741,
    0.0760894927419266,
    0.0077604214040978,
    0.0024647284755382,
    0.0780348340049386,
    5.5059777270269628,
  )

  rkc = (
    0,
    0.0367762454319673,
    0.1249685262725025,
    0.2446177702277698,
    0.2476149531070420,
    0.2969311120382472,
    0.3978149645802642,
    0.5270854589440328,
    0.6981269994175695,
    0.8190890835352128,
    0.8527059887098624,
    0.8604711817462826,
    0.8627060376969976,
    0.8734213127600976,
  )

  rka, rkb, rkc
end
