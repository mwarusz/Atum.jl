export weightednorm
export entropyintegral

using LinearAlgebra: norm

weightednorm(dg, q, p = 2; componentwise=false) =
  weightednorm(q, Val(p), dg.MJ, componentwise)

function weightednorm(q, ::Val{p}, MJ, componentwise) where {p}
  n = map(components(q)) do f
    sum(MJ .* abs.(f) .^ p) ^ (1 // p)
  end
  componentwise ? n : norm(n, p)
end

function weightednorm(q, ::Val{Inf}, _, componentwise)
  n = map(components(q)) do f
    maximum(abs.(f))
  end
  componentwise ? n : norm(n, Inf)
end

entropyintegral(dg, q) = sum(dg.MJ .* entropy.(Ref(dg.law), q, dg.auxstate))
