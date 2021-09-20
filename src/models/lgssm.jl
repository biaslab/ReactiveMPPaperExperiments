export LGSSMModel

struct LGSSMModel <: AbstractModel end

using Random
using Distributions
using DrWatson

function generate_data(::LGSSMModel, parameters)

    @unpack n, d, A, B, P, Q, seed = parameters

    @assert size(A) === (d, d)
    @assert size(B) === (d, d)
    @assert size(P) === (d, d)
    @assert size(Q) === (d, d)

    rng    = MersenneTwister(seed)
    x_prev = zeros(d)

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, P))
        y[i] = rand(rng, MvNormal(B * x[i], Q))

        x_prev = x[i]
    end
   
    return x, y
end

