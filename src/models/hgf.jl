export HGFModel

struct HGFModel <: AbstractModel end

using Random
using Distributions
using DrWatson

function generate_data(::HGFModel, parameters)
    @unpack n, τ_z, τ_y, κ, ω, seed = parameters

    rng = MersenneTwister(seed)

    z = Vector{Float64}(undef, n)
    s = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)

	z_σ = sqrt(inv(τ_z))
    y_σ = sqrt(inv(τ_y))
    
    z[1] = zero(Float64)
    s[1] = zero(Float64)
    y[1] = rand(rng, Normal(s[1], y_σ))
    
    for i in 2:n
        z[i] = rand(rng, Normal(z[i - 1], z_σ))
        s[i] = rand(rng, Normal(s[i - 1], sqrt(exp(κ * z[i] + ω))))
        y[i] = rand(rng, Normal(s[i], y_σ))
    end
    
    return z, s, y
end
