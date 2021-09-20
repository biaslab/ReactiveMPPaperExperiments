export HMMModel

struct HMMModel <: AbstractModel end

using Random
using Distributions
using DrWatson

function generate_data(::HMMModel, parameters)
    @unpack n, A, B, seed = parameters

    rng = MersenneTwister(seed)

    # Initial state
    z_0 = [1.0, 0.0, 0.0] 
	
	# one-hot encoding of the states and of the observations
    z = Vector{Vector{Float64}}(undef, n) 
    y = Vector{Vector{Float64}}(undef, n)
    
    z_prev = z_0
    
    for t in 1:n
        z[t] = random_vector(rng, Categorical(normalise(A * z_prev)))
        y[t] = random_vector(rng, Categorical(normalise(B * z[t])))
        z_prev = z[t]
    end
    
    return z, y
end

