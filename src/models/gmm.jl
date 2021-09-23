export GMMModel

struct GMMModel <: AbstractModel end

using Random
using Distributions
using DrWatson

function generate_data(::GMMModel, params)
    @unpack seed, nmixtures, cdistance, angle, n = params

	rng = MersenneTwister(seed)
	
	mixing = ones(nmixtures) # rand(rng, nmixtures)
	mixing = mixing ./ sum(mixing)
	
	switch = Categorical(mixing)
	
	mixtures = map(1:nmixtures) do index
		langle     = angle * (index - 1)
        basis_v    = cdistance * [ 1.0, 0.0 ]
        rotationm  = [ cos(langle) -sin(langle); sin(langle) cos(langle) ]
        mean       = rotationm * basis_v 
        covariance = rotationm * [ 16.0 0.0; 0.0 1.0 ] * transpose(rotationm)
		return MvNormal(mean, Matrix(Hermitian(covariance)))
	end
	
	mixture = MixtureModel(mixtures, mixing)
	
	return collect.(eachcol(rand(rng, mixture, n))), mixtures
end