export rotation_matrix, diagonal_matrix, random_posdef_matrix

using LinearAlgebra
using Random

import Distributions

function rotation_matrix(θ)
    return [ 
        cos(θ) -sin(θ); 
        sin(θ) cos(θ) 
    ]
end

function diagonal_matrix(values)
    return Matrix(Diagonal(values))
end

function random_posdef_matrix(rng, dimension)
    L = rand(rng, dimension, dimension)
	return L' * L
end

function random_vector(rng, distribution::Distributions.Categorical) 
    k = ncategories(distribution)
    s = zeros(k)
    s[ rand(rng, distribution) ] = 1.0
    s
end

function normalise(a)
	return a ./ sum(a)
end