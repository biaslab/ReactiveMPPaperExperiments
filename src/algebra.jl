export rotation_matrix, random_rotation_matrix, diagonal_matrix, random_posdef_matrix

using LinearAlgebra
using Random

import Distributions

function rotation_matrix(θ)
    return [ 
        cos(θ) -sin(θ); 
        sin(θ) cos(θ) 
    ]
end

function random_rotation_matrix(rng, dimension)
    R = Matrix(Diagonal(ones(dimension)))

    θ = π/20 * rand(rng)

    for i in 1:dimension 
        for j in (i + 1):dimension
            S = Matrix(Diagonal(ones(dimension)))
            S[i, i] = cos(θ)
            S[j, j] = cos(θ)
            S[i, j] = sin(θ)
            S[j, i] = -sin(θ)
            R = R * S
        end
    end
    return R
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