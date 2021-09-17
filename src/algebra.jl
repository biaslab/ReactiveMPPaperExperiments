export rotation_matrix, diagonal_matrix, random_posdef_matrix

using LinearAlgebra
using Random

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