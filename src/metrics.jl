export average_mse

using Distributions
using Random

import ReactiveMP
import Turing
import ForneyLab

function average_mse end

function average_mse(states::AbstractVector, estimated::AbstractVector)
    return average_mse(eltype(states), eltype(estimated), states, estimated)
end

function average_mse(::Type{T}, ::Type{ Any }, states, estimated) where { T }
    return average_mse(T, typeof(first(estimated)), states, estimated)
end

## ReactiveMP generic 

function average_mse(::Type{T}, ::Type{ <: ReactiveMP.Marginal }, states, estimated) where { T, F }
    return average_mse(T, typeof(ReactiveMP.getdata(first(estimated))), states, estimated)
end

## ForneyLab generic

function average_mse(::Type{T}, ::Type{ ForneyLab.ProbabilityDistribution }, states, estimated) where { T }
    return average_mse(T, typeof(first(estimated)), states, estimated)
end

function average_mse(::Type{T}, ::Type{ <: ForneyLab.ProbabilityDistribution{V, F} }, states, estimated) where { T, V, F }
    return average_mse(T, F, states, estimated)
end