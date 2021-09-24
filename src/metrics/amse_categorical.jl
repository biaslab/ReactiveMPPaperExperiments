
import Distributions
import ReactiveMP

function average_mse(::Type{ <: AbstractVector }, ::Type{ <: Distributions.Categorical }, states, estimated)
    @assert all(s -> sum(s) == 1 && findnext(==(1), s, 1) !== nothing, states) "Corrupted one-hot encoded data"
    return average_mse(eltype(first(states)), Categorical, map(argmax, states), estimated)
end

## ReactiveMP

function average_mse(::Type{ <: Real }, ::Type{ <: Distributions.Categorical }, states, estimated)
    return mapreduce(+, zip(states, estimated)) do (s, e)
        return mapreduce(+, enumerate(ReactiveMP.probvec(e))) do (i, p)
            return abs(i - s) * p
        end
    end
end

## ForneyLab

function average_mse(::Type{T}, ::Type{ <: ForneyLab.Categorical }, states, estimates) where T
    converted = map(estimates) do e 
        return Distributions.Categorical(normalise(ForneyLab.unsafeMeanVector(e)))
    end
    return average_mse(states, converted)
end

## Turing 

function average_mse(states::Vector{ <: Vector }, estimated::Turing.Chains, s::Symbol, ::Type{ Distributions.Categorical })
    k = length(first(states))
    
    N = length(states)
    m = zeros(k, N)

    for t = 1:N
        dd = [ (i, count(==(i), Turing.group(estimated, s).value.data[:, t])) for i in float.(1:k) ]
        num1, num2, num3 = dd[1][2], dd[2][2], dd[3][2]
        num = num1 + num2 + num3
        m[1,t] = num1/num
        m[2,t] = num2/num
        m[3,t] = num3/num
    end

    cat_estimated = map(e -> Distributions.Categorical(normalise(e)), eachcol(m));

    return average_mse(states, cat_estimated)
end
