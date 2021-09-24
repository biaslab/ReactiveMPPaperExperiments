
import ReactiveMP
import Turing
import ForneyLab

## ReactiveMP

function average_mse(::Type{ <: Real }, ::Type{ <: ReactiveMP.UnivariateGaussianDistributionsFamily }, states, estimated)
    return mapreduce(+, zip(states, estimated)) do (s, e)
        return var(e) + abs2(s - mean(e))
    end
end

function average_mse(::Type{ <: AbstractVector }, ::Type{ <: ReactiveMP.MultivariateGaussianDistributionsFamily }, states, estimated)
    return mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return tr(cov(e)) + diff' * diff 
    end
end

## ForneyLab

function average_mse(::Type{ <: AbstractVector }, ::Type{ <: ForneyLab.Gaussian }, states, estimates)
    converted = map(estimates) do e 
        return ReactiveMP.MvNormalMeanCovariance(ForneyLab.unsafeMeanCov(e)...)
    end
    return average_mse(states, converted)
end

function average_mse(::Type{ <: Real }, ::Type{ <: ForneyLab.Gaussian }, states, estimates)
    converted = map(estimates) do e 
        return ReactiveMP.NormalMeanVariance(ForneyLab.unsafeMeanCov(e)...)
    end
    return average_mse(states, converted)
end


## Turing

function average_mse(states::AbstractVector, chains::Turing.Chains, s::Symbol, ::Type{ MvNormal })
    d = length(first(states))
    
    reshape_turing_data = (data) -> transpose(reshape(data, (d, Int(length(data) / d))))

    n_turing = length(states)
    samples  = get(chains, s)
    
    means = reshape_turing_data([ mean(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect
    covs = reshape_turing_data([ var(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect

    estimated = map(e -> MvNormal(vec(e[1]), vec(e[2])), zip(means, covs))
    
    return mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return tr(cov(e)) + diff' * diff 
    end
end

function average_mse(states::AbstractVector, chains::Turing.Chains, s::Symbol, ::Type{ Normal })
    d = length(first(states))
    
    reshape_turing_data = (data) -> transpose(reshape(data, (d, Int(length(data) / d))))

    n_turing = length(states)
    samples  = get(chains, s)
    
    means = reshape_turing_data([ mean(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect
    stds = reshape_turing_data([ std(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect

    estimated = map(e -> Normal(e[1], e[2]), zip(means, stds))
    
    return mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return var(e) + diff' * diff 
    end
end