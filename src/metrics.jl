export average_mse

using Distributions
using Random

import ReactiveMP
import Turing
import ForneyLab

function average_mse(real, estimated::E; seed = 42, nsamples = 10_000) where { E <: AbstractVector{ <: ReactiveMP.Marginal } }
    rng     = MersenneTwister(seed)
    average = zero(first(real))

    for _ in 1:nsamples
        samples = map(e -> rand(rng, ReactiveMP.getdata(e)), estimated)
        mse = sqrt.(mapreduce(r -> abs2.(r[1] - r[2]), +, zip(real, samples)))
        average += (mse ./ nsamples)
    end

    return sum(average)
end

function average_mse(real, estimated::E; seed = 42, nsamples = 10_000) where { E <: AbstractVector{ <: ForneyLab.ProbabilityDistribution{<:ForneyLab.Multivariate, <: ForneyLab.Gaussian } } }
    rng     = MersenneTwister(seed)
    average = zero(first(real))

    dists = map(e -> MvNormal(ForneyLab.unsafeMean(e), ForneyLab.unsafeCov(e)), estimated)

    for _ in 1:nsamples
        samples = map(e -> rand(rng, e), dists)
        mse = sqrt.(mapreduce(r -> abs2.(r[1] - r[2]), +, zip(real, samples)))
        average += (mse ./ nsamples)
    end

    return sum(average)
end

function average_mse(real, estimated::Turing.Chains, s::Symbol, ::Type{ MvNormal }; seed = 42, nsamples = 10_000)
    reshape_turing_data = (data) -> transpose(reshape(data, (2, Int(length(data) / 2))))
    
    d = length(first(real))
    n_turing = length(real)
    samples  = get(estimated, s)
    
    means = reshape_turing_data([ mean(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect
    stds = reshape_turing_data([ std(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect

    dists = map(e -> MvNormal(e[1], e[2]), zip(means, stds))
    rng   = MersenneTwister(seed)

    average = zero(first(real))

    for _ in 1:nsamples
        samples = map(e -> rand(rng, e), dists)
        mse = sqrt.(mapreduce(r -> abs2.(r[1] - r[2]), +, zip(real, samples)))
        average += (mse ./ nsamples)
    end

    return sum(average)
end