export average_mse

using Distributions
using Random

import ReactiveMP
import Turing
import ForneyLab

# Gaussian Average MSE

function average_mse(real, estimated::E; seed = 42, nsamples = 2_000, posf = abs2) where { E <: AbstractVector{ <: ReactiveMP.Marginal } }
    rng     = MersenneTwister(seed)
    average = zero(first(real))

    for _ in 1:nsamples
        samples = map(e -> rand(rng, ReactiveMP.getdata(e)), estimated)
        mse = sqrt.(mapreduce(r -> posf.(r[1] - r[2]), +, zip(real, samples)))
        average += (mse ./ nsamples)
    end

    return sum(average)
end

function average_mse(real, estimated::E; seed = 42, nsamples = 2_000) where { E <: AbstractVector{ <: ForneyLab.ProbabilityDistribution{<:ForneyLab.Multivariate, <: ForneyLab.Gaussian } } }
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

function average_mse(real, estimated::Turing.Chains, s::Symbol, ::Type{ MvNormal }; seed = 42, nsamples = 2_000)
    d = length(first(real))
    
    reshape_turing_data = (data) -> transpose(reshape(data, (d, Int(length(data) / d))))

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

# Categorical Average MSE

function average_mse(real, estimated::E; seed = 42, nsamples = 2_000) where { E <: AbstractVector{ <: ForneyLab.ProbabilityDistribution{<:ForneyLab.Univariate, <: ForneyLab.Categorical } } }
    rng     = MersenneTwister(seed)
    average = zero(first(real))

    dists = map(e -> Categorical(ForneyLab.unsafeMeanVector(e)), estimated)

    for _ in 1:nsamples
        samples = map(e -> rand(rng, e), dists)
        mse = sqrt.(mapreduce(r -> abs.(r[1] - r[2]), +, zip(real, samples)))
        average += (mse ./ nsamples)
    end

    return sum(average)
end

function average_mse(real, estimated::Turing.Chains, s::Symbol, ::Type{ Distributions.Categorical }; seed = 42, nsamples = 2_000)
    rng = MersenneTwister(seed)
    k = length(first(real))
    
    N = length(real)
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

    return mapreduce(_ -> sum(abs.(map(e -> rand(rng, e), cat_estimated) .- argmax.(real))), +, 1:nsamples) ./ nsamples
end