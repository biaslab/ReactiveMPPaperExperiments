### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ dafb5428-9d09-11eb-1d29-b9cd53c89545
begin 
	using Revise
	using Pkg
end

# ╔═╡ 2b6b67e0-bdd3-44d9-abc5-13482f0a1056
Pkg.activate("$(@__DIR__)/../") # To disable pluto's built-in pkg manager

# ╔═╡ 2a441550-27d1-4d02-870d-02cc9fdbba40
begin

    using ReactiveMPPaperExperiments
	using DrWatson, PlutoUI, Images
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random, Plots

	if !in(:PlutoRunner, names(Main))
		using PGFPlotsX
		pgfplotsx()
	end

end

# ╔═╡ f0d8d8a8-a881-4d83-b662-eec38dd732fd
md"""
Example: Inferring the bias of a coin
"""

# ╔═╡ f34341cb-f999-4df0-a0a4-856a4ec27dd2
md"""
#### Model Specification

In a Bayesian setting, the first step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

We will assume that the outcome of each coin flip is governed by the Bernoulli distribution, i.e.

$y_i \sim \text{Bernoulli}(\theta)$

where $y_i = 1$ represents "heads", $y_i = 0$ represents "tails", and $θ ∈ [ 0, 1 ]$ is the underlying probability of the coin landing heads up for a single coin flip.

We will choose the conjugate prior of the Bernoulli likelihood function defined above, namely the beta distribution, i.e.

$θ \sim \text{Beta}(a, b)$

where $a$ and $b$ are the hyperparameters that encode our prior beliefs about the possible values of $θ$. We will assign values to the hyperparameters in a later step.

"""

# ╔═╡ 1b61b60e-07fa-4217-85bb-f32128658188
@model function coin_model(n, a, b)
    
    θ ~ Beta(a, b)
	
	y = datavar(Float64, n)
	
	for i in 1:n
		y[i] ~ Bernoulli(θ)		
	end
    
    return y, θ
end

# ╔═╡ 5e382a05-aaf9-4e6d-93f4-3698c6c3d379
md"""
As you can see, GraphPPL.jl offer a model specification syntax that resembles closely to the mathematical equations defined above. `datavar` placeholders are used to indicate variables that take specific values at a later date. For example, the way we feed observations into the model is by iteratively assigning each of the observations in our dataset to the data variables `y`.
"""

# ╔═╡ c3f9d84e-b5c5-454d-8400-af107812a3d7
md"""
We will simulate coin toss process by sampling $n$ values from a Bernoulli distribution. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on $p$ fraction of the trials (on average). We also use a `seed` parameter to make our experiments reproducible.
"""

# ╔═╡ 54a367e2-2a16-46a8-b3de-50c9e4f1827f
# number of coin tosses
n_slider = @bind n ThrottledSlider(1:1000, default = 500) 

# ╔═╡ baafcf7c-52b4-458e-8de7-24cf26986fae
# p parameter of the Bernoulli distribution
p_slider = @bind p ThrottledSlider(0.0:0.01:1.0, default = 0.5)

# ╔═╡ 6d5d5123-4465-49de-aa9e-0828055da982
# Seed value used for data generation
seed_slider = @bind seed ThrottledSlider(1:1000, default = 42)

# ╔═╡ 16c2aee7-61ba-4440-9326-a1483e8872d0
begin 
	rng = MersenneTwister(seed)
	dataset = map(Float64, rand(rng, Bernoulli(p), n))
end

# ╔═╡ 66612a6f-111d-496c-83ea-e6f0de30757f
md"""
Once we have defined our model, the next step is to use ReactiveMP.jl API to run a reactive message-passing algorithm that solves our given inference problem. To do this, we need to specify which variables we are interested in. We obtain a posterior marginal updates stream by calling `getmarginal()` function and pass `θ` as an argument. 

We use `subscribe!` function from `Rocket.jl` to subscribe on updates and as soon as update is ready we simply save it in local variable which we return later as a result of the function.

To pass observations to our model we use `update!` function. Inference engine will wait for all observations in the model and will react as soon as all of them have been passed. 
"""

# ╔═╡ 2797b5df-a3da-426d-bb53-d810721a817f
md"""
### Inference
"""

# ╔═╡ 6e43fdb2-7b39-4f04-b526-e20976e9e3c9
function inference(data)
    model, (y, θ) = coin_model(n, 1.0, 1.0)
    
    θs     = nothing
    θ_sub  = subscribe!(getmarginal(θ), (θ) -> θs = θ)
    
    update!(y, data)
    
    return θs
end

# ╔═╡ 1ab48849-070a-4e93-b6f8-88ebd012caf1
estimated_θ = inference(dataset)

# ╔═╡ 894a2168-ea45-4021-a08a-121fe590fd86
md"""
### Results

|       |      |
| ----- | ---- |
| n     | $(n_slider)    |
| p     | $(p_slider)    |
| seed     | $(seed_slider)    |
"""

# ╔═╡ 081d2f7e-ac43-413e-9f65-932b3e959f11
md"""
Our `inference` function simply returns a posterior marginal distribution over θ parameter in our model. We can take a `mean` of that distribution to compare it with the real parameter used to generate data

|                        |                                           |
| ---------------------- | ----------------------------------------- |
| real_θ                 | $(round(p, digits = 4))                   |
| mean(estimated\_θ)     | $(round(mean(estimated_θ), digits = 4))   |
| var(estimated\_θ)      | $(round(var(estimated_θ), digits = 4))    |
| difference             | $(round(abs(p - mean(estimated_θ)), digits = 4))    |

As we can see θ parameter has been estimated correctly with a very high precision.
"""

# ╔═╡ Cell order:
# ╠═dafb5428-9d09-11eb-1d29-b9cd53c89545
# ╠═2b6b67e0-bdd3-44d9-abc5-13482f0a1056
# ╠═2a441550-27d1-4d02-870d-02cc9fdbba40
# ╟─f0d8d8a8-a881-4d83-b662-eec38dd732fd
# ╟─f34341cb-f999-4df0-a0a4-856a4ec27dd2
# ╠═1b61b60e-07fa-4217-85bb-f32128658188
# ╟─5e382a05-aaf9-4e6d-93f4-3698c6c3d379
# ╟─c3f9d84e-b5c5-454d-8400-af107812a3d7
# ╠═54a367e2-2a16-46a8-b3de-50c9e4f1827f
# ╠═baafcf7c-52b4-458e-8de7-24cf26986fae
# ╠═6d5d5123-4465-49de-aa9e-0828055da982
# ╠═16c2aee7-61ba-4440-9326-a1483e8872d0
# ╟─66612a6f-111d-496c-83ea-e6f0de30757f
# ╟─2797b5df-a3da-426d-bb53-d810721a817f
# ╠═6e43fdb2-7b39-4f04-b526-e20976e9e3c9
# ╠═1ab48849-070a-4e93-b6f8-88ebd012caf1
# ╟─894a2168-ea45-4021-a08a-121fe590fd86
# ╟─081d2f7e-ac43-413e-9f65-932b3e959f11
