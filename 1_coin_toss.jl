### A Pluto.jl notebook ###
# v0.14.1

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
using Revise

# ╔═╡ dd08386e-a5e6-40ef-b7d1-1163ce3926a9
using PlutoUI

# ╔═╡ 70f1e0a2-3f9c-42be-98e0-334c5abb08e7
using Rocket, ReactiveMP, GraphPPL, Distributions, Plots

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
We will simulate coin toss process by sampling $n$ values from a Bernoulli distribution. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on $p$ fraction of the trials (on average).
"""

# ╔═╡ 54a367e2-2a16-46a8-b3de-50c9e4f1827f
# number of coin tosses
@bind n Slider(1:1000, default=500, show_value=true) 

# ╔═╡ baafcf7c-52b4-458e-8de7-24cf26986fae
# p parameter of the Bernoulli distribution
@bind p Slider(0.0:0.01:1.0, default=0.5, show_value=true)

# ╔═╡ 16c2aee7-61ba-4440-9326-a1483e8872d0
dataset = map(Float64, rand(Bernoulli(p), n))

# ╔═╡ 66612a6f-111d-496c-83ea-e6f0de30757f
md"""
Once we have defined our model, the next step is to use ReactiveMP.jl API to run a reactive message-passing algorithm that solves our given inference problem. To do this, we need to specify which variables we are interested in. We obtain a posterior marginal updates stream by calling `getmarginal()` function and pass `θ` as an argument. 

We use `subscribe!` function from `Rocket.jl` to subscribe on updates and as soon as update is ready we simply save it in local variable which we return later as a result of the function.

To pass observations to our model we use `update!` function. Inference engine will wait for all observations in the model and will react as soon as all of them have been passed. 
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

# ╔═╡ 6c418f65-3fc2-4bd0-bfa2-6869f37ddbc8
mean(estimated_θ)

# ╔═╡ Cell order:
# ╠═dafb5428-9d09-11eb-1d29-b9cd53c89545
# ╠═dd08386e-a5e6-40ef-b7d1-1163ce3926a9
# ╠═70f1e0a2-3f9c-42be-98e0-334c5abb08e7
# ╟─f0d8d8a8-a881-4d83-b662-eec38dd732fd
# ╟─f34341cb-f999-4df0-a0a4-856a4ec27dd2
# ╠═1b61b60e-07fa-4217-85bb-f32128658188
# ╟─5e382a05-aaf9-4e6d-93f4-3698c6c3d379
# ╟─c3f9d84e-b5c5-454d-8400-af107812a3d7
# ╠═54a367e2-2a16-46a8-b3de-50c9e4f1827f
# ╠═baafcf7c-52b4-458e-8de7-24cf26986fae
# ╠═16c2aee7-61ba-4440-9326-a1483e8872d0
# ╟─66612a6f-111d-496c-83ea-e6f0de30757f
# ╠═6e43fdb2-7b39-4f04-b526-e20976e9e3c9
# ╠═1ab48849-070a-4e93-b6f8-88ebd012caf1
# ╠═6c418f65-3fc2-4bd0-bfa2-6869f37ddbc8
