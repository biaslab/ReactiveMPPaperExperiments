### A Pluto.jl notebook ###
# v0.14.2

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

# ╔═╡ d160581c-9d1f-11eb-05f7-c5f29954488b
using Revise

# ╔═╡ 95beaa74-12b5-4bf1-aeb1-a9e726c49cc9
using DrWatson

# ╔═╡ 6f1ba6c5-9359-40a0-b069-7194dee87e14
begin 
	@quickactivate "ReactiveMPPaperExperiments"
	using ReactiveMPPaperExperiments
end

# ╔═╡ f1cede2f-0e34-497b-9913-3204e9c75fd7
begin
	using PlutoUI, Images
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random, Plots
	if !in(:PlutoRunner, names(Main))
		using PGFPlotsX
		pgfplotsx()
	end
end

# ╔═╡ bbb878a0-1854-4bc4-9274-47edc8899795
md"""
#### Linear Multivariate Gaussian State-space Model

In this demo, the goal is to perform both Kalman filtering and smoothing inference algorithms with a state-space model (SSM).

We wil use the following model:

```math
\begin{equation}
  \begin{aligned}
    \mathbf{x}_k & \sim \, \mathcal{N}(\mathbf{A}\mathbf{x}_{k - 1}, \mathcal{P}) \\
    \mathbf{y}_k & \sim \, \mathcal{N}(\mathbf{B}\mathbf{x}_{k}, \mathcal{Q}) \\
  \end{aligned}
\end{equation}
```

In this model, we denote by $\mathbf{x}_k$ the current state of the system (at time step $k$), by $\mathbf{x}_{k - 1}$ the previous state at time $k-1$, $\mathbf{A}$ and $\mathbf{B}$ are a constant system inputs and $\mathbf{y}_k$ is a noisy observation of $\mathbf{x}_k$. We further assume that the states and the observations are corrupted by i.i.d. Gaussian noise with variances $\mathcal{P}$ and $\mathcal{Q}$ respectively.

$(load(projectdir("figures", "ssm_model.png")))

The SSM can be represented by the following factor graph, where the pictured section is chained over time:

For smoothing (forward-backward) estimation in ReactiveMP.jl, it is possible to build a full graph of the model.
"""

# ╔═╡ 9a8ce058-e7c3-4730-b4bd-b8782cead88f
md"""
### Model specification for full graph
"""

# ╔═╡ e39f30bf-5b19-4743-9a0e-16cafeed8d13
@model function linear_gaussian_ssm_full_graph(n, A, B, P, Q)
     
	# We create a vector of random variables with length n
    x = randomvar(n) 
	
	# Create a vector of observations with length n
    y = datavar(Vector{Float64}, n) 
	
	# We create a `constvar` references for constants in our model
	# to hint inference engine and to make it a little bit more efficient
	cA = constvar(A)
	cB = constvar(B)
	cP = constvar(P)
	cQ = constvar(Q)
    
	# Set a prior distribution for x[1]
    x[1] ~ MvGaussianMeanCovariance([ 0.0, 0.0 ], [ 100.0 0.0; 0.0 100.0 ]) 
    y[1] ~ MvGaussianMeanCovariance(cB * x[1], cQ)
    
    for t in 2:n
        x[t] ~ MvGaussianMeanCovariance(cA * x[t - 1], cP)
        y[t] ~ MvGaussianMeanCovariance(cB * x[t], cQ)    
    end
    
    return x, y
end

# ╔═╡ b6c7d8e3-c8c4-4b39-8c53-cbcb5f046b9b
md"""
For testing purposes we can use synthetically generated data where underlying data generation process matches our model specification.
"""

# ╔═╡ dd06f508-d820-4a56-92eb-04d821c1f215
function generate_data(; n, A, B, P, Q, seed)
	
	# We create a local RNG to make our results reproducable
    rng = MersenneTwister(seed)

    x_prev = zeros(2)

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, P))
        y[i] = rand(rng, MvNormal(B * x[i], Q))

        x_prev = x[i]
    end
   
    return x, y
end

# ╔═╡ 210d41a9-a8ff-4c24-9b88-524bed03cd7f
md"""
Pluto allows us to interactively explore and experiment with our models. Here we will create a set of sliders for later use. These sliders will allow us to dinamicaly change our model and data generation parameters and see changes immediatelly.
"""

# ╔═╡ 3011f498-9319-4dee-ba30-342ae0a2dc07
begin
	# Seed for random number generator for full graph
	seed_smoothing_slider = @bind(
		seed_smoothing, 
		ThrottledSlider(1:100, default = 42, show_value = true)
	)
	
	# Number of observations on our model for full graph
	n_smoothing_slider = @bind(
		n_smoothing, 
		ThrottledSlider(1:200, default = 100, show_value = true)
	)
	
	# θ parameter is a rotation angle for transition matrix
	θ_smoothing_slider = @bind(
		θ_smoothing, 
		ThrottledSlider(
			range(0.0, π/2, length = 100), default = π/20, show_value = true
		)
	)
end;

# ╔═╡ 7dcd84fd-c505-4f97-875d-49decba5c3f2
md"""
### Synthetic data

Lets first see how our data look like. We plot an individual line for each dimension in our dataset. To make it uncluttered we generate 2-dimensional observations with rotation matrix as a transition matrix with θ parameter which represents rotation angle
"""

# ╔═╡ ebc733ef-6638-4e42-a007-f2464ce3b5cf
begin
	A_smoothing = [ 
		cos(θ_smoothing) -sin(θ_smoothing); 
		sin(θ_smoothing) cos(θ_smoothing) 
	]
	B_smoothing = [ 1.3 0.0; 0.0 0.7 ]
	P_smoothing = [ 1.0 0.0; 0.0 1.0 ]
	Q_smoothing = [ 1.0 0.0; 0.0 1.0 ]
end

# ╔═╡ 2ce93b39-70ea-4b33-b9df-64e6ade6f896
md"""
|      |     |
| ---- | --- |
| seed | $(seed_smoothing_slider) |
| n    | $(n_smoothing_slider) |
| θ    | $(θ_smoothing_slider) |
"""

# ╔═╡ b0831de2-2aeb-432b-8987-872f4c5d74f0
x_smoothing, y_smoothing = generate_data(
	n    = n_smoothing, 
	A    = A_smoothing, 
	B    = B_smoothing, 
	P    = P_smoothing, 
	Q    = Q_smoothing, 
	seed = seed_smoothing
)

# ╔═╡ 934ad4d3-bb47-4174-b3d1-cbd6f8e5d75e
md"""
You may try to change data generation process parameters to immediatelly see how it affects data. We plot lines for real states $x_i$, and we plot scatter dots for noisy observations $y_i$.
"""

# ╔═╡ 099aa726-7694-4b38-875d-a015effc9d3a
begin 
	local reshape_data = (data) -> transpose(reduce(hcat, data))
	
	local ylimit = (-20, 20)
	
	local x_label = [ "x[:, 1]" "x[:, 2]" ]
	local y_label = [ "observations[:, 1]" "observations[:, 2]" ]
	
	local x_reshaped = x_smoothing |> reshape_data
	local y_reshaped = y_smoothing |> reshape_data
	
	local p = plot()
	local range = 1:n_smoothing
	
	p = plot!(p, range, x_reshaped, label = x_label)
	p = scatter!(p, range, y_reshaped, ms = 3, alpha = 0.5, label = y_label)
	p = plot!(p, legend = :bottomleft, ylimit = ylimit)
	
	@saveplot p "lgssm_smoothing_data"
end

# ╔═╡ 2530cf00-52c1-4c44-8d62-a3e4f0d411bc
md"""
### Inference

Next we need to define our inference procedure. Our model has no loops in it so we can easily perform sum-product. To do that we simply subscribe on posterior marginal updates in our model with `subscribe!` function and pass our observations with `update!` function. Here is a general template for inference function:
"""

# ╔═╡ fb94e6e9-10e4-4f9f-95e6-43cdd9184c09
function inference_full_graph(observations, A, B, P, Q)
	
	# We create a full graph based on how many observations
	# we have in our dataset
    n = length(observations) 
    
	# We call a `linear_gaussian_ssm_full_graph` function 
	# from our model specification above
    model, (x, y) = linear_gaussian_ssm_full_graph(
		n, A, B, P, Q, options = (limit_stack_depth = 100, )
	)
    
	# Rocket.jl provides some handy default actors
	# `buffer` actor simply copies all received updates 
	# into an internal buffer with length n
    xbuffer = buffer(Marginal, n)
    
	# For a collection of random variables we can use 
	# `getmarginals()` function which returns a stream of vectors
    xsubscription = subscribe!(getmarginals(x), xbuffer)
    
    update!(y, observations)
    
	# Usually we need to unsubscribe every time we're done with our model
    unsubscribe!(xsubscription)
    
    return getvalues(xbuffer)
end

# ╔═╡ 84c171fc-fd79-43f2-942f-7ec6acd63c14
md"""
### Smothing results

|      |     |
| ---- | --- |
| seed | $(seed_smoothing_slider) |
| n    | $(n_smoothing_slider) |
| θ    | $(θ_smoothing_slider) |
"""

# ╔═╡ 8981b0f8-9ff2-4c90-a958-0fe4da538809
x_smoothing_estimated = inference_full_graph(
	y_smoothing, 
	A_smoothing, 
	B_smoothing, 
	P_smoothing, 
	Q_smoothing
);

# ╔═╡ cf1426f1-98d9-402a-80f6-27545fd06d94
begin
	local reshape_data = (data) -> transpose(reduce(hcat, data))
	
	local ylimit = (-20, 20)
	
	local x_label = [ "x[:, 1]" "x[:, 2]" ]
	local y_label = [ "observations[:, 1]" "observations[:, 2]" ]
	local i_label = [ "inferred[:, 1]" "inferred[:, 2]" ]
	
	local x_reshaped = x_smoothing |> reshape_data
	local y_reshaped = y_smoothing |> reshape_data
	
	local x_inferred_means = mean.(x_smoothing_estimated) |> reshape_data
	local x_inferred_stds  = var.(x_smoothing_estimated) |> reshape_data
	
	local p = plot()
	local range = 1:n_smoothing
	
	p = plot!(p, range, x_reshaped, label = x_label)
	p = plot!(p, range, x_inferred_means, ribbon = x_inferred_stds, label = i_label)
	p = scatter!(p, range, y_reshaped, ms = 3, alpha = 0.5, label = y_label)
	p = plot!(p, legend = :bottomleft, ylimit = ylimit)
	
	@saveplot p "lgssm_smoothing_inference"
end

# ╔═╡ 1c6ee7dc-3ecc-43f6-a467-d21ef9c79b34
md"""
From inference results we can see that our model predicted latent states correctly and with high precision.
"""

# ╔═╡ 9d4eecce-37d5-4177-8c52-1ad0da74e7ce
md"""
### Kalman filter

We may perform forward-only inference task which resembles Kalman filter for the same model. To do that it is enough to build a single time step of the model and to redirect posterior marginal updates from the next step to the priors of the previous time step. First, lets define a single time step of the model.
"""

# ╔═╡ 7671e1cc-4ff6-4c2b-b811-aa389a82c6b2
@model function linear_gaussian_ssm_single_time_segment(A, B, P, Q)
    
	cA = constvar(A)
	cB = constvar(B)
	cP = constvar(P)
	cQ = constvar(Q)
	
	# We create a `datavar` placeholders for priors for the previous time step
	# We will later iteratively change our priors based on posterior marginals
	# on each time step
    x_min_t_mean = datavar(Vector{Float64})
    x_min_t_cov  = datavar(Matrix{Float64})
    
    x_min_t ~ MvGaussianMeanCovariance(x_min_t_mean, x_min_t_cov)
    x_t     ~ MvGaussianMeanCovariance(cA * x_min_t, cP)
    
    y_t = datavar(Vector{Float64})
    y_t ~ MvGaussianMeanCovariance(cB * x_t, cQ)
    
    return x_min_t_mean, x_min_t_cov, x_t, y_t
end

# ╔═╡ 3bdd805a-913c-48ac-8fd7-da7ba9ac99bd
md"""
### Inference
"""

# ╔═╡ 9b9b7b02-a8fa-4d67-8ace-bcd30663e312
function inference_single_time_segment(observations, A, B, P, Q)
	
    n = length(observations) 
    
    model, (x_min_t_mean, x_min_t_cov, x_t, y_t) =
		linear_gaussian_ssm_single_time_segment(A, B, P, Q)
    
	# We want to keep a full history of posterior marginal updates 
	# for all time steps
    xbuffer = keep(Marginal)
    
	# Here we redirect our posterior marginal distribution from the current
	# time step to the priors and proceed with the next time step
    redirect_to_prior = subscribe!(getmarginal(x_t), (x_t_posterior) -> begin
        update!(x_min_t_mean, mean(x_t_posterior))
        update!(x_min_t_cov, cov(x_t_posterior))    
    end)
    
    xsubscription = subscribe!(getmarginal(x_t), xbuffer)
    
	# Priors for very first observation
    update!(x_min_t_mean, [ 0.0, 0.0 ])
    update!(x_min_t_cov, [ 100.0 0.0; 0.0 100.0 ])
    
    for observation in observations
        update!(y_t, observation)
    end
    
    unsubscribe!(xsubscription)
    unsubscribe!(redirect_to_prior)
    
    return getvalues(xbuffer)
end

# ╔═╡ b8dea87a-258a-4849-bb12-7e9b5d1420ae
begin 

	# Seed for random number generator for filtering case
	seed_filtering_slider = @bind(
		seed_filtering, 
		ThrottledSlider(1:100, default = 42, show_value = true)
	)
	
	# Number of observations on our model for filtering case
	n_filtering_slider = @bind(
		n_filtering, 
		ThrottledSlider(1:200, default = 100, show_value = true)
	)
	
	# θ parameter is a rotation angle for transition matrix
	θ_filtering_slider = @bind(
		θ_filtering, 
		ThrottledSlider(
			range(0.0, π/2, length = 100), default = π/20, show_value = true
		)
	)
end;

# ╔═╡ 633bbd88-84d3-4d91-a2b6-e5e953171e45
begin
	A_filtering = [ 
		cos(θ_filtering) -sin(θ_filtering); 
		sin(θ_filtering) cos(θ_filtering) 
	]
	B_filtering = [ 1.3 0.0; 0.0 0.7 ]
	P_filtering = [ 1.0 0.0; 0.0 1.0 ]
	Q_filtering = [ 1.0 0.0; 0.0 1.0 ]
end;

# ╔═╡ 0ee27a13-39b5-4f4c-8153-51c132663e2e
md"""
|      |     |
| ---- | --- |
| seed | $(seed_filtering_slider) |
| n    | $(n_filtering_slider) |
| θ    | $(θ_filtering_slider) |
"""

# ╔═╡ ca8a7196-b573-4d72-8706-f0965e0f72d6
x_filtering, y_filtering = generate_data(
	n    = n_filtering, 
	A    = A_filtering, 
	B    = B_filtering, 
	P    = P_filtering, 
	Q    = Q_filtering, 
	seed = seed_filtering
)

# ╔═╡ b006a807-016d-41aa-91ec-a02a4c270990
begin 
	local reshape_data = (data) -> transpose(reduce(hcat, data))
	
	local ylimit = (-20, 20)
	
	local x_label = [ "x[:, 1]" "x[:, 2]" ]
	local y_label = [ "observations[:, 1]" "observations[:, 2]" ]
	
	local x_reshaped = x_filtering |> reshape_data
	local y_reshaped = y_filtering |> reshape_data
	
	local p = plot()
	local range = 1:n_filtering
	
	p = plot!(p, range, x_reshaped, label = x_label)
	p = scatter!(p, range, y_reshaped, ms = 3, alpha = 0.5, label = y_label)
	p = plot!(p, legend = :bottomleft, ylimit = ylimit)
	
	@saveplot p "lgssm_filtering_data"
end

# ╔═╡ 952cce56-c832-47cc-95ec-6c0d114add79
md"""
### Filtering results

|      |     |
| ---- | --- |
| seed | $(seed_filtering_slider) |
| n    | $(n_filtering_slider) |
| θ    | $(θ_filtering_slider) |
"""

# ╔═╡ 989923c9-6871-449f-91eb-d20db563d568
x_filtering_estimated = inference_single_time_segment(
	y_filtering, 
	A_filtering, 
	B_filtering, 
	P_filtering, 
	Q_filtering
);

# ╔═╡ 3beedf02-e870-4da9-89f4-a2667e5bee18
begin
	local reshape_data = (data) -> transpose(reduce(hcat, data))
	
	local ylimit = (-20, 20)
	
	local x_label = [ "x[:, 1]" "x[:, 2]" ]
	local y_label = [ "observations[:, 1]" "observations[:, 2]" ]
	local i_label = [ "inferred[:, 1]" "inferred[:, 2]" ]
	
	local x_reshaped = x_filtering |> reshape_data
	local y_reshaped = y_filtering |> reshape_data
	
	local x_inferred_means = mean.(x_filtering_estimated) |> reshape_data
	local x_inferred_stds  = var.(x_filtering_estimated) |> reshape_data
	
	local p = plot()
	local range = 1:n_filtering
	
	p = plot!(p, range, x_reshaped, label = x_label)
	p = plot!(p, range, x_inferred_means, ribbon = x_inferred_stds, label = i_label)
	p = scatter!(p, range, y_reshaped, ms = 3, alpha = 0.5, label = y_label)
	p = plot!(p, legend = :bottomleft, ylimit = ylimit)
	
	@saveplot p "lgssm_filtering_inference"
end

# ╔═╡ c082bbff-08ce-461e-a096-0df699a6f12d
md"""
Compared to the previous demo (smoothing), the state estimation algorithm in this demo only passes messages forward in time. Therefore, the state estimates are less accurate than the smoothing result of the previous demo.
"""

# ╔═╡ Cell order:
# ╠═d160581c-9d1f-11eb-05f7-c5f29954488b
# ╠═95beaa74-12b5-4bf1-aeb1-a9e726c49cc9
# ╠═6f1ba6c5-9359-40a0-b069-7194dee87e14
# ╠═f1cede2f-0e34-497b-9913-3204e9c75fd7
# ╟─bbb878a0-1854-4bc4-9274-47edc8899795
# ╟─9a8ce058-e7c3-4730-b4bd-b8782cead88f
# ╠═e39f30bf-5b19-4743-9a0e-16cafeed8d13
# ╟─b6c7d8e3-c8c4-4b39-8c53-cbcb5f046b9b
# ╠═dd06f508-d820-4a56-92eb-04d821c1f215
# ╟─210d41a9-a8ff-4c24-9b88-524bed03cd7f
# ╠═3011f498-9319-4dee-ba30-342ae0a2dc07
# ╟─7dcd84fd-c505-4f97-875d-49decba5c3f2
# ╠═ebc733ef-6638-4e42-a007-f2464ce3b5cf
# ╟─2ce93b39-70ea-4b33-b9df-64e6ade6f896
# ╠═b0831de2-2aeb-432b-8987-872f4c5d74f0
# ╟─934ad4d3-bb47-4174-b3d1-cbd6f8e5d75e
# ╟─099aa726-7694-4b38-875d-a015effc9d3a
# ╟─2530cf00-52c1-4c44-8d62-a3e4f0d411bc
# ╠═fb94e6e9-10e4-4f9f-95e6-43cdd9184c09
# ╟─84c171fc-fd79-43f2-942f-7ec6acd63c14
# ╠═8981b0f8-9ff2-4c90-a958-0fe4da538809
# ╟─cf1426f1-98d9-402a-80f6-27545fd06d94
# ╟─1c6ee7dc-3ecc-43f6-a467-d21ef9c79b34
# ╟─9d4eecce-37d5-4177-8c52-1ad0da74e7ce
# ╠═7671e1cc-4ff6-4c2b-b811-aa389a82c6b2
# ╟─3bdd805a-913c-48ac-8fd7-da7ba9ac99bd
# ╠═9b9b7b02-a8fa-4d67-8ace-bcd30663e312
# ╠═b8dea87a-258a-4849-bb12-7e9b5d1420ae
# ╠═633bbd88-84d3-4d91-a2b6-e5e953171e45
# ╟─0ee27a13-39b5-4f4c-8153-51c132663e2e
# ╠═ca8a7196-b573-4d72-8706-f0965e0f72d6
# ╟─b006a807-016d-41aa-91ec-a02a4c270990
# ╟─952cce56-c832-47cc-95ec-6c0d114add79
# ╠═989923c9-6871-449f-91eb-d20db563d568
# ╟─3beedf02-e870-4da9-89f4-a2667e5bee18
# ╟─c082bbff-08ce-461e-a096-0df699a6f12d
