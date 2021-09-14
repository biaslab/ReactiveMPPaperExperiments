### A Pluto.jl notebook ###
# v0.16.0

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
begin 
	using Revise
	using Pkg
end

# ╔═╡ 95beaa74-12b5-4bf1-aeb1-a9e726c49cc9
Pkg.activate("$(@__DIR__)/../") # To disable pluto's built-in pkg manager

# ╔═╡ f1cede2f-0e34-497b-9913-3204e9c75fd7
begin

	using ReactiveMPPaperExperiments
	using DrWatson, PlutoUI, Images
	using CairoMakie
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random
	using BenchmarkTools, DataFrames, Query, LinearAlgebra
	
	import ReactiveMP: update!
end

# ╔═╡ bbb878a0-1854-4bc4-9274-47edc8899795
md"""
#### Linear Multivariate Gaussian State-space Model

In this demo, the goal is to perform both Kalman filtering and smoothing algorithms for a Linear Multivariate Gaussian state-space model (LGSSM).

We wil use the following model:

```math
\begin{equation}
  \begin{aligned}
    p(\mathbf{x}_k|\mathbf{x}_{k - 1}) & = \, \mathcal{N}(\mathbf{x}_k|\mathbf{A}\mathbf{x}_{k - 1}, \mathcal{P}) \\
    p(\mathbf{y}_k|\mathbf{x}_k) & = \, \mathcal{N}(\mathbf{y}_k|\mathbf{B}\mathbf{x}_{k}, \mathcal{Q}) \\
  \end{aligned}
\end{equation}
```

In this model, we denote by $\mathbf{x}_k$ the current state of the system (at time step $k$), by $\mathbf{x}_{k - 1}$ the previous state at time $k-1$, $\mathbf{A}$ and $\mathbf{B}$ are a constant system inputs and $\mathbf{y}_k$ is a noisy observation of $\mathbf{x}_k$. We further assume that the states and the observations are corrupted by i.i.d. Gaussian noise with variances $\mathcal{P}$ and $\mathcal{Q}$ respectively.

The SSM can be represented by the following factor graph, where the pictured section is chained over time:

$(load(projectdir("figures", "ssm_model.png")))

Usually this type of model is used for a linear differential equations where the measured quantities were linear functions of the state. For example this can be the dynamics of the car or noisy pendulum model [Bayesian Filtering and Smoothing, Särkkä, Simo, p.~44].
"""

# ╔═╡ 5fc0ccdf-70e2-46ac-a77e-34f01b885dec
md"""
### Data
"""

# ╔═╡ f1353252-62c4-4ec4-acea-bdfb18c747ae
md"""
For testing purposes we can use synthetically generated data where underlying data generation process matches our model specification.
"""

# ╔═╡ 87d0a5d1-743d-49a7-863e-fb3b795d72f3
function generate_data(; n, A, B, P, Q, seed)
	
	# We create a local RNG to make our results reproducable
    rng = MersenneTwister(seed)

    x_prev = zeros(first(size(A)))

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        x[i] = rand(rng, MvNormal(A * x_prev, P))
        y[i] = rand(rng, MvNormal(B * x[i], Q))

        x_prev = x[i]
    end
   
    return x, y
end

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
	
	d = first(size(A))
	pm = zeros(d)
	pc = Matrix(Diagonal(100.0 * ones(d)))
    
	# Set a prior distribution for x[1]
    x[1] ~ MvGaussianMeanCovariance(pm, pc) 
    y[1] ~ MvGaussianMeanCovariance(cB * x[1], cQ)
    
    for t in 2:n
        x[t] ~ MvGaussianMeanCovariance(cA * x[t - 1], cP)
        y[t] ~ MvGaussianMeanCovariance(cB * x[t], cQ)    
    end
    
    return x, y
end

# ╔═╡ 65e339e3-a90d-47b6-b5f2-b60addc93791
md"""
GraphPPL.jl offers a model specification syntax that resembles closely to the mathematical equations defined above. In this particular implementation we use `MvGaussianMeanCovariance` node for $\mathcal{N}(\mu, \Sigma)$ distribution. ReactiveMP.jl inference backend also supports `MvGaussianMeanPrecision` and `MvGaussianWeightedMeanPrecision` parametrisations for factor nodes. `datavar` placeholders are used to indicate variables that take specific values at a later date. For example, the way we feed observations into the model is by iteratively assigning each of the observations in our dataset to the data variables `y`.
"""

# ╔═╡ 210d41a9-a8ff-4c24-9b88-524bed03cd7f
md"""
### Interactivity

Pluto allows us to interactively explore and experiment with our models. Here we will create a set of sliders for later use. These sliders will allow us to dinamicaly change our model and data generation parameters and see changes immediatelly.
"""

# ╔═╡ 3011f498-9319-4dee-ba30-342ae0a2dc07
begin
	# Seed for random number generator for full graph
	seed_smoothing_slider = @bind(
		seed_smoothing, ThrottledSlider(1:100, default = 42)
	)
	
	# Number of observations on our model for full graph
	n_smoothing_slider = @bind(
		n_smoothing, ThrottledSlider(1:200, default = 50)
	)
	
	# θ parameter is a rotation angle for transition matrix
	θ_smoothing_slider = @bind(
		θ_smoothing, ThrottledSlider(range(0.0, π/2, length = 100), default = π/20)
	)
end;

# ╔═╡ 7dcd84fd-c505-4f97-875d-49decba5c3f2
md"""
Lets first see how our synthetic data look like. We plot an individual line for each dimension in our dataset. To make it uncluttered we generate 2-dimensional observations with rotation matrix as a transition matrix with θ parameter which represents rotation angle
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

# ╔═╡ 5becdba8-d38f-4d75-9c24-6790c73ff48b
let 
	local edim = (d...) -> (x) -> map(e -> e[d...], x)
	
	local ylimit = (-15, 20)	
	local range = 1:n_smoothing
	local c = Makie.wong_colors()
	
	local fig = Figure(resolution = (550, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimit)
	
	# ax.title = "Smoothing synthetic data"
	# ax.titlesize = 20
	
	ax.xlabel = "Time step k"
	ax.xlabelsize = 16
	
	ax.ylabel = "Latent states"
	ax.ylabelsize = 16
	
	lines!(ax, 
		range, x_smoothing |> edim(1), color = :red3, label = "x[:, 1]",
		linewidth = 1.75
	)
	scatter!(ax, 
		range, y_smoothing |> edim(1), color = (:red3, 0.65), 
		markersize = 12, marker = :cross,
		label = "y[:, 1]"
	)
	
	lines!(ax, 
		range, x_smoothing |> edim(2), color = :purple, label = "x[:, 2]",
		linewidth = 1.75, linestyle = :dash
	)
	scatter!(ax, range, 
		y_smoothing |> edim(2), color = (:purple, 0.65), 
		markersize = 8, marker = :circle,
		label = "y[:, 2]"
	)
	
	axislegend(ax, position = :lt)
	
	@saveplot fig "lgssm_smoothing_data"
end

# ╔═╡ 2530cf00-52c1-4c44-8d62-a3e4f0d411bc
md"""
### Inference

Next we need to define our inference procedure. FFG that represents our model has no loops hence we may perform exact Bayesian inference with sum-product algorithm. 

To obtain posterior marginal distributions of our latent state variables we simply use  `subscribe!` function together with `getmarginal` observable. To start inference we pass our observations with `update!` function. Here is a general template for inference function:
"""

# ╔═╡ fb94e6e9-10e4-4f9f-95e6-43cdd9184c09
function inference_full_graph(observations, A, B, P, Q)
	
	# We create a full graph based on how many observations
	# we have in our dataset
    n = length(observations) 
    
	# We call a `linear_gaussian_ssm_full_graph` function 
	# from our model specification above
    model, (x, y) = linear_gaussian_ssm_full_graph(
		n, A, B, P, Q, options = (limit_stack_depth = 500, )
	)
    
	# Rocket.jl provides some handy default actors
	# `buffer` actor simply copies all received updates 
	# into an internal buffer with length n
    xbuffer = buffer(Marginal, n)
	bfe     = ScoreActor(Float64)
    
	# For a collection of random variables we can use 
	# `getmarginals()` function which returns a stream of vectors
    xsubscription = subscribe!(getmarginals(x), xbuffer)
	fsubscription = subscribe!(score(Float64, BetheFreeEnergy(), model), bfe)
    
    update!(y, observations)
    
	# Usually we need to unsubscribe every time we're done with our model
    unsubscribe!(xsubscription)
    
    return map(getvalues, (xbuffer, bfe))
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
x_smoothing_estimated, bfe_smoothing = inference_full_graph(
	y_smoothing, 
	A_smoothing, 
	B_smoothing, 
	P_smoothing, 
	Q_smoothing
);

# ╔═╡ cf1426f1-98d9-402a-80f6-27545fd06d94
let
	local edim = (d...) -> (x) -> map(e -> e[d...], x)
	
	local ylimit = (-15, 20)
	local c = Makie.wong_colors() 
	
	local x_inferred_means = mean.(x_smoothing_estimated)
	local x_inferred_stds  = diag.(std.(x_smoothing_estimated))
	local range = 1:n_smoothing
	
	local fig = Figure(resolution = (550, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimit)
	
	# ax.title = "Smoothing inference results"
	# ax.titlesize = 20
	
	ax.xlabel = "Time step k"
	ax.xlabelsize = 16
	
	ax.ylabel = "Latent states"
	ax.ylabelsize = 16
	
	# Real dim1
	
	lines!(ax, 
		range, x_smoothing |> edim(1), color = :red3, label = "x[:, 1]",
		linewidth = 1.75
	)
	scatter!(ax, 
		range, y_smoothing |> edim(1), color = (:red3, 0.35), 
		markersize = 10, marker = :cross,
	)
	
	# Estimated dim1
	
	lines!(ax,
		range, x_inferred_means |> edim(1), color = c[3], label = "estimated[:, 1]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(1)) .+ (x_inferred_stds |> edim(1)),
		(x_inferred_means |> edim(1)) .- (x_inferred_stds |> edim(1)),
		color = (c[3], 0.65)
	)
	
	
	# Real dim2
	
	lines!(ax, 
		range, x_smoothing |> edim(2), color = :purple, label = "x[:, 2]",
		linewidth = 1.75, linestyle = :dash
	)
	scatter!(ax, range, 
		y_smoothing |> edim(2), color = (:purple, 0.35), 
		markersize = 6, marker = :circle,
	)
	
	# Estimated dim2
	
	lines!(ax,
		range, x_inferred_means |> edim(2), color = c[1], label = "estimated[:, 2]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(2)) .+ (x_inferred_stds |> edim(2)),
		(x_inferred_means |> edim(2)) .- (x_inferred_stds |> edim(2)),
		color = (c[1], 0.65)
	)
	
	axislegend(ax, position = :lt)
	
	@saveplot fig "lgssm_smoothing_inference"
end

# ╔═╡ 1c6ee7dc-3ecc-43f6-a467-d21ef9c79b34
md"""
From inference results we can see that our model predicted latent states correctly and with high precision.
"""

# ╔═╡ 9d4eecce-37d5-4177-8c52-1ad0da74e7ce
md"""
### Kalman filter

We may perform forward-only message-passing scheme which resembles Kalman filter for the same model. To do that it is enough to build a single time step of the model and to redirect posterior marginal updates from the next step to the priors of the previous time step. First, lets define a single time step of the model.
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
	bfe     = ScoreActor(Float64)
    
	# Here we redirect our posterior marginal distribution from the current
	# time step to the priors and proceed with the next time step
    redirect_to_prior = subscribe!(getmarginal(x_t), (x_t_posterior) -> begin
        update!(x_min_t_mean, mean(x_t_posterior))
        update!(x_min_t_cov, cov(x_t_posterior))    
    end)
    
    xsubscription = subscribe!(getmarginal(x_t), xbuffer)
	fsubscription = subscribe!(score(Float64, BetheFreeEnergy(), model), bfe)
    
	d = first(size(A))
	pm = zeros(d)
	pc = Matrix(Diagonal(100.0 * ones(d)))
	
	# Priors for the very first observation
    update!(x_min_t_mean, pm)
    update!(x_min_t_cov, pc)
    
    for observation in observations
        update!(y_t, observation)
		release!(bfe)
    end
    
    unsubscribe!(xsubscription)
    unsubscribe!(redirect_to_prior)
    
    return map(getvalues, (xbuffer, bfe))
end

# ╔═╡ b8dea87a-258a-4849-bb12-7e9b5d1420ae
begin 

	# Seed for random number generator for filtering case
	seed_filtering_slider = @bind(
		seed_filtering, ThrottledSlider(1:100, default = 42)
	)
	
	# Number of observations on our model for filtering case
	n_filtering_slider = @bind(
		n_filtering, ThrottledSlider(1:200, default = 50)
	)
	
	# θ parameter is a rotation angle for transition matrix
	θ_filtering_slider = @bind(
		θ_filtering, ThrottledSlider(range(0.0, π/2, length = 100), default = π/20)
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
let 
	local edim = (d...) -> (x) -> map(e -> e[d...], x)
	
	local ylimit = (-15, 20)	
	local range = 1:n_filtering
	
	local fig = Figure(resolution = (550, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimit)
	
	# ax.title = "Filtering synthetic data"
	# ax.titlesize = 20
	
	ax.xlabel = "Time step k"
	ax.xlabelsize = 16
	
	ax.ylabel = "Latent states"
	ax.ylabelsize = 16
	
	lines!(ax, 
		range, x_filtering |> edim(1), color = :red3, label = "x[:, 1]",
		linewidth = 1.75
	)
	scatter!(ax, 
		range, y_filtering |> edim(1), color = (:red3, 0.65), 
		markersize = 12, marker = :cross,
		label = "y[:, 1]"
	)
	
	lines!(ax, 
		range, x_filtering |> edim(2), color = :purple, label = "x[:, 2]",
		linewidth = 1.75, linestyle = :dash
	)
	scatter!(ax, range, 
		y_filtering |> edim(2), color = (:purple, 0.65), 
		markersize = 8, marker = :circle,
		label = "y[:, 2]"
	)
	
	axislegend(ax, position = :lt)
	
	@saveplot fig "lgssm_filtering_data"
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
x_filtering_estimated, bfe_filtering = inference_single_time_segment(
	y_filtering, 
	A_filtering, 
	B_filtering, 
	P_filtering, 
	Q_filtering
);

# ╔═╡ 3beedf02-e870-4da9-89f4-a2667e5bee18
let
	local edim = (d...) -> (x) -> map(e -> e[d...], x)
	
	local ylimit = (-15, 20)
	
	local x_inferred_means = mean.(x_filtering_estimated)
	local x_inferred_stds  = diag.(std.(x_filtering_estimated))
	local range = 1:n_filtering
	local c = Makie.wong_colors()
	
	local fig = Figure(resolution = (550, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimit)
	
	# ax.title = "Filtering inference results"
	# ax.titlesize = 20
	
	ax.xlabel = "Time step k"
	ax.xlabelsize = 16
	
	ax.ylabel = "Latent states"
	ax.ylabelsize = 16
	
	# Real dim1
	
	lines!(ax, 
		range, x_filtering |> edim(1), color = :red3, label = "x[:, 1]",
		linewidth = 1.75
	)
	scatter!(ax, 
		range, y_filtering |> edim(1), color = (:red3, 0.35), 
		markersize = 10, marker = :cross,
	)
	
	# Estimated dim1
	
	lines!(ax,
		range, x_inferred_means |> edim(1), color = c[3], label = "estimated[:, 1]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(1)) .+ (x_inferred_stds |> edim(1)),
		(x_inferred_means |> edim(1)) .- (x_inferred_stds |> edim(1)),
		color = (c[3], 0.65)
	)
	
	
	# Real dim2
	
	lines!(ax, 
		range, x_filtering |> edim(2), color = :purple, label = "x[:, 2]",
		linewidth = 1.75, linestyle = :dash
	)
	scatter!(ax, range, 
		y_filtering |> edim(2), color = (:purple, 0.35), 
		markersize = 6, marker = :circle,
	)
	
	# Estimated dim2
	
	lines!(ax,
		range, x_inferred_means |> edim(2), color = c[1], label = "estimated[:, 2]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(2)) .+ (x_inferred_stds |> edim(2)),
		(x_inferred_means |> edim(2)) .- (x_inferred_stds |> edim(2)),
		color = (c[1], 0.65)
	)
	
	axislegend(ax, position = :lt, labelsize = 16)
	
	@saveplot fig "lgssm_filtering_inference"
end

# ╔═╡ c082bbff-08ce-461e-a096-0df699a6f12d
md"""
Compared to the previous demo (smoothing), the state estimation algorithm in this demo only passes messages forward in time. Therefore, the state estimates are less accurate than the smoothing result of the previous demo.
"""

# ╔═╡ 4d3f9bdd-f89d-4fc4-924d-55008cd2c979
md"""
### Benchmarking

In this section we will benchmark inference performance with the help of BenchmarkTools package. `ReactiveMP.jl` has been designed to be efficient and scalable as much as possible. To show `ReactiveMP.jl` performance capabilities we run a series of benchmark tests for linear gaussian state space model with different number of observations and different seeds and θ parameters. We show that execution time scales linearly on number of observations and does not depend on seed or θ. 
"""

# ╔═╡ 895e61c4-c8ba-4bb8-9c04-be6cea67d5eb
function random_posdef_matrix(rng, dimension)
	L = rand(rng, dimension, dimension)
	return L' * L
end

# ╔═╡ 10965ed2-da40-4b6c-80a8-2f84590803a8
function run_benchmark(inference_fn::Function, params)
	@unpack n, d, seed = params
	
	rng = MersenneTwister(seed)
	
	A = random_posdef_matrix(rng, d)
	B = random_posdef_matrix(rng, d)
	P = Matrix(Diagonal(ones(d)))
	Q = Matrix(Diagonal(ones(d)))
	
	x, y = generate_data(
		n    = n, 
		A    = A, 
		B    = B, 
		P    = P, 
		Q    = Q, 
		seed = seed
	)
	
	x_estimated, bfe = inference_fn(y, A, B, P, Q);
	benchmark        = @benchmark $inference_fn($y, $A, $B, $P, $Q)
	
	output = @strdict n d seed x_estimated bfe benchmark
	
	return output
end

# ╔═╡ fe122f6a-a30d-4fad-abf5-2c62cac09836
# Here we create a list of parameters we want to run our benchmarks with
benchmark_allparams = dict_list(Dict(
	"n"    => [ 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000 ],
	"d"    => [ 3, 10, 30 ],	
	"seed" => [ 42 ]
));

# ╔═╡ f8b90eeb-719f-456d-9fe5-d84fca13c65c
md"""
We want to perform benchmarking for different sets of parameters. In principal, timings should not depend on seed and θ parameters, but only on n.
"""

# ╔═╡ 425cccc7-6a38-477b-a128-7a513b056e4c
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create new benchmarks 
# but will reload it from data folder
smoothing_benchmarks = map(benchmark_allparams) do params
	path = datadir("benchmark", "lgssm", "smoothing")
	result, _ = produce_or_load(path, params; tag = false) do p
		run_benchmark(inference_full_graph, p)
	end
	return result
end;

# ╔═╡ d38bc0a0-3fde-4d71-aa4c-11a83532fc05
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create aew benchmarks 
# but will reload it from data folder
filtering_benchmarks = map(benchmark_allparams) do params
	path = datadir("benchmark", "lgssm", "filtering")
	result, _ = produce_or_load(path, params; tag = false) do p
		run_benchmark(inference_single_time_segment, p)
	end
	return result
end;

# ╔═╡ f9491325-a06a-47ca-af5a-b55077596730
begin 
	target_seed = 42
	target_d = 3
end;

# ╔═╡ 5b0c1a35-b06f-43d4-b255-a1ab045de83c
md"""
Here we extract benchmarking results in a `DataFrame` table for seed = $(target_seed) and d = $(target_d)
"""

# ╔═╡ 3bf227ef-5390-4002-8285-5cabd8a50ec5
begin
	local path_filtering = datadir("benchmark", "lgssm", "filtering")
	local path_smoothing = datadir("benchmark", "lgssm", "smoothing")
	
	local white_list   = [ "n", "seed", "d" ]
	local special_list = [
		:min => (data) -> string(
			round(minimum(data["benchmark"]).time / 1_000_000, digits = 2), "ms"
		),
		:mean => (data) -> string(
			round(mean(data["benchmark"]).time / 1_000_000, digits = 2), "ms"
		)
	]
	
	local df_filtering = collect_results(path_filtering, 
		white_list = white_list,
		special_list = special_list
	)
	
	local df_smoothing = collect_results(path_smoothing, 
		white_list = white_list,
		special_list = special_list
	)
	
	local query_filtering = @from row in df_filtering begin
		@where row.seed == target_seed && row.d == target_d
		@orderby ascending(row.n)
		@select { row.n, row.min, row.mean }
	end
	
	local query_smoothing = @from row in df_smoothing begin
		@where row.seed == target_seed && row.d == target_d
		@orderby ascending(row.n)
		@select { row.n, row.min, row.mean }
	end
	
	local res_filtering = DataFrame(query_filtering)
	local res_smoothing = DataFrame(query_smoothing)
	
	
	local df = rightjoin(res_filtering, res_smoothing, on = :n, makeunique = true)
	
	df = rename(df, 
		:min => :min_filtering, :mean => :mean_filtering,
		:min_1 => :min_smoothing, :mean_1 => :mean_smoothing
	)
end

# ╔═╡ 4871b82d-15db-4c80-9b95-7f2c8086b864
md"""
We can see from benchmark results execution time scales linearly with the number of observations. This is in line with our expectations because the complexity of both filtering and smoothing algorithms are linear on number of observations.
"""

# ╔═╡ 653dafb5-173d-40f7-93b3-ae4fbfb5d0d6
md"""
Lets also plot benchmark timings for both smoothing and filtering algorithms against number of observation. Notice that we use a log scale axis to have a linear-like plot.
"""

# ╔═╡ d485a985-39ec-422b-9476-d55b98786093
let
	local s_filtered = filter(smoothing_benchmarks) do b
		return b["d"] === target_d && b["seed"] === target_seed
	end
	
	local f_filtered = filter(filtering_benchmarks) do b
		return b["d"] === target_d && b["seed"] === target_seed
	end
	
	@assert length(s_filtered) !== 0 "Empty benchmark set"
	@assert length(f_filtered) !== 0 "Empty benchmark set"
	
	local s_range      = map(f -> f["n"], s_filtered)
	local s_benchmarks = map(f -> f["benchmark"], s_filtered)
	local s_timings    = map(t -> t.time, minimum.(s_benchmarks)) ./ 1_000_000
	local s_memories   = map(t -> t.memory, minimum.(s_benchmarks)) ./ 1024
	
	local f_range      = map(f -> f["n"], f_filtered)
	local f_benchmarks = map(f -> f["benchmark"], f_filtered)
	local f_timings    = map(t -> t.time, minimum.(f_benchmarks)) ./ 1_000_000
	local f_memories   = map(t -> t.memory, minimum.(f_benchmarks)) ./ 1024
	
	fig = Figure(resolution = (500, 300))
	
	ax = Makie.Axis(fig[1, 1])
	
	ax.xlabel = "Number of observartions in static dataset (log-scale)"
	ax.xlabelsize = 16
	ax.ylabel = "Time (in ms, log-scale)"
	ax.ylabelsize = 16
	ax.yscale = Makie.pseudolog10
	ax.xscale = Makie.pseudolog10
	
	ax.yticks = (
		[ 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000 ], 
		[ "3", "5", "10", "20", "50", "100", "200", "500", "1e3", "2e3" ]
	)
	
	ax.xticks = (
		[ 50, 100, 250, 500, 1000, 2000, 5000, 10000 ], 
		[ "50", "100", "250", "500", "1e3", "2e3", "5e3", "1e4" ]
	)
	
	lines!(ax,
		s_range, s_timings, label = "Smoothing", linewidth = 3
	)
	scatter!(ax,
		s_range, s_timings, marker = :utriangle, markersize = 16,
	)
	
	lines!(ax,
		f_range, f_timings, label = "Filtering", linewidth = 3,
	)
	scatter!(ax,
		f_range, f_timings, marker = :diamond, markersize = 16,
	)
	
	axislegend(ax, position = :lt, labelsize = 16)
	
	@saveplot fig "lgssm_benchmark"
end

# ╔═╡ aa64496d-a65c-4b38-88b6-b5f9f14c447d
md"""
### Comparison with Turing.jl

In this section we want to compare results and performance of ReactiveMP.jl with another probabilistic programming library which is called Turing.jl. Turing is a general probabilistic programming toolbox and does not use message passing for inference procedure, but sampling. Message passing has an advantage over sampling approach for conjugate models (which our linear gaussian state space model is) because it may fallback to analytically tractable update rules, where sampling cannot. 
"""

# ╔═╡ 0bd929eb-0a4d-4a09-82f6-e87d992cd98b
import Turing

# ╔═╡ f380961f-1d2b-4032-86b8-721d2ba821b6
Turing.@model LinearGaussianSSM(y, A, B, P, Q) = begin
    n = length(y)

    # State sequence.
    x = Vector(undef, n)
	
	d = first(size(A))
	pm = zeros(d)
	pc = Matrix(Diagonal(100.0 * ones(d)))

    # Observe each point of the input.
    x[1] ~ MvNormal(pm, pc)
    y[1] ~ MvNormal(B * x[1], Q)

    for t in 2:n
        x[t] ~ MvNormal(A * x[t - 1], P)
        y[t] ~ MvNormal(B * x[t], Q)
    end
end;

# ╔═╡ 1fbb51a2-c8f6-4c4b-9fd0-f42884a091c6
begin
	seed_turing = 42
	n_turing    = 50
	θ_turing    = π / 20
	
	A_turing = [ 
		cos(θ_turing) -sin(θ_turing); 
		sin(θ_turing) cos(θ_turing) 
	]
	B_turing = [ 1.3 0.0; 0.0 0.7 ]
	P_turing = [ 1.0 0.0; 0.0 1.0 ]
	Q_turing = [ 1.0 0.0; 0.0 1.0 ]
	
	x_turing, y_turing = generate_data(
		n    = n_turing, 
		A    = A_turing, 
		B    = B_turing, 
		P    = P_turing, 
		Q    = Q_turing, 
		seed = seed_turing
	)
end;

# ╔═╡ 1fd698ae-d6a3-4b36-b1e5-80cb5fcd0771
md"""
We use Turing's builtin HMC sampler to perform inference on this model because we want to obtain minus log\_density as a metric for our model.
"""

# ╔═╡ c0b48a41-703d-4d50-bbb3-545117189c9f
function inference_turing(observations, A, B, P, Q; nsamples = 250, seed = 42)
	rng = MersenneTwister(seed)
    return Turing.sample(rng, 
		LinearGaussianSSM(observations, A, B, P, Q), Turing.HMC(0.1, 10), 
		nsamples
	)
end

# ╔═╡ 753d9f01-02bd-4a0b-ac69-77218ac56530
x_turing_estimated = inference_turing(
	y_turing, A_turing, B_turing, P_turing, Q_turing
);

# ╔═╡ 43649fce-8ee4-42a9-819b-ba17fa9de998
let
	local edim = (d...) -> (x) -> map(e -> e[d...], x)
	local reshape_data = (data) -> transpose(reduce(hcat, data))
	local reshape_turing_data = (data) -> transpose(
		reshape(data, (2, Int(length(data) / 2)))
	)
	
	local ylimit = (-15, 20)
	local c = Makie.wong_colors()
	
	local samples = get(x_turing_estimated, :x)
	local x_inferred_means = reshape_turing_data(
		[ mean(samples.x[i].data) for i in 1:2n_turing ]
	) |> collect |> eachrow |> collect
	local x_inferred_stds = reshape_turing_data(
		[std(samples.x[i].data) for i in 1:2n_turing]
	) |> collect |> eachrow |> collect
	
	local range = 1:n_turing
	
	local fig = Figure(resolution = (550, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimit)
	
	# ax.title = "Filtering inference results"
	# ax.titlesize = 20
	
	ax.xlabel = "Time step k"
	ax.xlabelsize = 16
	
	ax.ylabel = "Latent states"
	ax.ylabelsize = 16
	
	# Real dim1
	
	lines!(ax, 
		range, x_turing |> edim(1), color = :red3, label = "x[:, 1]",
		linewidth = 1.75
	)
	scatter!(ax, 
		range, y_turing |> edim(1), color = (:red3, 0.35), 
		markersize = 10, marker = :cross,
	)
	
	# Estimated dim1
	
	lines!(ax,
		range, x_inferred_means |> edim(1), color = c[3], label = "estimated[:, 1]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(1)) .+ (x_inferred_stds |> edim(1)),
		(x_inferred_means |> edim(1)) .- (x_inferred_stds |> edim(1)),
		color = (c[3], 0.65)
	)
	
	
	# Real dim2
	
	lines!(ax, 
		range, x_turing |> edim(2), color = :purple, label = "x[:, 2]",
		linewidth = 1.75, linestyle = :dash
	)
	scatter!(ax, range, 
		y_turing |> edim(2), color = (:purple, 0.35), 
		markersize = 6, marker = :circle,
	)
	
	# Estimated dim2
	
	lines!(ax,
		range, x_inferred_means |> edim(2), color = c[1], label = "estimated[:, 2]"
	)
	band!(ax,
		range, 
		(x_inferred_means |> edim(2)) .+ (x_inferred_stds |> edim(2)),
		(x_inferred_means |> edim(2)) .- (x_inferred_stds |> edim(2)),
		color = (c[1], 0.65)
	)
	
	axislegend(ax, position = :lt, labelsize = 16)
	
	@saveplot fig "lgssm_turing_inference"
end

# ╔═╡ 1751a9f2-2a5f-4cd7-8f0d-d5a3cae5518f
function run_turing_benchmark(params)
	@unpack n, seed, d, nsamples = params
	
	rng = MersenneTwister(seed)
	
	A = random_posdef_matrix(rng, d)
	B = random_posdef_matrix(rng, d)
	P = Matrix(Diagonal(ones(d)))
	Q = Matrix(Diagonal(ones(d)))
	
	x, y = generate_data(
		n    = n, 
		A    = A, 
		B    = B, 
		P    = P, 
		Q    = Q, 
		seed = seed
	)
	
	x_estimated = inference_turing(y, A, B, P, Q, nsamples = nsamples);
	benchmark   = @benchmark inference_turing(
		$y, $A, $B, $P, $Q, nsamples = $nsamples
	)
	
	output = @strdict n seed d nsamples x_estimated benchmark
	
	return output
end

# ╔═╡ c41e8630-767a-4072-80a7-5ef8c5ecc4e0
# Here we create a list of parameters we want to run our benchmarks with
benchmark_allparams_turing = dict_list(Dict(
	"n"        => [ 50, 100, 250 ], # 500, 1000
	"seed"     => 42,
	"d"        => [ 3, 10, 30 ],
	"nsamples" => [ 
		250, 500, @onlyif("n" <= 250, 1000) 
	]
));

# ╔═╡ f9436160-034a-455d-8489-ba80107932f7
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create new benchmarks 
# but will reload it from data folder
turing_benchmarks = map(benchmark_allparams_turing) do params
	path = datadir("benchmark", "lgssm", "turing")
	result, _ = produce_or_load(path, params) do p
		run_turing_benchmark(p)
	end
	return result
end;

# ╔═╡ d6253937-b32e-4c6d-a0a1-8eb35afe92c5
md"""
Here we compare `Turing.jl` performance results against smoothing algorithm performed in `ReactiveMP.jl`. We can see that `ReactiveMP.jl` outperforms `Turing.jl` significantly. It is worth noting that this model contains many conjugate prior and likelihood pairings that lead to analytically computable Bayesian posteriors. For these types of models, ReactiveMP.jl takes advantage of the conjugate pairings and beats general-purpose probabilistic programming packages like Turing.jl easily in terms of computational load, speed, memory and accuracy. On the other hand, Turing.jl is currently still capable of running inference for a broader set of models.
"""

# ╔═╡ 97855300-b472-404c-b5f7-daa5f908e34b
begin
	local path_turing    = datadir("benchmark", "lgssm", "turing")
	local path_smoothing = datadir("benchmark", "lgssm", "smoothing")
	
	local white_list   = [ "n", "seed", "θ" ]
	local special_list = [
		:min => (data) -> string(
			round(minimum(data["benchmark"]).time / 1_000_000_000, digits = 3), "s"
		),
		:mean => (data) -> string(
			round(mean(data["benchmark"]).time / 1_000_000_000, digits = 3), "s"
		),
	]
	
	local df_turing = collect_results(path_turing, 
		white_list = [ white_list... , "nsamples" ],
		special_list = special_list
	)
	
	local df_smoothing = collect_results(path_smoothing, 
		white_list = white_list,
		special_list = special_list
	)
	
	local query_turing = @from row in df_turing begin
		@where row.seed == target_seed && row.θ == target_θ && row.nsamples == 1000
		@orderby ascending(row.n)
		@select { row.n, row.min, row.mean }
	end
	
	local query_smoothing = @from row in df_smoothing begin
		@where row.seed == target_seed && row.θ == target_θ
		@orderby ascending(row.n)
		@select { row.n, row.min, row.mean }
	end
	
	local res_turing    = DataFrame(query_turing)
	local res_smoothing = DataFrame(query_smoothing)
	
	
	local df = rightjoin(res_turing, res_smoothing, on = :n, makeunique = true)
	
	df = rename(df, 
		:min => :min_turing, :mean => :mean_turing,
		:min_1 => :min_smoothing, :mean_1 => :mean_smoothing
	) |> @dropna() |> DataFrame
end

# ╔═╡ 6b572117-9b58-41c5-a507-4f9e38de9db9
let
	local s_filtered = filter(smoothing_benchmarks) do b
		return b["θ"] === target_θ && b["seed"] === target_seed
	end
	
	local t_filtered = filter(turing_benchmarks) do b
		return b["θ"] === target_θ && 
			b["seed"] === target_seed && 
			b["nsamples"] === 500
	end
	
	local ylimits = (0, 5e10)
	
	@assert length(s_filtered) !== 0 "Empty benchmark set"
	@assert length(t_filtered) !== 0 "Empty benchmark set"
	
	local s_range      = map(f -> f["n"], s_filtered)
	local s_benchmarks = map(f -> f["benchmark"], s_filtered)
	local s_timings    = map(t -> t.time, minimum.(s_benchmarks)) ./ 1_000_000
	local s_stds       = map(t -> t.time, std.(s_benchmarks)) ./ 1_000_000
	local s_memories   = map(t -> t.memory, minimum.(s_benchmarks)) ./ 1024
	
	local t_range      = map(f -> f["n"], t_filtered)
	local t_benchmarks = map(f -> f["benchmark"], t_filtered)
	local t_timings    = map(t -> t.time, minimum.(t_benchmarks)) ./ 1_000_000
	local t_stds       = map(t -> t.time, std.(t_benchmarks)) ./ 1_000_000
	local t_memories   = map(t -> t.memory, minimum.(t_benchmarks)) ./ 1024

	local fig = Figure(resolution = (500, 300))
	
	local ax = Makie.Axis(fig[1, 1])
	
	ylims!(ax, ylimits)
	
	ax.xlabel = "Number of observations in dataset (log-scale)"
	ax.xlabelsize = 16
	
	ax.ylabel = "Time (in ms, log-scale)"
	ax.ylabelsize = 16
	
	ax.xscale = Makie.pseudolog10
	ax.yscale = Makie.pseudolog10
	
	ax.yticks = (
		[ 10, 100, 1000, 10000, 100_000, 1_000_000, 1e7, 1e8, 1e9 ], 
		[ "10", "100", "1e3", "1e4", "1e5", "1e6", "1e7", "1e8", "1e9" ]
	)
	
	ax.xticks = (
		[ 50, 100, 250, 500, 1000, 2000, 5000, 10000 ], 
		[ "50", "100", "250", "500", "1e3", "2e3", "5e3", "1e4" ]
	)
	
	lines!(ax, s_range, s_timings, linewidth = 3, label = "Smoothing ReactiveMP")
	scatter!(ax, s_range, s_timings, marker = :utriangle, markersize = 16)
	# errorbars!(ax, s_range, s_timings, s_stds, whiskerwidth = 12, color = :orangered)
	
	lines!(ax, t_range, t_timings, linewidth = 3, label = "HMC Turing")
	scatter!(ax, t_range, t_timings, marker = :utriangle, markersize = 16)
	# errorbars!(ax, t_range, t_timings, t_stds, whiskerwidth = 12, color = :orangered)
	
	axislegend(ax, position = :lt, labelsize = 14)
	
	@saveplot fig "lgssm_benchmark_turing"
end

# ╔═╡ Cell order:
# ╠═d160581c-9d1f-11eb-05f7-c5f29954488b
# ╠═95beaa74-12b5-4bf1-aeb1-a9e726c49cc9
# ╠═f1cede2f-0e34-497b-9913-3204e9c75fd7
# ╟─bbb878a0-1854-4bc4-9274-47edc8899795
# ╟─5fc0ccdf-70e2-46ac-a77e-34f01b885dec
# ╟─f1353252-62c4-4ec4-acea-bdfb18c747ae
# ╠═87d0a5d1-743d-49a7-863e-fb3b795d72f3
# ╟─9a8ce058-e7c3-4730-b4bd-b8782cead88f
# ╠═e39f30bf-5b19-4743-9a0e-16cafeed8d13
# ╟─65e339e3-a90d-47b6-b5f2-b60addc93791
# ╟─210d41a9-a8ff-4c24-9b88-524bed03cd7f
# ╠═3011f498-9319-4dee-ba30-342ae0a2dc07
# ╟─7dcd84fd-c505-4f97-875d-49decba5c3f2
# ╠═ebc733ef-6638-4e42-a007-f2464ce3b5cf
# ╟─2ce93b39-70ea-4b33-b9df-64e6ade6f896
# ╠═b0831de2-2aeb-432b-8987-872f4c5d74f0
# ╟─934ad4d3-bb47-4174-b3d1-cbd6f8e5d75e
# ╟─5becdba8-d38f-4d75-9c24-6790c73ff48b
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
# ╟─4d3f9bdd-f89d-4fc4-924d-55008cd2c979
# ╠═895e61c4-c8ba-4bb8-9c04-be6cea67d5eb
# ╠═10965ed2-da40-4b6c-80a8-2f84590803a8
# ╠═fe122f6a-a30d-4fad-abf5-2c62cac09836
# ╟─f8b90eeb-719f-456d-9fe5-d84fca13c65c
# ╠═425cccc7-6a38-477b-a128-7a513b056e4c
# ╠═d38bc0a0-3fde-4d71-aa4c-11a83532fc05
# ╠═f9491325-a06a-47ca-af5a-b55077596730
# ╟─5b0c1a35-b06f-43d4-b255-a1ab045de83c
# ╟─3bf227ef-5390-4002-8285-5cabd8a50ec5
# ╟─4871b82d-15db-4c80-9b95-7f2c8086b864
# ╟─653dafb5-173d-40f7-93b3-ae4fbfb5d0d6
# ╟─d485a985-39ec-422b-9476-d55b98786093
# ╟─aa64496d-a65c-4b38-88b6-b5f9f14c447d
# ╠═0bd929eb-0a4d-4a09-82f6-e87d992cd98b
# ╠═f380961f-1d2b-4032-86b8-721d2ba821b6
# ╠═1fbb51a2-c8f6-4c4b-9fd0-f42884a091c6
# ╟─1fd698ae-d6a3-4b36-b1e5-80cb5fcd0771
# ╠═c0b48a41-703d-4d50-bbb3-545117189c9f
# ╠═753d9f01-02bd-4a0b-ac69-77218ac56530
# ╟─43649fce-8ee4-42a9-819b-ba17fa9de998
# ╠═1751a9f2-2a5f-4cd7-8f0d-d5a3cae5518f
# ╠═c41e8630-767a-4072-80a7-5ef8c5ecc4e0
# ╠═f9436160-034a-455d-8489-ba80107932f7
# ╟─d6253937-b32e-4c6d-a0a1-8eb35afe92c5
# ╟─97855300-b472-404c-b5f7-daa5f908e34b
# ╟─6b572117-9b58-41c5-a507-4f9e38de9db9
