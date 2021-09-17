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

# ╔═╡ a4affbe0-9d3d-11eb-1ca5-059daf3d3141
begin 
	using Revise
	using Pkg
end

# ╔═╡ f1b341d4-d730-4f98-89d8-3337e6bc0ce1
Pkg.activate("$(@__DIR__)/../") # To disable pluto's built-in pkg manager

# ╔═╡ 4a6a323b-4424-441c-9807-6707ea7ba410
begin

	using ReactiveMPPaperExperiments
	using DrWatson, PlutoUI, Images
	using CairoMakie
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random
	using BenchmarkTools, DataFrames, Query, LinearAlgebra

	import ReactiveMP: update!
end

# ╔═╡ 83eb01c5-e383-4c8d-b187-12fc9dc87ecb
md"""
### Hidden Markov Model

In this demo the goal is to perform approximate variational Bayesian Inference for Hidden Markov Model (HMM). Hidden Markov Model can be viewed as a specific instance of the state space mode in which latent variables are discrete.

The HMM is widely used in speech recognition (Jelinek, 1997; Rabiner and Juang, 1993), natural language modelling (Mannina and Schütze, 1999) and in may other related fields.

We consider a first-order HMM with latent states $s_k$ and observations $x_k$ governed by a state transition probability matrix $\mathbf{A}$ and an observation matrix $\mathbf{B}$:

```math
\begin{equation}
  \begin{aligned}
    p(\mathbf{s}_k|\mathbf{s}_{k-1}) & = \, \text{Cat}(\mathbf{s}_k|\mathbf{A}\mathbf{s}_{k - 1})\\
    p(\mathbf{x}_k|\mathbf{s}_k) & = \, \text{Cat}(\mathbf{x}_k|\mathbf{B}\mathbf{s}_{k})\\
  \end{aligned}
\end{equation}
```

where $\text{Cat}$ denotes Categorical distribution. It is convenient to use one-hot encoding scheme for latent variables $s_k$ and model it with a categorical distribution $\text{Cat}(s|p)$ where p is a vector of probabilities of each possible state. Because the latent variables are $K$-dimensional binary variables, this conditional distribution corresponds to a table of numbers that we denote by $\bf{A}$, the elements of which are known as *transition probabilities*. They are given by 

```math
A_{jk} \equiv p(z_{nk} = 1|z_{n - 1, j} = 1),\: 0 \leq A_{jk} \leq 1,\:\sum_kA_{jk}=1
```

Also, we denote by $\mathbf{s}_k$ the current state of the system (at time step $k$), by $\mathbf{s}_{k - 1}$ the previous state at time $k-1$. For simplicity we assume $\mathbf{A}$ and $\mathbf{B}$ are a constant system inputs and $\mathbf{x}_k$ are noise-free observations.

To have a full Bayesian treatmeant of the problem, both $\mathbf{A}$ and $\mathbf{B}$ are endowed with `MatrixDirichlet` priors. `MatrixDirichlet` is a matrix-variate generalisation of `Dirichlet` distribution and consists of `Dirichlet` distributions on the columns.

We will build a full graph for this model and perform VMP iterations during an inference procedure.
"""

# ╔═╡ df0529bb-70fb-43b2-954d-838fe2165b76
md"""
### Model specification
"""

# ╔═╡ 5c14ef6a-9a9a-4fb6-9e11-90a61b878866
@model [ default_factorisation = MeanField() ] function hidden_markov_model(n)
    
	# A and B are unknown and are random variables with predefined priors
    A ~ MatrixDirichlet(ones(3, 3))
    B ~ MatrixDirichlet([ 10.0 1.0 1.0; 1.0 10.0 1.0; 1.0 1.0 10.0 ])
    
	# We create a vector of random variables for our latent states `s`
    s = randomvar(n)
	
	# We create a vector of observation placeholders with `datavar()` function
    x = datavar(Vector{Float64}, n)
	
	s[1] ~ Categorical(fill(1.0 / 3.0, 3))
	x[1] ~ Transition(s[1], B)
    
    for t in 2:n
		# `where` syntax allows us to pass extra arguments 
		# or a node creation procedure
		# In this example we create a structured posterior factorisation
		# around latent states transition node
        s[t] ~ Transition(s[t - 1], A) where { q = q(out, in)q(a) }
        x[t] ~ Transition(s[t], B)
    end
    
    return s, x, A, B
end

# ╔═╡ 927a5ad7-7ac0-4e8a-a755-d1223093b992
md"""
GraphPPL.jl offers a model specification syntax that resembles closely to the mathematical equations defined above. We use `Transition(s|x, A)` distribution as an alias for `Cat(s|A*x)` distribution, `datavar` placeholders are used to indicate variables that take specific values at a later date. For example, the way we feed observations into the model is by iteratively assigning each of the observations in our dataset to the data variables `x`.

In our model specification we assume a relatively strong prior on the observarion probability matrix, expressing the prior knowledge that we are most likely to observer the true state. On the other hand, we assumme an unifnromative prior on the state transition matrix, expressing no prior knowledge about transition probabilities and structure.
"""

# ╔═╡ c01151b2-5476-4170-971c-518019e891f8
md"""
### Data

We will simulate hidden markov model process by sampling $n$ values from corresponding distributions with fixed A and B matrices. We also use a `seed` parameter to make our experiments reproducible.
"""

# ╔═╡ 9aff5f73-0e50-4ed3-ac35-355afa0d4137
function rand_vec(distribution::Categorical) 
    k = ncategories(distribution)
    s = zeros(k)
    s[ rand(distribution) ] = 1.0
    s
end

# ╔═╡ 9e84c744-5157-4e7c-b910-481f8b6e086c
function normalise(a)
	return a ./ sum(a)
end

# ╔═╡ 30f46dae-6665-4bbc-9451-ee6744c5a6aa
begin
	# Transition probabilities (some transitions are impossible)
    A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9]
    # Observation noise
    B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9] 
end

# ╔═╡ 34ef9070-dc89-4a56-8914-b3c8bd3288ba
function generate_data(n_samples, A, B; seed = 124)
    Random.seed!(seed)
    
    # Initial state
    s_0 = [1.0, 0.0, 0.0] 
	
	# one-hot encoding of the states and of the observations
    s = Vector{Vector{Float64}}(undef, n_samples) 
    x = Vector{Vector{Float64}}(undef, n_samples)
    
    s_prev = s_0
    
    for t in 1:n_samples
        s[t] = rand_vec(Categorical(normalise(A * s_prev)))
        x[t] = rand_vec(Categorical(normalise(B * s[t])))
        s_prev = s[t]
    end
    
    return x, s
end

# ╔═╡ d02a557f-2ecb-4016-bd36-579c7e8bb012
md"""
### Interactivity 

Pluto notebooks allow us to dynamically change some parameters and arguments for our experiments and to immediatelly see the results. We will create a set of sliders which we may want to use later.
"""

# ╔═╡ 1d23082e-ab18-4ca6-9383-aaebddb29f00
begin
	seed_slider = @bind(seed, ThrottledSlider(1:100, default = 54))
	n_slider    = @bind(n, ThrottledSlider(2:1000, default = 50))
end;

# ╔═╡ f28c42fc-1e8a-4e3b-a0cf-73da0c7875cd
md"""
|             |                 |
| ----------- | ----------------|
| seed        | $(seed_slider)  |
| n           | $(n_slider)     |
"""

# ╔═╡ dd0e02b4-e5c7-46eb-8a80-060d22e640b9
md"""
### Inference

Once we have defined our model and generated some synthetic data, the next step is to use ReactiveMP.jl API to run a reactive message-passing algorithm that solves our given inference problem. To do this, we need to specify which variables we are interested in. We obtain a posterior marginal updates stream by calling `getmarginal()` function and pass `A` as an argument as an example. We can also use `getmarginals()` function to obtain a stream of updates over a collection of random variables.

We use `subscribe!` function from `Rocket.jl` to subscribe on posterior marginal updates and to store them in a local buffers. 

In this model we are also interested in Bethe Free Energy functional evaluation. To get and to subscribe on a stream of free energy values over iterations we use `score()` function.

To pass observations to our model we use `update!` function. Inference engine will wait for all observations in the model and will react as soon as all of them have been passed. 
"""

# ╔═╡ 3866d103-7f73-4ff1-8949-20ab0cfd3704
function inference(observations, n_its)
	
	# We create a full graph for this model based on number of observations
    n = length(observations)
    
    model, (s, x, A, B) = hidden_markov_model(
		n, options = (limit_stack_depth = 500, )
	)
    
	# Preallocated buffers for posterior marginals updates
    sbuffer = keep(Vector{Marginal})
    Abuffer = keep(Marginal)
    Bbuffer = keep(Marginal)
    fe      = ScoreActor(Float64)

    ssub  = subscribe!(getmarginals(s), sbuffer)
    Asub  = subscribe!(getmarginal(A), Abuffer)
    Bsub  = subscribe!(getmarginal(B), Bbuffer)
    fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
    
    # Sometimes it is essential to have an initial posterior marginals 
	# to start VMP iterations. Without this step inference engine 
	# will wait for them forever and will never react on new observations
    setmarginal!(A, vague(MatrixDirichlet, 3, 3))
    setmarginal!(B, vague(MatrixDirichlet, 3, 3))

	# To make many vmp iterations we simply pass our observations
	# in our model multiple time. It forces an inference engine to react on them 
	# multiple times and update posterior marginals several times
    for i in 1:n_its
        update!(x, observations)
    end
    
	# It is a good practice to always unsubscribe at the end of the 
	# inference stage
    unsubscribe!(ssub)
    unsubscribe!(Asub)
    unsubscribe!(Bsub)
    unsubscribe!(fesub)
    
    return map(getvalues, (sbuffer, Abuffer, Bbuffer, fe))
end

# ╔═╡ 31f009ba-fb29-4633-b3a6-56e10544126a
md"""
Since this a approximate variational message passing results may differ depending on the number of VMP iterations performed.
"""

# ╔═╡ ebcbb7a5-5fb5-4de9-bb93-ec3d9c6af031
n_its_slider = @bind(n_itr, ThrottledSlider(2:25, default = 15, throttle = 150));

# ╔═╡ 010364bf-b778-4696-be41-8ccdeb3132d3
md"""
Here we may modify some of parameters for the data generation process and VMP inference procedure.
"""

# ╔═╡ f267254c-84f8-4b67-b5c7-1dfabda12e3d
md"""
|             |                  |
| ----------- | ---------------- |
| seed        | $(seed_slider)   |
| n           | $(n_slider)      |
| n_its       | $(n_its_slider)  |
"""

# ╔═╡ 9697161a-ebfd-4071-8424-a0c5c83c9ce8
x, s = generate_data(n, A, B, seed = seed);

# ╔═╡ c9dd7f62-e02b-4b50-ae47-151b8772a899
s_est, A_est, B_est, fe = inference(x, n_itr);

# ╔═╡ 5b60460c-5f93-41b5-b44b-820523098385
begin
	
	local fig = Figure(resolution = (500, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	lines!(ax, 1:length(fe), fe, label = "Bethe Free Energy", linewidth = 2)
	
	axislegend(ax, labelsize = 16, position = :rt)
	
	@saveplot fig "hmm_fe"
end

# ╔═╡ eeef9d32-805f-4958-aabb-b552430f2d4d
md"""
We can see that in all of the cases our algorithm minimises Bethe Free Energy functional correctly and, hence, should lead to a proper approximate solution.
"""

# ╔═╡ 97176b6d-881d-4be1-922e-4f89b99bf792
md"""
### Verification

To inspect the quality of the inferred state sequence, we plot the simulated process, observations and inferred state sequence.
"""

# ╔═╡ 64a3414c-5b95-46a3-94d1-9f2d7f24284d
md"""
|             |                  |
| ----------- | ---------------- |
| seed        | $(seed_slider)   |
| n           | $(n_slider)      |
| n_its       | $(n_its_slider)  |
"""

# ╔═╡ 6fe2241b-80d5-45b8-84c9-3d834f2a4121
begin
	local fig = Figure(resolution = (500, 350))
	local ax  = Makie.Axis(fig[1, 1])
	
	local range       = 1:n
	local s_states    = argmax.(s)
	local s_estimated = mean.(last(s_est))
	local s_err       = std.(last(s_est))
	local c           = Makie.wong_colors()
	
	lines!(ax, range, s_estimated, color = c[1], label = "estimated")
	band!(ax, range, 
		s_estimated .- s_err, s_estimated .+ s_err,
		color = (c[1], 0.45)
	)
	scatter!(ax, range, s_states, color = c[6], markersize = 5, label = "real")
	
	# p = scatter!(p, range, s_states, ms = 3, label = "Real states", color = :red3, alpha = 0.75)
	# p = plot!(p, range, s_estimated, ribbon = s_err, fillalpha = 0.2, label = "Estimated", color = :orange)
	# p = plot!(p, legend = :bottomright)
	# p = plot!(xlabel = "Time-step index", xguidefontsize = 8)
	
	axislegend(ax, labelsize = 16, position = :rb)
	
	@saveplot fig "hmm_inference"
end

# ╔═╡ c908eba5-8e36-416b-a2d7-3d994c454b85
md"""
We may also interested in our state transition matrices estimations. We may plot real and estimated matrices together side by side on one plot to verify that we actually find it correctly.
"""

# ╔═╡ 74924c8e-7141-4e0d-aaa6-78732726498e
# begin
# 	local rotate90 = (m) -> begin
# 		hcat(reverse(collect(eachcol(transpose(m))))...)
# 	end
	
# 	local rA     = rotate90(A)
# 	local rA_est = rotate90(mean(last(A_est)))
# 	local rB     = rotate90(B)
# 	local rB_est = rotate90(mean(last(B_est)))
# 	local p1 = plot()
# 	local p2 = plot()
# 	local p3 = plot()
# 	local p4 = plot()
	
# 	color = :blues

# 	p1 = heatmap!(
# 		p1, [ "1", "2", "3" ], [ "3", "2", "1" ], rA_est, 
# 		title = "Estimated matrix coefficients", titlefont = 8,
# 		clim = (0.0, 1.0), color=color
# 	)
# 	p2 = heatmap!(
# 		p2, [ "1", "2", "3" ], [ "3", "2", "1" ], 
# 		title = "Squared error difference", titlefont = 8,
# 		abs2.(rA .- rA_est), 
# 		clim = (0.0, 1.0), color=color
# 	)
	
# 	p3 = heatmap!(
# 		p3, [ "1", "2", "3" ], [ "3", "2", "1" ], rB_est, 
# 		title = "Estimated matrix coefficients", titlefont = 8,
# 		clim = (0.0, 1.0), color=color
# 	)
# 	p4 = heatmap!(
# 		p4, [ "1", "2", "3" ], [ "3", "2", "1" ], 
# 		title = "Squared error difference", titlefont = 8,
# 		abs2.(rB .- rB_est),
# 		clim = (0.0, 1.0), color=color
# 	)
	
# 	local p = plot(
# 		p1, p2, p3, p4, size = (500, 400), layout = @layout([ a b; c d ])
# 	)
	
# 	@saveplot plot(p1, p2) "hmm_A"
# 	@saveplot plot(p3, p4) "hmm_B"
	
# 	p
# end

# ╔═╡ ff8aa1bb-300d-471f-b260-25b477366e22
md"""
On the left side we may see an estimated matrices $A$ and $B$. On the right there is a difference beetween real matrices and the estimated ones. We can see that the algorithm correctly predicted matrices $A$ and $B$ in our model with very small error.
"""

# ╔═╡ 99869875-9699-4c3b-8ea1-8a4905ef260d
md"""
### Benchmarking

In this section we will benchmark inference performance with the help of BenchmarkTools package. ReactiveMP.jl has been designed to be efficient and scalable as much as possible. To show ReactiveMP.jl performance capabilities we run a series of benchmark tests for hidden markov model with different number of observations.
"""

# ╔═╡ 30eb1c10-cd89-4304-abbe-08934a6dcdae
function run_benchmark(params)
	@unpack n, n_itr, seed = params
	
	# Transition probabilities (some transitions are impossible)
    A = [0.9 0.0 0.1; 0.1 0.9 0.0; 0.0 0.1 0.9] 
    # Observation noise
    B = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9] 
	
	x, s           = generate_data(n, A, B, seed = seed)
	s_, A_, B_, fe = inference(x, n_itr);
	benchmark      = @benchmark inference($x, $n_itr);
	
	s_estimated = last(s_)
	A_estimated = last(A_)
	B_estimated = last(B_)
	
	output = 
		@strdict n n_itr seed x s s_estimated A_estimated B_estimated fe benchmark
	
	return output
end

# ╔═╡ ec396167-5410-47f0-bd49-63d0e77f409c
# Here we create a list of parameters we want to run our benchmarks with
benchmark_allparams = dict_list(Dict(
	"n"     => [ 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25_000 ],
	"n_itr" => [ 5, 10, 15, 20, 25, @onlyif("n" >= 5000, 60) ],
	"seed"  => 42,
));

# ╔═╡ c6c92af2-5797-4d69-9dda-e07eb68ff359
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create a new benchmarks 
# but will reload it from data folder
hmm_benchmarks = map(benchmark_allparams) do params
	path = datadir("benchmark", "hmm", "smoothing")
	result, _ = produce_or_load(path, params, run_benchmark)
	return result
end;

# ╔═╡ d1b2fe90-c446-41ef-a446-e33ca6f3d376
begin
	target_seed = 42
	target_n_itr = 15
end

# ╔═╡ d9c069b9-2fdf-493f-9745-82e118ae2e80
begin
	local path_smoothing = datadir("benchmark", "hmm", "smoothing")
	
	local white_list   = [ "n", "n_itr", "seed" ]
	local special_list = [
		:min => (data) -> string(
			round(minimum(data["benchmark"]).time / 1_000_000, digits = 2), "ms"
		),
		:mean => (data) -> string(
			round(mean(data["benchmark"]).time / 1_000_000, digits = 2), "ms"
		),
		:gctime => (data) -> string(
			round(minimum(data["benchmark"]).gctime / minimum(data["benchmark"]).time * 100, digits = 2), "%"
		)
	]
	
	local df_smoothing = collect_results(path_smoothing, 
		white_list = white_list,
		special_list = special_list
	)
	
	local query_smoothing = @from row in df_smoothing begin
		@where row.seed == target_seed && row.n_itr == target_n_itr
		@orderby ascending(row.n)
		@select { row.n, row.min, row.gctime }
	end
	
	local res_smoothing = DataFrame(query_smoothing)
	
	res_smoothing
end

# ╔═╡ b0f8b89a-b4e4-4199-94c9-0b9a9fd06902
target_n_itrs = [ 5, 15, 25 ]

# ╔═╡ 6c3b504a-b8d6-41e2-8178-fc819a465f2b
begin
	
	local fig = Figure(resolution = (500, 350))
	
	local ax = Makie.Axis(fig[1, 1])
	
	ax.xlabel = "Number of observations in dataset log-scale()"
	ax.ylabel = "Time (in ms, log-scale)"
	ax.xscale = Makie.pseudolog10
	ax.yscale = Makie.pseudolog10
	
	ax.xticks = (
		[ 50, 100, 200, 500, 1000, 2000, 5000, 10_000 ], 
		[ "50", "100", "200", "500", "1e3", "2e3", "5e3", "1e4" ]
	)
	
	ax.yticks = (
		[ 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5_000, 10_000 ], 
		[ "5", "10", "20", "50", "100", "200", "500", "1e3", "2e3", "5e3", "1e4" ]
	)
	
	local mshapes = [ :utriangle, :diamond, :pentagon ]
	
	for (mshape, target_n_itr) in zip(mshapes, target_n_itrs)
		local filtered = filter(hmm_benchmarks) do b
			return b["n_itr"] === target_n_itr
		end

		local range      = map(f -> f["n"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000
		
		lines!(ax, range, timings, label = "VMP n_itr = $(target_n_itr)")
		scatter!(ax, range, timings, marker = mshape, markersize = 16)
	end
	
	axislegend(ax, labelsize = 16, position = :lt)
	
	@saveplot fig "hmm_benchmark_observations"
end

# ╔═╡ f2b495ab-f3fc-4e09-b0d9-33311c2d08f8
target_ns = [ 50, 500, 5000 ]

# ╔═╡ 2796dc7a-cd99-4f8e-89eb-89ea4e302813
begin
	
	local fig = Figure(resolution = (500, 350))
	
	local ax = Makie.Axis(fig[1, 1])
	
	ax.xlabel = "Number of performed VMP iterations (log-scale)"
	ax.ylabel = "Time (in ms, log-scale)"
	ax.xscale = Makie.pseudolog10
	ax.yscale = Makie.pseudolog10
	
	ax.xticks = (
		[ 5, 10, 15, 20, 25 ],
		string.([ 5, 10, 15, 20, 25 ])
	)
	
	ax.yticks = (
		[ 5, 20, 100, 300, 1000, 3_000, 10_000, 3e4, 1e5 ], 
		[ "5", "20", "100", "300", "1e3", "3e3", "1e4", "3e4", "1e5" ]
	)
	
	ylims!(ax, (2, 3e5 ))
	
	local mshapes = [ :utriangle, :diamond, :pentagon ]
	
	for (mshape, target_n) in zip(mshapes, target_ns)
		local filtered = filter(hmm_benchmarks) do b
			return b["n"] === target_n
		end

		local range      = map(f -> f["n_itr"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000
		local ylim       = (1e0, 10maximum(timings))
		
		lines!(ax, range, timings, label = "n_observations = $(target_n)")
		scatter!(ax, range, timings, marker = mshape, markersize = 16)
	end
	
	axislegend(ax, labelsize = 16, position = :lt)
	
	@saveplot fig "hmm_benchmark_iterations"
end

# ╔═╡ 244e48cd-d6f2-46b5-bc1b-1edbc23b504f
md"""
### Comparison with Turing.jl

In this section we want to compare results and performance of ReactiveMP.jl with another probabilistic programming library which is called Turing.jl. Turing is a general probabilistic programming toolbox and does not use message passing for inference procedure, but sampling. Message passing has an advantage over sampling approach for conjugate models (which our linear gaussian state space model is) because it may fallback to analytically tractable update rules, where sampling cannot. 
"""

# ╔═╡ ae607537-67e0-48aa-95ef-69152c28394a
import Turing, MCMCChains

# ╔═╡ 57386c71-97b0-4284-8c21-e22954f4c1ce
# Turing model definition.
Turing.@model BayesHmm(y, K) = begin
    # Get observation length.
    N = length(y)

    # State sequence.
    s = Turing.tzeros(Int, N)

    # Transition matrix.
    A = Vector{Vector}(undef, K)
	
    # Observations model matrix.
	B = Vector{Vector}(undef, K)

    # Assign distributions to each element
    # of the transition matrix and the
    # emission matrix.
    for i = 1:K
        A[i] ~ Dirichlet(ones(K)/K)
		B[i] ~ Dirichlet(ones(K)/K)
    end

    # Observe each point of the input.
    s[1] ~ Categorical(K)
    y[1] ~ Categorical(vec(B[s[1]]))

    for i = 2:N
        s[i] ~ Categorical(vec(A[s[i - 1]]))
        y[i] ~ Categorical(vec(B[s[i]]))
    end
end;

# ╔═╡ d3fde0c7-c45a-4d4f-a8cc-f3fc482a4119
function inference_turing(observations; nsamples = 250, seed = 42)
	rng     = MersenneTwister(seed)
	sampler = Turing.Gibbs(Turing.HMC(0.1, 20, :A, :B), Turing.PG(100, :s))
    return Turing.sample(rng, BayesHmm(observations, 3), sampler, nsamples)
end

# ╔═╡ 4918f180-03e4-4ad4-bf96-14a3a167feb6
begin 
	n_turing = 50
	x_turing, s_turing = generate_data(n_turing, A, B, seed = seed);
	x_turing           = float.(last.(findmax.(x_turing))) # Turing format
end;

# ╔═╡ bd9a52e2-a9bc-4513-babc-037068a939e8
s_turing_estimated = inference_turing(x_turing);

# ╔═╡ f9c03f28-6eef-4929-9ea8-86d6a27ba192
summary = Turing.summarize(s_turing_estimated, Turing.mean, Turing.std);

# ╔═╡ 2e24dbb8-9e86-4cad-94e6-77b4544593d5
invertf(x; d = 3) = (1 - d) / (d - 1) * x - ((1 - d) / (d - 1) - d)

# ╔═╡ 0280b410-4603-486d-b681-59d103812220
begin
	turing_s_means = invertf.(map(i -> summary[Symbol("s[$i]")].nt.mean |> first, 1:n))
	turing_s_stds = map(i -> summary[Symbol("s[$i]")].nt.std |> first, 1:n)
end

# ╔═╡ dedfdd8e-dbd0-42ef-954a-43ae0745c92d
begin
	f = Figure()
	
	ax = CairoMakie.Axis(f[1, 1])
	
	lines!(ax, turing_s_means)
	band!(ax, 1:n, turing_s_means .- turing_s_stds, turing_s_means .+ turing_s_stds )
	lines!(ax, float.(last.(findmax.(s_turing))))
	
	f
end

# ╔═╡ Cell order:
# ╠═a4affbe0-9d3d-11eb-1ca5-059daf3d3141
# ╠═f1b341d4-d730-4f98-89d8-3337e6bc0ce1
# ╠═4a6a323b-4424-441c-9807-6707ea7ba410
# ╟─83eb01c5-e383-4c8d-b187-12fc9dc87ecb
# ╟─df0529bb-70fb-43b2-954d-838fe2165b76
# ╠═5c14ef6a-9a9a-4fb6-9e11-90a61b878866
# ╟─927a5ad7-7ac0-4e8a-a755-d1223093b992
# ╟─c01151b2-5476-4170-971c-518019e891f8
# ╠═9aff5f73-0e50-4ed3-ac35-355afa0d4137
# ╠═9e84c744-5157-4e7c-b910-481f8b6e086c
# ╠═30f46dae-6665-4bbc-9451-ee6744c5a6aa
# ╠═34ef9070-dc89-4a56-8914-b3c8bd3288ba
# ╟─d02a557f-2ecb-4016-bd36-579c7e8bb012
# ╠═1d23082e-ab18-4ca6-9383-aaebddb29f00
# ╟─f28c42fc-1e8a-4e3b-a0cf-73da0c7875cd
# ╟─dd0e02b4-e5c7-46eb-8a80-060d22e640b9
# ╠═3866d103-7f73-4ff1-8949-20ab0cfd3704
# ╟─31f009ba-fb29-4633-b3a6-56e10544126a
# ╠═ebcbb7a5-5fb5-4de9-bb93-ec3d9c6af031
# ╟─010364bf-b778-4696-be41-8ccdeb3132d3
# ╟─f267254c-84f8-4b67-b5c7-1dfabda12e3d
# ╠═9697161a-ebfd-4071-8424-a0c5c83c9ce8
# ╠═c9dd7f62-e02b-4b50-ae47-151b8772a899
# ╟─5b60460c-5f93-41b5-b44b-820523098385
# ╟─eeef9d32-805f-4958-aabb-b552430f2d4d
# ╟─97176b6d-881d-4be1-922e-4f89b99bf792
# ╟─64a3414c-5b95-46a3-94d1-9f2d7f24284d
# ╟─6fe2241b-80d5-45b8-84c9-3d834f2a4121
# ╟─c908eba5-8e36-416b-a2d7-3d994c454b85
# ╠═74924c8e-7141-4e0d-aaa6-78732726498e
# ╟─ff8aa1bb-300d-471f-b260-25b477366e22
# ╟─99869875-9699-4c3b-8ea1-8a4905ef260d
# ╠═30eb1c10-cd89-4304-abbe-08934a6dcdae
# ╠═ec396167-5410-47f0-bd49-63d0e77f409c
# ╠═c6c92af2-5797-4d69-9dda-e07eb68ff359
# ╠═d1b2fe90-c446-41ef-a446-e33ca6f3d376
# ╟─d9c069b9-2fdf-493f-9745-82e118ae2e80
# ╠═b0f8b89a-b4e4-4199-94c9-0b9a9fd06902
# ╟─6c3b504a-b8d6-41e2-8178-fc819a465f2b
# ╠═f2b495ab-f3fc-4e09-b0d9-33311c2d08f8
# ╟─2796dc7a-cd99-4f8e-89eb-89ea4e302813
# ╟─244e48cd-d6f2-46b5-bc1b-1edbc23b504f
# ╠═ae607537-67e0-48aa-95ef-69152c28394a
# ╠═57386c71-97b0-4284-8c21-e22954f4c1ce
# ╠═d3fde0c7-c45a-4d4f-a8cc-f3fc482a4119
# ╠═4918f180-03e4-4ad4-bf96-14a3a167feb6
# ╠═bd9a52e2-a9bc-4513-babc-037068a939e8
# ╠═f9c03f28-6eef-4929-9ea8-86d6a27ba192
# ╠═0280b410-4603-486d-b681-59d103812220
# ╠═2e24dbb8-9e86-4cad-94e6-77b4544593d5
# ╟─dedfdd8e-dbd0-42ef-954a-43ae0745c92d
