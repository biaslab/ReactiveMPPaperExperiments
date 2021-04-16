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

# ╔═╡ a4affbe0-9d3d-11eb-1ca5-059daf3d3141
using Revise

# ╔═╡ 99b2fcde-0ad6-4388-9eea-e0200fb2615d
using ReactiveMPPaperExperiments; ReactiveMPPaperExperiments.instantiate();

# ╔═╡ 74d2bcc7-2538-41eb-8937-b912d456f4bf
using PlutoUI, Images

# ╔═╡ 9dfa1b26-1080-4432-b8f0-ef513f76bfd0
using Rocket, ReactiveMP, GraphPPL, Distributions, Random, Plots

# ╔═╡ 83eb01c5-e383-4c8d-b187-12fc9dc87ecb
md"""
### Hidden Markov Model

In this demo the goal is to perform approximate variational Bayesian Inference for Hidden Markov Model (HMM).

We will use the following model:

```math
\begin{equation}
  \begin{aligned}
    \mathbf{s}_k & \sim \, \text{Cat}(\mathbf{A}\mathbf{s}_{k - 1})\\
    \mathbf{x}_k & \sim \, \text{Cat}(\mathbf{B}\mathbf{s}_{k})\\
  \end{aligned}
\end{equation}
```

where $\text{Cat}$ denotes Categorical distribution. Also, we denote by $\mathbf{s}_k$ the current state of the system (at time step $k$), by $\mathbf{s}_{k - 1}$ the previous state at time $k-1$, $\mathbf{A}$ and $\mathbf{B}$ are a constant system inputs and $\mathbf{x}_k$ are noise-free observations.

We will build a full graph for this model and perform VMP iterations during an inference procedure.
"""

# ╔═╡ df0529bb-70fb-43b2-954d-838fe2165b76
md"""
### Model specification
"""

# ╔═╡ 927a5ad7-7ac0-4e8a-a755-d1223093b992
md"""
GraphPPL.jl offer a model specification syntax that resembles closely to the mathematical equations defined above. We use `Transition` node for `Cat(Ax)` distribution, `datavar` placeholders are used to indicate variables that take specific values at a later date. For example, the way we feed observations into the model is by iteratively assigning each of the observations in our dataset to the data variables `x`.
"""

# ╔═╡ c01151b2-5476-4170-971c-518019e891f8
md"""
### Synthetic data generation

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

# ╔═╡ 1d23082e-ab18-4ca6-9383-aaebddb29f00
begin
	seed_slider = 
		@bind(seed, ThrottledSlider(1:100, default = 41, show_value = true))
	
	n_slider = 
		@bind(n, ThrottledSlider(2:100, default = 75, show_value = true))
end;

# ╔═╡ f28c42fc-1e8a-4e3b-a0cf-73da0c7875cd
md"""
|             |                 |
| ----------- | ----------------|
| seed        | $(seed_slider)  |
| n           | $(n_slider)     |
"""

# ╔═╡ 56917bc7-dce2-4251-9997-6164a2c2f24f
x, s = generate_data(n, A, B, seed = seed)

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
	# multiple times and leads to a better approximations 
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
n_its_slider = @bind(
	n_itr, ThrottledSlider(2:25, default = 15, show_value = true, throttle = 150)
);

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

# ╔═╡ c9dd7f62-e02b-4b50-ae47-151b8772a899
s_est, A_est, B_est, fe = inference(x, n_itr);

# ╔═╡ 5b60460c-5f93-41b5-b44b-820523098385
begin
	local p = plot()
	
	p = plot!(p, title = "Bethe Free Energy functional", titlefontsize = 10)
	p = plot!(p, fe, xticks = 1:3:n_itr, label = "\$BFE\$")
	p = plot!(p, ylabel = "Free energy", yguidefontsize = 8)
	p = plot!(p, xlabel = "Iteration index", xguidefontsize = 8)
	
	if n_itr > 10
		local range        = 1:n_itr
		local lens_x_range = [ Int(round(0.75 * n_itr)), n_itr ]

		local diff = 0.5abs(maximum(fe[lens_x_range]) - minimum(fe[lens_x_range]))

		local lens_y_range = [ 
			minimum(fe[lens_x_range]) - diff, maximum(fe[lens_x_range]) + diff 
		]

		p = lens!(
			p, lens_x_range, lens_y_range, 
			inset = (1, bbox(0.5, 0.4, 0.4, 0.2))
		)
	end
	
	ReactiveMPPaperExperiments.saveplot(p, "hmm_fe")
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
	local p = plot()
	
	local range       = 1:n
	local s_states    = argmax.(s)
	local s_estimated = mean.(last(s_est))
	local s_err       = std.(last(s_est))
	
	p = scatter!(p, range, s_states, ms = 3, label = "Real states")
	p = plot!(p, range, s_estimated, ribbon = s_err, fillalpha = 0.2, label = "Estimated")
	p = plot!(xlabel = "Iteration index", xguidefontsize = 8)
	
	ReactiveMPPaperExperiments.saveplot(p, "hmm_inference")
end

# ╔═╡ c908eba5-8e36-416b-a2d7-3d994c454b85
md"""
We may also interested in our state transition matrices estimations. We may plot real and estimated matrices together side by side on one plot to verify that we actually find it correctly.
"""

# ╔═╡ 74924c8e-7141-4e0d-aaa6-78732726498e
begin
	local rotate90 = (m) -> begin
		hcat(reverse(collect(eachcol(transpose(m))))...)
	end
	
	local rA     = rotate90(A)
	local rA_est = rotate90(mean(last(A_est)))
	local rB     = rotate90(B)
	local rB_est = rotate90(mean(last(B_est)))
	local p1 = plot()
	local p2 = plot()
	local p3 = plot()
	local p4 = plot()
	
	color = :blues

	p1 = heatmap!(
		p1, [ "1", "2", "3" ], [ "3", "2", "1" ], rA_est, 
		title = "Estimated matrix coefficients", titlefont = 8,
		clim = (0.0, 1.0), color=color
	)
	p2 = heatmap!(
		p2, [ "1", "2", "3" ], [ "3", "2", "1" ], 
		title = "Squared error difference", titlefont = 8,
		abs2.(rA .- rA_est), 
		clim = (0.0, 1.0), color=color
	)
	
	p3 = heatmap!(
		p3, [ "1", "2", "3" ], [ "3", "2", "1" ], rB_est, 
		title = "Estimated matrix coefficients", titlefont = 8,
		clim = (0.0, 1.0), color=color
	)
	p4 = heatmap!(
		p4, [ "1", "2", "3" ], [ "3", "2", "1" ], 
		title = "Squared error difference", titlefont = 8,
		abs2.(rB .- rB_est),
		clim = (0.0, 1.0), color=color
	)
	
	local p = plot(
		p1, p2, p3, p4, size = (500, 400), layout = @layout([ a b; c d ])
	)
	
	ReactiveMPPaperExperiments.saveplot(plot(p1, p2), "hmm_A")
	ReactiveMPPaperExperiments.saveplot(plot(p3, p4), "hmm_B")
	
	p
end

# ╔═╡ ff8aa1bb-300d-471f-b260-25b477366e22
md"""
On the left side we may see an estimated matrices $A$ and $B$. On the right there is a difference beetween real matrices and the estimated ones. We can see that the algorithm correctly predicted matrices $A$ and $B$ in our model with very small error.
"""

# ╔═╡ Cell order:
# ╠═a4affbe0-9d3d-11eb-1ca5-059daf3d3141
# ╠═99b2fcde-0ad6-4388-9eea-e0200fb2615d
# ╠═74d2bcc7-2538-41eb-8937-b912d456f4bf
# ╠═9dfa1b26-1080-4432-b8f0-ef513f76bfd0
# ╟─83eb01c5-e383-4c8d-b187-12fc9dc87ecb
# ╟─df0529bb-70fb-43b2-954d-838fe2165b76
# ╠═5c14ef6a-9a9a-4fb6-9e11-90a61b878866
# ╟─927a5ad7-7ac0-4e8a-a755-d1223093b992
# ╟─c01151b2-5476-4170-971c-518019e891f8
# ╠═9aff5f73-0e50-4ed3-ac35-355afa0d4137
# ╠═9e84c744-5157-4e7c-b910-481f8b6e086c
# ╠═30f46dae-6665-4bbc-9451-ee6744c5a6aa
# ╠═34ef9070-dc89-4a56-8914-b3c8bd3288ba
# ╠═1d23082e-ab18-4ca6-9383-aaebddb29f00
# ╟─f28c42fc-1e8a-4e3b-a0cf-73da0c7875cd
# ╠═56917bc7-dce2-4251-9997-6164a2c2f24f
# ╟─dd0e02b4-e5c7-46eb-8a80-060d22e640b9
# ╠═3866d103-7f73-4ff1-8949-20ab0cfd3704
# ╟─31f009ba-fb29-4633-b3a6-56e10544126a
# ╠═ebcbb7a5-5fb5-4de9-bb93-ec3d9c6af031
# ╟─010364bf-b778-4696-be41-8ccdeb3132d3
# ╟─f267254c-84f8-4b67-b5c7-1dfabda12e3d
# ╠═c9dd7f62-e02b-4b50-ae47-151b8772a899
# ╟─5b60460c-5f93-41b5-b44b-820523098385
# ╟─eeef9d32-805f-4958-aabb-b552430f2d4d
# ╟─97176b6d-881d-4be1-922e-4f89b99bf792
# ╟─64a3414c-5b95-46a3-94d1-9f2d7f24284d
# ╟─6fe2241b-80d5-45b8-84c9-3d834f2a4121
# ╟─c908eba5-8e36-416b-a2d7-3d994c454b85
# ╟─74924c8e-7141-4e0d-aaa6-78732726498e
# ╟─ff8aa1bb-300d-471f-b260-25b477366e22
