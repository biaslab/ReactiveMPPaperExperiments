### A Pluto.jl notebook ###
# v0.14.7

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

# ╔═╡ b7ab78ee-a296-11eb-3f8b-07d7bda0c5be
using Revise

# ╔═╡ b3858be8-bf87-470d-bc27-a441f3854600
using DrWatson

# ╔═╡ 3080c441-0541-47d9-82ae-c20680b444c9
begin
	@quickactivate "ReactiveMPPaperExperiments"
	using ReactiveMPPaperExperiments
end

# ╔═╡ a5ee57e2-b4c4-4365-ba66-135ce6c7d225
begin
	using PlutoUI, Images
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random, Plots
	using LinearAlgebra
	using BenchmarkTools
	if !in(:PlutoRunner, names(Main))
		using PGFPlotsX
		pgfplotsx()
	end
end

# ╔═╡ 7b34d8ee-fcbb-4981-9583-14c4ff4fabcc
md"""
### Multivariate Gaussian Mixture Estimation

In this example we are going to run inference on multivariate gaussian mixture model with ReactiveMP.jl.

The observations in this model are i.i.d., but the model itself shares the same parameters across all mixture components. It can be described in the following way:

```math
\begin{equation}
  \begin{aligned}
	z_i & \sim \, \text{Bernoulli}(\pi)\\	
	y_i & \sim \, \prod_{k = 1}^{N}\mathcal{N}(\bf{m}_k, \bf{W^{-1}}_k)^{z_{i}^{(k)}}
  \end{aligned}
\end{equation}
```

All model parameters are also endowed with priors:

```math
\begin{equation}
  \begin{aligned}
	\pi & \sim \, \text{Beta}(1, 1)\\	
	\bf{m}_k & \sim \, \mathcal{N}(\bf{m}_k^{(0)}, \bf{W^{-1}}_k^{(0)}) \\
	\bf{W}_k & \sim \, \text{Wishart}(\text{d\,}_k^{(0)}, \bf{S}_k^{(0)})
  \end{aligned}
\end{equation}
```
"""



# ╔═╡ fb4a905e-ef2c-4f72-9f4e-98eb78379d3c
md"""
### Syntethic data

First lets generate some synthetic data. We will generate some samples from real Gaussian Mixture distribution with the help of the `Distributions.jl` package.
"""

# ╔═╡ 122b7746-085c-4e85-a223-021c6eca0618
begin 
	# We will generate some slider for later use and interactivity
	seed_slider      = @bind seed ThrottledSlider(1:100, default = 42)
	nmixtures_slider = @bind nmixtures ThrottledSlider(1:12, default = 6)
	nsamples_slider  = @bind nsamples ThrottledSlider(1:1000, default = 500)
	cdistance_slider = @bind cdistance ThrottledSlider(1:50, default = 10)
	angle_slider     = @bind(
		angle, ThrottledSlider(range(0.0, π, length = 10), default = 1.5π / nmixtures)
	)
end;

# ╔═╡ 855217cd-d001-42de-bd8e-dbbe0a3282f8
function generate_data(nsamples, nmixtures, cdistance, angle, seed)
	rng = MersenneTwister(seed)
	
	mixing = ones(nmixtures) # rand(rng, nmixtures)
	mixing = mixing ./ sum(mixing)
	
	switch = Categorical(mixing)
	
	mixtures = map(1:nmixtures) do index
		langle     = angle * (index - 1)
        basis_v    = cdistance * [ 1.0, 0.0 ]
        rotationm  = [ cos(langle) -sin(langle); sin(langle) cos(langle) ]
        mean       = rotationm * basis_v 
        covariance = rotationm * [ 16.0 0.0; 0.0 1.0 ] * transpose(rotationm)
		return MvNormal(mean, Matrix(Hermitian(covariance)))
	end
	
	mixture = MixtureModel(mixtures, mixing)
	
	return collect.(eachcol(rand(rng, mixture, nsamples)))
end

# ╔═╡ cd1c546f-8a88-4e36-bb58-d25743210ba6
y = generate_data(nsamples, nmixtures, cdistance, angle, seed);

# ╔═╡ 008caff9-6acb-40ec-829c-a2be7c2831ca
md"""
Here we can interactively play around our dataset with sliders provided with Pluto notebooks. 
"""

# ╔═╡ 81d4b818-df2a-4863-bea4-c2f4091ed5e2
md"""

|              |              |
|--------------|--------------|
| seed            | $(seed_slider) |
| nsamples            | $(nsamples_slider) |
| nmixtures            | $(nmixtures_slider) |
| cdistance            | $(cdistance_slider) |
| angle            | $(angle_slider) |

"""

# ╔═╡ df6807d9-6e3a-4e74-87fe-b520ca9302c3
begin
	local sdim    = (n) -> (a) -> map(d -> d[n], a)
	local limits  = (-1.5cdistance, 1.5cdistance)
	
	local p = plot(
		title = "Synthetic data from Gaussian Mixture Distribution",
		titlefontsize = 9,
		xlim = limits, ylim = limits,
	)
	
	p = scatter!(
		p, y |> sdim(1), y |> sdim(2), ms = 3, alpha = 0.4, label = "Observations"
	)
	
	@saveplot p "gmm_data"
end

# ╔═╡ aa3f6725-7b64-46d6-9f8d-5aa8e0a0745b
md"""
### Model specification

GraphPPL.jl offers a model specification syntax that resembles closely to the mathematical equations defined above. `datavar` placeholders are used to indicate variables that take specific values at a later date. For example, the way we feed observations into the model is by iteratively assigning each of the observations in our dataset to the data variables `y`.

We also use a mean-field factorisation by default for all parameters in our model.
"""

# ╔═╡ 8babeb1e-2d6a-4245-a12e-141642ded660
# We use mean field factorisation by default for Gaussian Mixture estimation
@model [ default_factorisation = MeanField() ] function gaussian_mixture_model(nmixtures, nsamples)
    
    z = randomvar(nsamples)
    m = randomvar(nmixtures) # A sequence of random variables for means
    w = randomvar(nmixtures) # A sequence of random variables for precision
    y = datavar(Vector{Float64}, nsamples) # A sequence of observed samples
    
    # We set uninformative priors for means and precisions
    for i in 1:nmixtures
        m[i] ~ MvGaussianMeanCovariance(zeros(2), [ 1e6 0.0; 0.0 1e6 ])
        w[i] ~ Wishart(2, [ 1e5 0.0; 0.0 1e5 ])
    end
    
    # Use uninformative prior for switching distribution
    s ~ Dirichlet(ones(nmixtures))
    
    # GaussianMixture node accepts means and precisions arguments in a form of tuple
    means = tuple(m...)
    precs = tuple(w...)
    
    # For each sample assign a switch ditribution to be Categorical and
    #  connect observed sample to a GaussianMixture node
    for i in 1:nsamples
        z[i] ~ Categorical(s)
        y[i] ~ GaussianMixture(z[i], means, precs)
    end
    
    return s, z, m, w, y
end

# ╔═╡ aaff0db9-df96-4cb1-8a0a-e33b0fecd689
md"""
# Inference

Next we need to define our inference procedure. Since we used mean field factorisation we need to perform some VMP iterations. To do so we pass same data to `update!` function multiple times forcing inference backend to react multiple times and hence update posterior marginals of the model parameters.

To actually start VMP procedure we need initial marginals. We use `setmarginal!` function to define an initial marginal for the model parameters. We decouple initial marginals to avoid collapsing of clusters and to improve procedure convergence rate.
"""

# ╔═╡ d033184a-db65-47e7-8596-de3620d8b5bd
function inference(data, viters, nmixtures, cdistance, angle)
    n = length(data)
    
	# First we create a model based on the number of observations 
	# we have in our dataset
    model, (s, z, m, w, y) = gaussian_mixture_model(nmixtures, n)
    
	# Some preallocated buffers for our results
    means_estimates  = buffer(Marginal, nmixtures)
    precs_estimates  = buffer(Marginal, nmixtures)
    mixing_estimate  = nothing
    fe_values        = ScoreActor(Float64)

	# General receipt is to subscribe on marginals of interests
	# and to store them in some buffer for later analysis
    switchsub = subscribe!(getmarginal(s), (m) -> mixing_estimate = m)
    meanssub  = subscribe!(getmarginals(m), means_estimates)
    precssub  = subscribe!(getmarginals(w), precs_estimates)
    fesub     = subscribe!(score(Float64, BetheFreeEnergy(), model), fe_values)
    
	# We use `setmarginal!` function to set initial marginals for VMP procedure
    setmarginal!(s, vague(Dirichlet, nmixtures))
    
	# We decouple initial marginals to avoid collapsing of clusters
	basis_v = [ 1.0, 0.0 ]
    for i in 1:nmixtures
        angle_prior = angle * (i - 1)
        mean_mean_prior = [ 
			cos(angle_prior) -sin(angle_prior); 
			sin(angle_prior) cos(angle_prior) 
		] * basis_v
        mean_mean_cov   = [ 1e6 0.0; 0.0 1e6 ]
        
        setmarginal!(m[i], MvNormalMeanCovariance(mean_mean_prior, mean_mean_cov))
        setmarginal!(w[i], Wishart(2, [ 1e3 0.0; 0.0 1e3 ]))
    end
    
	# We pass data multiple times to force inference engine to react multiple times
	# and hence update posterior marginals multiple times
	for i in 1:viters
        update!(y, data)
    end
    
	# In general, it is always better to unsubscribe 
	# at the end of the inference procedure
    unsubscribe!(switchsub)
    unsubscribe!(meanssub)
    unsubscribe!(precssub)
    unsubscribe!(fesub)
    
    return (
		getvalues(means_estimates),
		getvalues(precs_estimates),
		mixing_estimate,
		getvalues(fe_values)
	)
end

# ╔═╡ 33a45c18-7755-42d9-89cc-41964e76531b
means_estimated, precs_estimated, mixing_estimated, fe = inference(
	y, 5, nmixtures, cdistance, angle
);

# ╔═╡ ea21fb47-d599-449e-a785-e655cb330836
md"""

|              |              |
|--------------|--------------|
| seed            | $(seed_slider) |
| nsamples            | $(nsamples_slider) |
| nmixtures            | $(nmixtures_slider) |
| cdistance            | $(cdistance_slider) |
| angle            | $(angle_slider) |

"""

# ╔═╡ 8182deab-8b09-491c-a802-877d2af25afc
begin
	local sdim    = (n) ->  (a) -> map(d -> d[n], a)
	local limits  = (-1.5cdistance, 1.5cdistance)
	
	local p = plot(
		# title = "Inference results for Gaussian Mixture", titlefontsize = 9,
		xlim = limits, ylim = limits,
	)
	
	p = scatter!(
		p, y |> sdim(1), y |> sdim(2), ms = 3, alpha = 0.4, label = "Observations"
	)
	
	local e_means = mean.(means_estimated)
	local e_precs = mean.(precs_estimated)
	local crange  = range(-2cdistance, 2cdistance, step = 0.25)
	
	for (e_m, e_w) in zip(e_means, e_precs)
	    gaussian = MvNormal(e_m, Matrix(Hermitian(inv(e_w))))
    	p = contour!(p, 
			crange, crange, (x, y) -> pdf(gaussian, [ x, y ]), 
			levels = 3, colorbar = false
		)
	end
	
	@saveplot p "gmm_inference"
end

# ╔═╡ b7d3507c-d244-4118-b20e-ddf662ee5df8
md"""
As we can see our model correctly predicted the underlying means and precisions for the actual mixture components in our gaussian mixture distribution.
"""

# ╔═╡ 3362a9ce-15e3-403d-8747-3e685150f8e2
begin
	local p = plot(
		# title = "Free energy for Gaussian Mixture Model Bayesian Inference",
		titlefontsize = 9
	)
	
	p = plot!(p, 1:length(fe), fe, legend = false)
	
	@saveplot p "gmm_fe"
end

# ╔═╡ fd5b774a-6475-4346-9fdc-e71f89b4be56
md"""
### Benchmark
"""

# ╔═╡ 924d5804-b3e0-4d25-b2ce-5ba8a5b56f75
function run_benchmark(params)
	@unpack n, iters, seed, nmixtures, cdistance, angle = params
	
	y = generate_data(n, nmixtures, cdistance, angle, seed);
	
	ms, ps, mixing, fe = inference(
		y, iters, nmixtures, cdistance, angle
	);
	
	benchmark = @benchmark inference($y, $iters, $nmixtures, $cdistance, $angle);
	
	output = @strdict n iters seed nmixtures cdistance angle ms ps mixing fe benchmark
	
	return output
end

# ╔═╡ 6ec68085-2b43-4eed-95d8-30a2a8b24d48
# Here we create a list of parameters we want to run our benchmarks with
benchmark_params = dict_list(Dict(
	"n"     => [ 50, 100, 250, 500, 750, 1000, 1500, 2000 ],
	"iters" => [ 5, 10, 15 ],
	"seed"  => 42,
	"nmixtures" => [ 4, 5, 6 ],
	"cdistance" => 10,
	"angle"     => 1.5π / 6
));

# ╔═╡ a639c21c-5f97-4fc8-9fc8-795037079480
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create new benchmarks 
# but will reload it from data folder
gmm_benchmarks = map(benchmark_params) do params
	path = datadir("benchmark", "gmm")
	result, _ = produce_or_load(path, params; tag = false) do p
		run_benchmark(p)
	end
	return result
end;

# ╔═╡ Cell order:
# ╠═b7ab78ee-a296-11eb-3f8b-07d7bda0c5be
# ╠═b3858be8-bf87-470d-bc27-a441f3854600
# ╠═3080c441-0541-47d9-82ae-c20680b444c9
# ╠═a5ee57e2-b4c4-4365-ba66-135ce6c7d225
# ╟─7b34d8ee-fcbb-4981-9583-14c4ff4fabcc
# ╟─fb4a905e-ef2c-4f72-9f4e-98eb78379d3c
# ╠═122b7746-085c-4e85-a223-021c6eca0618
# ╠═855217cd-d001-42de-bd8e-dbbe0a3282f8
# ╠═cd1c546f-8a88-4e36-bb58-d25743210ba6
# ╟─008caff9-6acb-40ec-829c-a2be7c2831ca
# ╟─81d4b818-df2a-4863-bea4-c2f4091ed5e2
# ╟─df6807d9-6e3a-4e74-87fe-b520ca9302c3
# ╟─aa3f6725-7b64-46d6-9f8d-5aa8e0a0745b
# ╠═8babeb1e-2d6a-4245-a12e-141642ded660
# ╟─aaff0db9-df96-4cb1-8a0a-e33b0fecd689
# ╠═d033184a-db65-47e7-8596-de3620d8b5bd
# ╠═33a45c18-7755-42d9-89cc-41964e76531b
# ╟─ea21fb47-d599-449e-a785-e655cb330836
# ╠═8182deab-8b09-491c-a802-877d2af25afc
# ╟─b7d3507c-d244-4118-b20e-ddf662ee5df8
# ╟─3362a9ce-15e3-403d-8747-3e685150f8e2
# ╟─fd5b774a-6475-4346-9fdc-e71f89b4be56
# ╠═924d5804-b3e0-4d25-b2ce-5ba8a5b56f75
# ╠═6ec68085-2b43-4eed-95d8-30a2a8b24d48
# ╠═a639c21c-5f97-4fc8-9fc8-795037079480
