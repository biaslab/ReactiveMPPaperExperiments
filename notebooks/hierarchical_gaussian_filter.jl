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

# ╔═╡ 4ccc3ae8-be67-46b1-a228-636d86321a79
begin 
	using Revise
	using Pkg
end

# ╔═╡ 5f80d3bc-9de9-11eb-1f38-af8cac91b6c0
Pkg.activate("$(@__DIR__)/../") # To disable pluto's built-in pkg manager

# ╔═╡ 1a6b8f2a-537e-44c0-90fe-afb57c2086f1
begin
	
	using ReactiveMPPaperExperiments
	using DrWatson, PlutoUI, Images
	using CairoMakie
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random
	using BenchmarkTools
	
	import ReactiveMP: update!
end

# ╔═╡ 8b370073-4495-4181-9edc-379783091753
md"""
### Hierarchical Gaussian Filter

In this demo the goal is to perform approximate variational Bayesian Inference for Univariate Hierarchical Gaussian Filter (HGF).

Simple HGF model can be defined as:

```math
\begin{equation}
  \begin{aligned}
    x^{(j)}_k & \sim \, \mathcal{N}(x^{(j)}_{k - 1}, f_k(x^{(j - 1)}_k)) \\
    y_k & \sim \, \mathcal{N}(x^{(j)}_k, \tau_k)
  \end{aligned}
\end{equation}
```

where $j$ is an index of layer in hierarchy, $k$ is a time step and $f_k$ is a variance activation function. `ReactiveMP.jl` export Gaussian Controlled Variance (GCV) node with $f_k = \exp(\kappa x + \omega)$ variance activation function. By default uses Gauss-Hermite cubature with a prespecified number of approximation points in the cubature. We can change the number of points in Gauss-Hermite cubature with the help of metadata structures in `ReactiveMP.jl`. 

```math
\begin{equation}
  \begin{aligned}
    z_k & \sim \, \mathcal{N}(z_{k - 1}, \mathcal{\tau_z}) \\
    x_k & \sim \, \mathcal{N}(x_{k - 1}, \exp(\kappa z_k + \omega)) \\
    y_k & \sim \, \mathcal{N}(x_k, \mathcal{\tau_y})
  \end{aligned}
\end{equation}
```

In this experiment we will create a single time step of the graph and perform variational message passing filtering alrogithm to estimate hidden states of the system.

For simplicity and smooth reactive graphs we will consider $\tau_z$, $\tau_y$, $\kappa$ and $\omega$ known and fixed.
"""

# ╔═╡ 6418daac-7b3f-49e7-94d6-0e76abdcffac
md"""
### Model Specification
"""

# ╔═╡ caf7ceab-2e0d-4c16-ac63-66fbed3702d4
@model [ default_factorisation = MeanField() ] function hgf(gh_n, τ_z, τ_y, κ, ω)
	
	# First as usual we create a placeholder inputs for our priors
	zv_prior = datavar(Float64, 2)
	sv_prior = datavar(Float64, 2)
	
	zv_min ~ NormalMeanPrecision(zv_prior[1], zv_prior[2])
	sv_min ~ NormalMeanPrecision(sv_prior[1], sv_prior[2])
	
	# Z-layer random walk 
	zv ~ NormalMeanPrecision(zv_min, τ_z)
	
	# We use Gauss Hermite approximation to approximate
	# nonlinearity between layers in hierarchy
	meta = GCVMetadata(GaussHermiteCubature(gh_n))
	
	# S-layer GCV with structured factorisation
	gcv, sv ~ GCV(sv_min, zv, κ, ω) where { 
		q = q(sv, sv_min)q(zv)q(κ)q(ω), meta = meta 
	}
	
	yv = datavar(Float64)
	yv ~ NormalMeanPrecision(sv, τ_y)
	
	return zv_prior, sv_prior, gcv, zv, sv, yv
end

# ╔═╡ 696dd629-f0d3-4608-9de2-82ea6b1a7253
function generate_test_data(n, τ_z, τ_y, κ, ω; seed = 42)
	
	rng = MersenneTwister(seed)
    
    z_data = Vector{Float64}(undef, n)
    s_data = Vector{Float64}(undef, n)
    y_data = Vector{Float64}(undef, n)

	z_σ = sqrt(inv(τ_z))
    y_σ = sqrt(inv(τ_y))
    
    z_data[1] = zero(Float64)
    s_data[1] = zero(Float64)
    y_data[1] = rand(rng, Normal(s_data[1], y_σ))
    
    for i in 2:n
        z_data[i] = rand(rng, Normal(z_data[i - 1], z_σ))
        s_data[i] = rand(rng, Normal(s_data[i - 1], sqrt(exp(κ * z_data[i] + ω))))
        y_data[i] = rand(rng, Normal(s_data[i], y_σ))
    end
    
    return z_data, s_data, y_data
end

# ╔═╡ eb491feb-1dfe-4787-90eb-1e0ecff07438
begin
	
	seed_slider = @bind(
		seed, ThrottledSlider(1:100, default = 43, show_value = true)
	)
	
	n_slider = @bind(
		n, ThrottledSlider(1:1000, default = 100, show_value = true)
	)
	
	nitr_slider = @bind(
		nitr, ThrottledSlider(1:100, default = 10, show_value = true)
	)
	
	gh_n_slider = @bind(
		gh_n, ThrottledSlider(1:13, default = 9, show_value = true)
	)
	
	τ_z_slider = @bind(
		τ_z, ThrottledSlider(0.01:0.01:10.0, default = 1.0, show_value = true)
	)
	
	τ_y_slider = @bind(
		τ_y, ThrottledSlider(0.01:0.01:10.0, default = 0.1, show_value = true)
	)
	
	κ_slider = @bind(
		κ, ThrottledSlider(0.0:0.1:1.0, default = 1.0, show_value = true)
	)
	
	ω_slider = @bind(
		ω, ThrottledSlider(-10.0:0.1:10.0, default = 2.0, show_value = true)
	)
	
end;

# ╔═╡ ebb8d862-29ab-4a78-a9ed-91774904bab8
z_data, s_data, y_data = generate_test_data(n, τ_z, τ_y, κ, ω, seed = seed);

# ╔═╡ 363a8fcd-ff86-4d8e-bafc-92e8e6324ffe
function inference(data, nitr, gh_n, τ_z, τ_y, κ, ω)

    zm = keep(Marginal)
    sm = keep(Marginal)
    fe = keep(Float64)

	# We create a single time section of a full graph here
    model, (zv_prior, sv_prior, gcv, zv, sv, yv) = hgf(gh_n, τ_z, τ_y, κ, ω)
	
	# We subscribe on all posterior marginal updates and free energy values
	zmsub = subscribe!(getmarginal(zv), zm)
	smsub = subscribe!(getmarginal(sv), sm)
	fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
	
	# We set an initial joint marginal around GCV node to be able 
	# to start inference procedure
    setmarginal!(gcv, :y_x, MvNormalMeanCovariance([ 0.0, 0.0 ], [ 5.0, 5.0 ]))
	
	# We keep track of 'current' posterior marginals at time step k
	sv_k = NormalMeanVariance(0.0, 5.0)
	zv_k = NormalMeanVariance(0.0, 5.0)
	
	setmarginal!(sv, sv_k)
    setmarginal!(zv, zv_k)

	# We run our online inference procedure for each observation in data 
    for observation in data
		
		# To perform multiple VMP iterations we pass our data multiple times
		# It forces an inference backend to react on data multiple times and 
		# hence update posterior marginals multiple times
		for i in 1:nitr
			update!(zv_prior[1], mean(zv_k))
			update!(zv_prior[2], precision(zv_k))
			update!(sv_prior[1], mean(sv_k))
			update!(sv_prior[2], precision(sv_k))
			update!(yv, observation)
		end

		# Update current posterior marginals at time step k
		zv_k = last(zm)
		sv_k = last(sm)
    end
    
	# It is a good practice to always unsubscribe from streams of data 
	# at the end of the inference procedure
	unsubscribe!(zmsub)
	unsubscribe!(smsub)
	unsubscribe!(fesub)
    
    return map(getvalues, (fe, zm, sm))
end

# ╔═╡ 0b0f6b07-7a88-413b-b3ad-a3de3ddc2e7e
fe, zm, sm = inference(y_data, nitr, gh_n, τ_z, τ_y, κ, ω);

# ╔═╡ 1594d934-9c3c-477b-9931-bc265af18046
md"""
|      |                |
| ---- | -------------- |
| seed | $(seed_slider) |
| n    | $(n_slider)    |
| nitr | $(nitr_slider)    |
| gh_n | $(gh_n_slider)    |
| τ_z  | $(τ_z_slider)  |
| τ_y  | $(τ_y_slider)  |
| κ  | $(κ_slider)  |
| ω  | $(ω_slider)  |
"""

# ╔═╡ 7b68e3c5-2dc6-4310-b437-8ab12f924dd6
begin 
	local c = Makie.wong_colors()
	
	local f1    = Figure(resolution = (350, 350))
	local f2    = Figure(resolution = (350, 350))
	local f3    = Figure(resolution = (350, 350))
	local shift = nitr
	local range = 1:shift:length(zm)
	local grid  = 1:n

	local ax1 = Makie.Axis(f1[1, 1])
	local ax2 = Makie.Axis(f2[1, 1])
	local ax3 = Makie.Axis(f3[1, 1])
	
	function plot_z(fig)
		
		lines!(fig, grid, z_data, color = :red3, label = "real")
		lines!(fig, grid, mean.(zm[range]), color = c[3], label = "estimated")
		band!(fig, grid, 
			mean.(zm[1:shift:end]) .- std.(zm[1:shift:end]),
			mean.(zm[1:shift:end]) .+ std.(zm[1:shift:end]),
			color = (c[3], 0.65)
		)
		
		axislegend(fig, labelsize = 16, position = :rt)
	end
	
	function plot_s(fig)
		
		lines!(fig, grid, s_data, color = :purple, label = "real")
		lines!(fig, grid, mean.(sm[range]), color = c[1], label = "estimated")
		band!(fig, grid, 
			mean.(sm[1:shift:end]) .- std.(sm[1:shift:end]),
			mean.(sm[1:shift:end]) .+ std.(sm[1:shift:end]),
			color = (c[1], 0.65)
		)
		
		axislegend(fig, labelsize = 16, position = :rb)
	end
	
	local rfe = vec(sum(reshape(fe, (nitr, n)), dims = 2))
	
	function plot_fe(fig)
		lines!(fig, 1:length(rfe), rfe, linewidth = 2, label = "Bethe Free Energy")
		axislegend(fig, labelsize = 16)
	end
	
	plot_z(ax1)
	plot_s(ax2)
	plot_fe(ax3)
	
	@saveplot f1 "hgf_inference_z"
	@saveplot f2 "hgf_inference_s"
	@saveplot f3 "hgf_inference_fe"
	
	local af = Figure(resolution = (350 * 3, 350))
	
	plot_z(Makie.Axis(af[1, 1]))
	plot_s(Makie.Axis(af[1, 2]))
	plot_fe(Makie.Axis(af[1, 3]))
	
	af
end

# ╔═╡ b9bd5a69-16fb-4b6b-84c7-80a514c91ccf
md"""
### Benchmark 
"""

# ╔═╡ cb60be26-87f5-4cf7-b3b4-d16197a7a065
function run_benchmark(params)
	@unpack n, iters, seed, τ_z, τ_y, κ, ω = params
	
	z_data, s_data, y_data = generate_test_data(n, τ_z, τ_y, κ, ω, seed = seed);
	
	fe, zm, sm = inference(y_data, iters, 12, τ_z, τ_y, κ, ω);
	benchmark  = @benchmark inference($y_data, $iters, 12, $τ_z, $τ_y, $κ, $ω);
	
	output = @strdict n iters seed τ_z τ_y κ ω fe zm sm benchmark
	
	return output
end

# ╔═╡ 597b54c3-d8ce-4aed-a961-9a0fc9640a22
# Here we create a list of parameters we want to run our benchmarks with
benchmark_params = dict_list(Dict(
	"n"     => [ 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 10000 ],
	"iters" => [ 5, 10, 15, 20 ],
	"seed"  => [ 42, 123 ],
	"τ_z"   => 1.0,
	"τ_y"   => 1.0,
	"κ"     => 1.0,
	"ω"     => 0.0,	
));

# ╔═╡ 3f045ae5-e696-4113-99f7-b8d1704b1bb3
# First run maybe slow, you may track the progress in the terminal
# Subsequent runs will not create new benchmarks 
# but will reload it from data folder
hgf_benchmarks = map(benchmark_params) do params
	path = datadir("benchmark", "hgf", "filtering")
	result, _ = produce_or_load(path, params; tag = false) do p
		run_benchmark(p)
	end
	return result
end;

# ╔═╡ 39fa40f2-4077-495b-b4e8-61d908ad4c83
target_n_itrs = [ 5, 10, 20 ]

# ╔═╡ bb85ebc5-5f25-4a9d-8f09-bd41061ccc70
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
		local filtered = filter(hgf_benchmarks) do b
			return b["iters"] === target_n_itr
		end

		local range      = map(f -> f["n"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000
		
		lines!(ax, range, timings, label = "VMP n_itr = $(target_n_itr)")
		scatter!(ax, range, timings, marker = mshape, markersize = 16)
	end
	
	axislegend(ax, labelsize = 16, position = :lt)
	
	@saveplot fig "hgf_benchmark_observations"
end

# ╔═╡ 1cc14e44-8744-41b0-860c-770ae3b3e07c
target_ns = [ 500, 2500, 10000 ]

# ╔═╡ 16e975fc-2692-4625-bb4a-f6693205df3a
begin
	
	local fig = Figure(resolution = (500, 350))
	
	local ax = Makie.Axis(fig[1, 1])
	
	ax.xlabel = "Number of performed VMP iterations (log-scale)"
	ax.ylabel = "Time (in ms, log-scale)"
	ax.xscale = Makie.pseudolog10
	ax.yscale = Makie.pseudolog10
	
	ax.xticks = (
		[ 5, 10, 15, 20 ],
		string.([ 5, 10, 15, 20 ])
	)
	
	ax.yticks = (
		[ 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5_000, 10_000 ], 
		[ "5", "10", "20", "50", "100", "200", "500", "1e3", "2e3", "5e3", "1e4" ]
	)
	
	ylims!(ax, (25, 8e4 ))
	
	local mshapes = [ :utriangle, :diamond, :pentagon ]
	
	for (mshape, target_n) in zip(mshapes, target_ns)
		local filtered = filter(hgf_benchmarks) do b
			return b["n"] === target_n
		end

		local range      = map(f -> f["iters"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000
		local ylim       = (1e0, 10maximum(timings))
		
		lines!(ax, range, timings, label = "n_observations = $(target_n)")
		scatter!(ax, range, timings, marker = mshape, markersize = 16)
	end
	
	axislegend(ax, labelsize = 16, position = :lt)
	
	@saveplot fig "hgf_benchmark_iterations"
end

# ╔═╡ Cell order:
# ╠═4ccc3ae8-be67-46b1-a228-636d86321a79
# ╠═5f80d3bc-9de9-11eb-1f38-af8cac91b6c0
# ╠═1a6b8f2a-537e-44c0-90fe-afb57c2086f1
# ╟─8b370073-4495-4181-9edc-379783091753
# ╟─6418daac-7b3f-49e7-94d6-0e76abdcffac
# ╠═caf7ceab-2e0d-4c16-ac63-66fbed3702d4
# ╠═696dd629-f0d3-4608-9de2-82ea6b1a7253
# ╠═eb491feb-1dfe-4787-90eb-1e0ecff07438
# ╠═ebb8d862-29ab-4a78-a9ed-91774904bab8
# ╠═363a8fcd-ff86-4d8e-bafc-92e8e6324ffe
# ╠═0b0f6b07-7a88-413b-b3ad-a3de3ddc2e7e
# ╟─1594d934-9c3c-477b-9931-bc265af18046
# ╟─7b68e3c5-2dc6-4310-b437-8ab12f924dd6
# ╟─b9bd5a69-16fb-4b6b-84c7-80a514c91ccf
# ╠═cb60be26-87f5-4cf7-b3b4-d16197a7a065
# ╠═597b54c3-d8ce-4aed-a961-9a0fc9640a22
# ╠═3f045ae5-e696-4113-99f7-b8d1704b1bb3
# ╟─39fa40f2-4077-495b-b4e8-61d908ad4c83
# ╟─bb85ebc5-5f25-4a9d-8f09-bd41061ccc70
# ╠═1cc14e44-8744-41b0-860c-770ae3b3e07c
# ╟─16e975fc-2692-4625-bb4a-f6693205df3a
