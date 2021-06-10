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

# ╔═╡ 5f80d3bc-9de9-11eb-1f38-af8cac91b6c0
using Revise

# ╔═╡ 9e746f15-2336-4503-a480-a788f62e37f6
using DrWatson

# ╔═╡ 4f959e2f-db67-4cdf-b6a7-bc0e6eb56f21
begin 
	@quickactivate "ReactiveMPPaperExperiments"
	using ReactiveMPPaperExperiments
end

# ╔═╡ 1a6b8f2a-537e-44c0-90fe-afb57c2086f1
begin
	using PlutoUI, Images
    using ReactiveMP, Rocket, GraphPPL, Distributions, Random, Plots
	using BenchmarkTools
	if !in(:PlutoRunner, names(Main))
		using PGFPlotsX
		pgfplotsx()
	end
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

In this experiemtn we will create a single time step of the graph and perform variational message passing filtering alrogithm to estimate hidden states of the system.

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
		seed, ThrottledSlider(1:100, default = 34, show_value = true)
	)
	
	n_slider = @bind(
		n, ThrottledSlider(1:1000, default = 150, show_value = true)
	)
	
	τ_z_slider = @bind(
		τ_z, ThrottledSlider(0.01:0.01:10.0, default = 1.0, show_value = true)
	)
	
	τ_y_slider = @bind(
		τ_y, ThrottledSlider(0.01:0.01:10.0, default = 1.0, show_value = true)
	)
	
	κ_slider = @bind(
		κ, ThrottledSlider(0.0:0.1:1.0, default = 1.0, show_value = true)
	)
	
	ω_slider = @bind(
		ω, ThrottledSlider(-10.0:0.1:10.0, default = 0.0, show_value = true)
	)
	
end;

# ╔═╡ ebb8d862-29ab-4a78-a9ed-91774904bab8
	z_data, s_data, y_data = generate_test_data(n, τ_z, τ_y, κ, ω, seed = seed);

# ╔═╡ a8f1b2f6-3788-4545-8292-bf0e468c465f
function redirect_prior(marginal_of, scheduler, callback)
	return subscribe!(
		getmarginal(marginal_of, IncludeAll()) |> schedule_on(scheduler), 
		callback
	)
end

# ╔═╡ cb09a131-abd1-4c70-8c80-95508368ac0a
function inference(data, nitr, gh_n, τ_z, τ_y, κ, ω)

    zm = keep(Marginal)
    sm = keep(Marginal)
	
    fe = ScoreActor(Float64)

    model, (zv_prior, sv_prior, gcv, zv, sv, yv) = hgf(gh_n, τ_z, τ_y, κ, ω)
    
	rd_scheduler = PendingScheduler()
	
	fesub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
	
    zsub = subscribe!(getmarginal(zv) |> schedule_on(rd_scheduler), zm)
    ssub = subscribe!(getmarginal(sv) |> schedule_on(rd_scheduler), sm)
    
	z_prior_sub = redirect_prior(zv, rd_scheduler, (posterior) -> begin
		update!(zv_prior[1], mean(posterior))
    	update!(zv_prior[2], precision(posterior))	
	end)
	
	s_prior_sub = redirect_prior(sv, rd_scheduler, (posterior) -> begin
		update!(sv_prior[1], mean(posterior))
    	update!(sv_prior[2], precision(posterior))	
	end)
	
	setmarginal!(sv, NormalMeanVariance(0.0, 5.0))
    setmarginal!(zv, NormalMeanVariance(0.0, 5.0))
	
    setmarginal!(gcv, :y_x, MvNormalMeanCovariance([ 0.0, 0.0 ], [ 5.0, 5.0 ]))
	
	release!(rd_scheduler)

    for observation in data
		update!(yv, observation)
		repeat!(model, nitr)
		release!(rd_scheduler)
		release!(fe)
    end
        
    unsubscribe!(z_prior_sub)
    unsubscribe!(s_prior_sub)
	unsubscribe!(zsub)
	unsubscribe!(ssub)
	unsubscribe!(fesub)
    
    return map(getvalues, (fe, zm, sm))
end

# ╔═╡ 0b0f6b07-7a88-413b-b3ad-a3de3ddc2e7e
fe, zm, sm = inference(y_data, 10, 9, τ_z, τ_y, κ, ω);

# ╔═╡ 1594d934-9c3c-477b-9931-bc265af18046
md"""
|      |                |
| ---- | -------------- |
| seed | $(seed_slider) |
| n    | $(n_slider)    |
| τ_z  | $(τ_z_slider)  |
| τ_y  | $(τ_y_slider)  |
| κ  | $(κ_slider)  |
| ω  | $(ω_slider)  |
"""

# ╔═╡ 7b68e3c5-2dc6-4310-b437-8ab12f924dd6
begin 
	local p1 = plot()
	local p2 = plot()

	p1 = plot!(p1, z_data)
	p1 = plot!(p1, mean.(zm), ribbon = std.(zm), legend = false)


	p2 = plot!(p2, s_data)
	p2 = plot!(p2, mean.(sm), ribbon = std.(sm), legend = false)

	
	p3 = plot(fe, legend = false)
	
	@saveplot p1 "hgf_inference_1"
	@saveplot p1 "hgf_inference_2"
	@saveplot p1 "hgf_inference_"
	
	plot(p1, p2, p3, layout = @layout([ a b; c ]))
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
	
	local p = plot(
		title = "Hierarchical Gaussian Filter Benchmark (number of observations)",
		titlefontsize = 10, legend = :bottomright,
		xlabel = "Number of observations in dataset (log-scale)", 
		xguidefontsize = 9,
		ylabel = "Time (in ms, log-scale)", 
		yguidefontsize = 9
	)
	
	local mshapes = [ :utriangle, :diamond, :pentagon ]
	
	for (mshape, target_n_itr) in zip(mshapes, target_n_itrs)
		local filtered = filter(hgf_benchmarks) do b
			return b["iters"] === target_n_itr
		end

		local range      = map(f -> f["n"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000



		p = plot!(
			p, range, timings,
			yscale = :log10, xscale = :log10,
			markershape = mshape, label = "VMP n_itr = $(target_n_itr)"
		)
	end
	
	@saveplot p "hgf_benchmark_observations"
end

# ╔═╡ 1cc14e44-8744-41b0-860c-770ae3b3e07c
target_ns = [ 500, 2500, 10000 ]

# ╔═╡ 16e975fc-2692-4625-bb4a-f6693205df3a
begin
	local p = plot(
		title = "Hierarchical Gaussian Filter Benchmark (iterations)",
		titlefontsize = 10, legend = :bottomright,
		xlabel = "Number of performed VMP iterations (log-scale)", 
		xguidefontsize = 9,
		ylabel = "Time (in ms, log-scale)", 
		yguidefontsize = 9
	)
	
	local mshapes = [ :utriangle, :diamond, :pentagon, :dtriangle ]
	
	for (mshape, target_n) in zip(mshapes, target_ns)
		local filtered = filter(hgf_benchmarks) do b
			return b["n"] === target_n
		end

		local range      = map(f -> f["iters"], filtered)
		local benchmarks = map(f -> f["benchmark"], filtered)
		local timings    = map(t -> t.time, minimum.(benchmarks)) ./ 1_000_000
		local ylim       = (1e0, 10maximum(timings))


		p = plot!(
			p, string.(range), timings,
			yscale = :log10, xscale = :log10,
			markershape = mshape, label = "n_observations = $(target_n)", ylim = ylim
		)
	end
	
	@saveplot p "hgf_benchmark_iterations"
end

# ╔═╡ Cell order:
# ╠═5f80d3bc-9de9-11eb-1f38-af8cac91b6c0
# ╠═9e746f15-2336-4503-a480-a788f62e37f6
# ╠═4f959e2f-db67-4cdf-b6a7-bc0e6eb56f21
# ╠═1a6b8f2a-537e-44c0-90fe-afb57c2086f1
# ╟─8b370073-4495-4181-9edc-379783091753
# ╟─6418daac-7b3f-49e7-94d6-0e76abdcffac
# ╠═caf7ceab-2e0d-4c16-ac63-66fbed3702d4
# ╠═696dd629-f0d3-4608-9de2-82ea6b1a7253
# ╠═eb491feb-1dfe-4787-90eb-1e0ecff07438
# ╠═ebb8d862-29ab-4a78-a9ed-91774904bab8
# ╠═a8f1b2f6-3788-4545-8292-bf0e468c465f
# ╠═cb09a131-abd1-4c70-8c80-95508368ac0a
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
