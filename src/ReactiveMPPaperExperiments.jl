module ReactiveMPPaperExperiments

export @saveplot, @saveplot_force

include("helpers.jl")
include("algebra.jl")

include("metrics.jl")
include("metrics/amse_gaussian.jl")
include("metrics/amse_categorical.jl")

include("models.jl")
include("models/lgssm.jl")
include("models/hmm.jl")
include("models/hgf.jl")
include("models/gmm.jl")

macro saveplot(p, name)
    output = quote
        if true
            local output_tikz = string($name, ".pdf")
            local output_png = string($name, ".png")
            local output_eps = string($name, ".eps")
            save(plotsdir(output_tikz), $p)
            save(plotsdir(output_png), $p)
            save(plotsdir(output_eps), $p)
        end
        $p
    end
    return esc(output)
end

macro saveplot_force(p, name)
    output = quote
        local output_tikz = string($name, ".pdf")
        local output_png = string($name, ".png")
        local output_eps = string($name, ".eps")
        save(plotsdir(output_tikz), $p)
        save(plotsdir(output_png), $p)
        save(plotsdir(output_eps), $p)
        $p
    end
    return esc(output)
end

end # module
