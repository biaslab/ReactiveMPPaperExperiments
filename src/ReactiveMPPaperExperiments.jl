module ReactiveMPPaperExperiments

export @saveplot, @saveplot_force

include("helpers.jl")
include("algebra.jl")
include("helpers/throttled_slider.jl")

include("models.jl")
include("models/lgssm.jl")

macro saveplot(p, name)
    output = quote
        if false
            local output_tikz = string($name, ".pdf")
            local output_png = string($name, ".png")
            save(plotsdir(output_tikz), $p)
            save(plotsdir(output_png), $p)
        end
        $p
    end
    return esc(output)
end

macro saveplot_force(p, name)
    output = quote
        local output_tikz = string($name, ".pdf")
        local output_png = string($name, ".png")
        save(plotsdir(output_tikz), $p)
        save(plotsdir(output_png), $p)
        $p
    end
    return esc(output)
end

end # module
