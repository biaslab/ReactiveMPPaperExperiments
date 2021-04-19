module ReactiveMPPaperExperiments

export @saveplot

include("helpers/throttled_slider.jl")

macro saveplot(p, name)
    output = quote
        if !in(:PlutoRunner, names(Main))
            output_tikz = string($name, ".tikz")
            output_png = string($name, ".png")
            savefig($p, plotsdir(output_tikz))
            savefig($p, plotsdir(output_png))
        end
        $p
    end
    return esc(output)
end

end # module
