module ReactiveMPPaperExperiments

export @saveplot

include("helpers/throttled_slider.jl")

macro saveplot(p, name)
    output = quote
        if !in(:PlutoRunner, names(Main))
            output_tikz = string($name, ".pdf")
            output_png = string($name, ".png")
            save(plotsdir(output_tikz), $p)
            save(plotsdir(output_png), $p)
        end
        $p
    end
    return esc(output)
end

end # module
