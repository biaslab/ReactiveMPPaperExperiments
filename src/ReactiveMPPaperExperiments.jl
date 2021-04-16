module ReactiveMPPaperExperiments

# include("helpers/throttled_slider.jl")

# using Plots
# using PGFPlotsX

# is_in_pluto() = in(:PlutoRunner, names(Main))

# function instantiate()
#     if is_in_pluto()
#         gr()
#     else
#         eval("using PGFPlotsX")
#         pgfplotsx()
#         println("Script instantiated")
#     end

#     return nothing
# end

# function saveplot(p, name)
#     if is_in_pluto()
#         return p
#     end

#     savefig(p, "figures/$(name).tikz")
#     savefig(p, "figures/$(name).png")

#     println("Saved figures/$(name)*")
    
#     return p
# end

end # module
