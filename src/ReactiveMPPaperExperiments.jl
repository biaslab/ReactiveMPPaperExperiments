module ReactiveMPPaperExperiments

export debounced, saveplot

using Plots
using PGFPlotsX

is_in_pluto() = in(:PlutoRunner, names(Main))

function instantiate()
    if is_in_pluto()
        gr()
    else
        eval("using PGFPlotsX")
        pgfplotsx()
    end

    return nothing
end

function saveplot(p, name)
    if is_in_pluto()
        return p
    end

    savefig(p, "figures/$(name).tikz")
    savefig(p, "figures/$(name).png")
    
    return p
end


# Thanks to Fons in Zulip

struct Debounced
    name::Symbol
    x::Any
    wait::Real
end

debounced(name, x; wait = 50) = Debounced(name, x, wait)

function Base.show(io::IO, m::MIME"text/html", d::Debounced)
    id = String(rand('a':'z',10))

    print(io, "<span id=$(id)></span>")
    show(io, m, d.x)

    print(io, """
        <script>
        const span = document.querySelector('#$(id)')
        const el = span.nextElementSibling

        const _ = window._

        el.addEventListener("input", (e) => {
            e.stopPropagation()
        })

        var value = null

        const debounced_setter = _.debounce(() => {
            span.value = value
            span.dispatchEvent(new CustomEvent("input", {}))
        }, $(d.wait))

        const dothing = async () => {
            for(const value_promise of Generators.input(el)) {
                value = await value_promise
                debounced_setter()
            }
        }

        dothing()
        </script>
        """)
end

Base.get(d::Debounced) = try
    Base.get(d.x)
catch
    missing
end

end # module
