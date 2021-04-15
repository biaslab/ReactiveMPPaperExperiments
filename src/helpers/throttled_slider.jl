export ThrottledSlider

struct ThrottledSlider
    range      :: AbstractRange
    default    :: Number
    throttle   :: Number
    show_value :: Bool
end

"""A Slider on the given `range`.
## Examples
`@bind x ThrottledSlider(1:10)`
`@bind x ThrottledSlider(0.00 : 0.01 : 0.30)`
`@bind x ThrottledSlider(1:10; default=8, show_value=true)`
`@bind x ThrottledSlider(1:10; default=8, show_value=true, throttle = 50)`
"""
function ThrottledSlider(range::AbstractRange; default=missing, show_value=false, throttle = 32) 
    return ThrottledSlider(range, (default === missing) ? first(range) : default, throttle, show_value)
end

function Base.show(io::IO, ::MIME"text/html", slider::ThrottledSlider)

    sid = string(join(rand('a':'z', 10)));
    id  = string(join(rand('a':'z', 10)));

    print(io, "<span id='$(sid)'></span>")

    print(io, """<input id='$(id)'
        type="range" 
        min="$(first(slider.range))" 
        step="$(step(slider.range))" 
        max="$(last(slider.range))" 
        value="$(slider.default)"
        >""")
    
    if slider.show_value
        print(io, """<output>$(typeof(slider.default) <: Float64 ? round(slider.default, digits = 3) : slider.default)</output>""")
    end

    script = """
        <script>
            (function() {
                const sp = document.getElementById('$(sid)')
                const el = document.getElementById('$(id)')

                const _ = window._
                const c = window.console

                let value = null

                const dispatchEvent = () => {
                    sp.value = value
                    sp.dispatchEvent(new CustomEvent("input", {}))
                    el.nextElementSibling.value = Number.isInteger(value) ? value : value.toPrecision(2)
                }

                const debouncedEvent = _.throttle(dispatchEvent, $(slider.throttle))

                el.addEventListener("input", (event) => {
                    value = event.target.valueAsNumber;
                    debouncedEvent(event);
                    event.stopImmediatePropagation();
                    event.preventDefault();
                })

            }())
        </script>
    """

    print(io, script)
end

Base.get(slider::ThrottledSlider) = slider.default