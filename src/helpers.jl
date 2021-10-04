export edim, collect_benchmarks

import DataFrames

edim(dims...) = (array) -> map(e -> e[dims...], array)

function collect_benchmarks(data::AbstractVector{D}; kwargs...) where { D <: Dict }
    special = if haskey(kwargs, :special_list)
        map(data) do row 
            extra = map(kwargs[:special_list]) do spair
                return string(first(spair)) => last(spair)(row)
            end
            return Dict(extra)
        end
    else
        map(_ -> Dict(), data) 
    end

    if haskey(kwargs, :white_list)
        data = map(row -> filter(d -> d[1] ∈ kwargs[:white_list], row), data)
    end

    data = map(d -> merge(d[1], d[2]), zip(data, special))

    return reduce(vcat, DataFrames.DataFrame.(data))
end