export edim

edim(dims...) = (array) -> map(e -> e[dims...], array)