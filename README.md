# ReactiveMPPaperExperiments

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> ReactiveMPPaperExperiments

It is authored by Bagaev Dmitry <bvdmitri@gmail.com>.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

# Notebooks

All notebooks with code examples are present in `notebooks/` folder. Notebooks have been written with [`Pluto.jl`](https://github.com/fonsp/Pluto.jl) package. By default `Pluto` is added in `Project.toml`. To run and explore experiments go to `notebooks/` folder and use the following command:

```bash
julia -e 'import Pluto; Pluto.run()'
```
