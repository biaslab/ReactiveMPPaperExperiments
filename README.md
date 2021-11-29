# ReactiveMP experiments

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named *ReactiveMPPaperExperiments*.

This repository has been authored by Bagaev Dmitry <d.v.bagaev@tue.nl>.

To (locally) reproduce this project, do the following:

0. Install the [Julia](https://julialang.org/) programming language, [Jupyter](https://jupyter.org/) notebooks and [`IJulia`](https://github.com/JuliaLang/IJulia.jl) kernel.
1. Install `DrWatson` julia package with the following command:
   ```bash
   julia -e 'import Pkg; Pkg.add("DrWatson")'
   ```
2. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently (see Step 3.).
3. Run the following command in the root directory of this repository:
   ```bash
   julia --project -e 'import Pkg; Pkg.instantiate()'
   ```
   This command will replicate the exact same environment as was used during the experiments. Any other `Pkg` related commands may alter this enviroment or/and change versions of packages.
4. (Optional) Download precomputed benchmark .JLD2 files from [releases](https://github.com/biaslab/ReactiveMPPaperExperiments/releases) page and unzip them in `data` folder.

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

# Notebooks

All notebooks with code examples are present in `notebooks/` folder. Notebooks have been written with the [Jupyter](https://jupyter.org/). By default `IJulia` is not included in `Project.toml`. To run and explore experiments you will need Jupyter and IJulia kernel installed on your system.
