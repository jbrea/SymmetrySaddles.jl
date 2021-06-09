# Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances

This is the code to reproduce the figures in
[Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances](https://arxiv.org/abs/2105.12221),
Berfin Şimşek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, Johanni Brea

## Install dependencies

Download this code base, open a [Julia](julialang.org) REPL and do:
```julia
julia> using Pkg
julia> Pkg.activate("path/to/this/project")
julia> Pkg.instantiate()
```

## Training

In a terminal run
```
export PARALLEL=
julia scripts/training1.jl
julia scripts/training3.jl
julia scripts/training_g.jl
```
Uncomment the first line, if you do not want to use multiple cores.

## Plotting

```
julia scripts/fig1.jl
julia scripts/fig5.jl
```

For the Figure 8 in the appendix run
```
julia scripts/mnist.jl
```
This script will take quite a bit of time to run.
