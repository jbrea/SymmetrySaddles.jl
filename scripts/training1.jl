using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Distributed, DrWatson
if haskey(ENV, "PARALLEL")
    nprocs() < Sys.CPU_THREADS && addprocs(Sys.CPU_THREADS - nprocs(),
                                           exeflags = "--project",
                                           dir = projectdir())
end
@everywhere using Flux, DrWatson, Random, SymmetrySaddles

allparams = Dict(:seed => collect(1:50),
                 :n => [5, 45],
                 :activation => [:exp, :sigmoid],
                 :N => 10,
                 :Ninner => 10^5,
                 :gitcommit => gitdescribe(),
                 :maxtime => 10^3)

@sync @distributed for p in dict_list(allparams)
    Random.seed!(p[:seed])
    a = getproperty(Flux, p[:activation])
    m = model(p[:n], activation = a)
    input, target, = default_target(activation = a)
    result = train!(m, input, target,
                    N = p[:N], Ninner = p[:Ninner], maxtime = p[:maxtime])
    fname = savename(p, "bson")
    subsample = union(floor.(Int, 10 .^ (0:.05:4)))
    p[:trace] = result.trace[[subsample; subsample[end]+1:end]]
    p[:loss] = result.loss[[subsample; subsample[end]+1:end]]
    p[:iterations] = result.iterations[[subsample; subsample[end]+1:end]]
    wsave(datadir("simulations", fname), p)
end
