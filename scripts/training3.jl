using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Distributed, DrWatson
if haskey(ENV, "PARALLEL")
    nprocs() < Sys.CPU_THREADS && addprocs(Sys.CPU_THREADS - nprocs(),
                                           exeflags = "--project",
                                           dir = projectdir())
end
@everywhere begin
    using Flux, DrWatson, Random, SymmetrySaddles
    import SymmetrySaddles: Linear
end

allparams = Dict(:seed => collect(1:50),
                 :n => [5, 20],
                 :activation => [:tanh],
                 :N => 10,
                 :layers => 3,
                 :Ninner => 10^5,
                 :gitcommit => gitdescribe(),
                 :maxtime => 10^3)

@everywhere begin
    function default_target3()
        input = let x = -5:.25:5
            hcat([[x, y] for x in x, y in x]...)
        end
        # result of fitting to f(x, y) = sin(2*x) + x + cos(3*y) - .4*(y-1)^2
        teacher = Chain(Dense([-0.18295849092410896 0.7804681785573752;
                               0.07884022077670189 -0.1465591508168933;
                               -0.0006776100961034238 -0.5753342110597969;
                               -0.004087729656236544 0.18765919486973165],
                              [2.5965546364462906, 0.5436550791983767, 0.1023093314704724, -0.2827111440903739],
                              tanh),
                        Dense([-1.4966248703636236 -0.12076718222817126 0.894756433267219 1.5114479474313065;
                               -0.22777233953771267 -2.0365527674966297 -0.48308633806391305 -2.5956210355303915;
                               2.281192356630861 -0.07241999324324101 -2.851031848646282 -6.108432396679688;
                               0.3665586834900288 1.128325600897833 2.8640316335653946 8.276582305785169],
                              [1.376429442213494, 0.03694563285674877, -2.8764368931239526, 2.6449408721177603],
                              tanh),
                        Dense([1.8731785007685788 14.996529203089828 -1.6545236215750898 5.218306385868382;
                               3.7936449798531915 -1.1612043338560116 1.1178745772483494 -1.4514282248737453;
                               -9.89278446944757 0.4163114508522701 -5.0124951156923245 -3.326074479899465;
                               -1.0008398398686116 2.3821573915129037 -0.5335246759338316 0.5269891920106062],
                              [-1.2307808733517698, 1.8997166238206356, 1.8468200537557915, 1.0909995507783499],
                              tanh),
                        Dense([-2.039005552438561 -16.369169803157757 -11.109545132565406 -13.54102305403078],
                              [6.034865881710452], identity))
        (input, teacher(input), teacher)
    end
end

@sync @distributed for p in dict_list(allparams)
    Random.seed!(p[:seed])
    a = getproperty(Flux, p[:activation])
    n = p[:n]
    m = Chain([Dense(Float64.(l.W), Float64.(l.b), l.σ)
               for l in Chain(Dense(2, n, a),
                              Dense(n, n, a),
                              Dense(n, n, a),
                              Dense(n, 1)).layers]...)
    input, target,  = default_target3()
    result = train!(m, input, target,
                    N = p[:N], Ninner = p[:Ninner], maxtime = p[:maxtime],
                    collect_history = false)
    fname = savename(p, "bson")
    p[:loss] = result.loss
    p[:iterations] = result.iterations
    p[:student] = m
    wsave(datadir("simulations", fname), p)
end
