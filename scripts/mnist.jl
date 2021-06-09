using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Flux, MLDatasets, Zygote, Plots, DataFrames, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs

function getdata()
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    xtest = Flux.flatten(xtest)
    ytest = onehotbatch(ytest, 0:9)
    DataLoader(xtest, ytest, batchsize = 10^4)
end

function build_model(n; imgsize=(28,28,1), nclasses=10)
    Chain(Dense(prod(imgsize), n, softplus), Dense(n, nclasses))
end

mutable struct MyADAM
    ϵ::Float64
    eta::Float64
    beta::Tuple{Float64,Float64}
    state::IdDict
    norms::Vector{Float64}
end

MyADAM(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8) = MyADAM(ϵ, η, β, IdDict(), [])

function Flux.Optimise.apply!(o::MyADAM, x, Δ)
    η, β = o.eta, o.beta

    mt, vt, βp = get!(o.state, x) do
      (zero(x), zero(x), Float64[β[1], β[2]])
    end :: Tuple{typeof(x),typeof(x),Vector{Float64}}

    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.ϵ) * η
    βp .= βp .* β

    push!(o.norms, sum(abs2, Δ))
    return Δ
end

function run(n, T)
    data = getdata()
    m = build_model(n)
    ps = params(m)
    opt = MyADAM()
    x, y = first(data)
    losses = Float32[]
    norms = Float32[]
    for _ in 1:T
        gs = Zygote.gradient(ps) do
            loss = logitcrossentropy(m(x), y)
            Zygote.ignore() do
                push!(losses, loss)
            end
            loss
        end
        push!(norms, sum(sum.(abs2, gs[p] for p in ps)))
        Flux.update!(opt, ps, gs)
    end
    losses, norms, reshape(sum(reshape(opt.norms, length(ps), :), dims = 1), :)
end

results = DataFrame(n = Int[], losses = [], gnorms = [], lrgnorms = [])

ns = (10, 20, 50, 100, 250, 1000)
for n in ns
    for _ in 1:10
        println(n)
        losses, norms, lrnorms = run(n, 500)
        push!(results, [n, Ref(losses), Ref(norms), Ref(lrnorms)])
    end
end

gr()
plots = []
for n in 1:length(ns)
    p0 = plot()
    p1 = plot()
    p2 = plot()
    for i in (n-1)*10 + 1:n*10
        plot!(p0, results.losses[i][], title = "$(ns[n]) loss")
        plot!(p1, results.gnorms[i][], title = "$(ns[n]) |∇|")
        plot!(p2, results.lrgnorms[i][], title = "$(ns[n]) |lr*∇̂|")
    end
    push!(plots, plot(p0, p1, p2, layout = (1, 3), legend = false, yaxis = :log10, ))
end
p = plot(plots..., layout = (3, 2), size = (2048, 1024))
savefig("figure.pdf")
