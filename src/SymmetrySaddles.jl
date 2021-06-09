module SymmetrySaddles
using Flux, Zygote, ForwardDiff, NLopt, Statistics, PGFPlotsX, LinearAlgebra

export model, coords, train!, hess, grad, plot_traj, plot_result, plot_alignment,
default_target, duplicate, perturb!, cluster

###
### Linear Layer
###

struct Linear{F,S}
    w::S
    σ::F
end
function Linear(in::Integer, out::Integer, σ = identity;
                initW = Flux.glorot_uniform)
    Linear(Float64.(initW(out, in)), σ)
end
Flux.@functor Linear
(l::Linear)(x) = l.σ.(l.w * x)

###
### Neural Network
###

model(n; activation = exp) = Chain(Linear(2, n, activation), Linear(n, 1))
function model(coords::NTuple{3, Vector{<:Number}}; activation = exp)
    Chain(Linear([coords[1] coords[2]], activation),
        Linear(reshape(coords[3], 1, :), identity))
end
function model(x::Vector{<:Number}; activation = exp)
    n = length(x) ÷ 3
    model((x[1:n], x[n+1:2n], x[2n+1:3n]); activation = activation)
end
coords(m) = (m.layers[1].w[:, 1], m.layers[1].w[:, 2], reshape(copy(m.layers[2].w), :))

function default_target(; activation = exp)
    coords = ([0.6, -0.5, -0.2, 0.1],
              [0.5, 0.5, -0.6, -0.6],
              [1.0, 1.0, 1.0, 1.0])
    input = let x = -5:.25:5
        hcat([[x, y] for x in x, y in x]...)
    end
    m = model(coords, activation = activation)
    y = m(input)
    (input = input, y = y, coords = coords, model = m)
end

###
### Loss Function and Training
###

ℓ(y, ŷ) = mean(abs2, y .- ŷ)
function gℓ(model, ps, x, y)
    l = zero(Float64)
    gs = gradient(ps) do
        l = ℓ(y, model(x))
    end
    gs, l
end
function set!(l::Linear, theta, start)
    s = start[]
    l.w .= reshape(theta[s:s + length(l.w) - 1], size(l.w)...)
    start[] += length(l.w)
end
function set!(l::Dense, theta, start)
    s = start[]
    l.W .= reshape(theta[s:s + length(l.W) - 1], size(l.W)...)
    start[] += length(l.W)
    s = start[]
    l.b .= theta[s:s + length(l.b) - 1]
    start[] += length(l.b)
end
function lossfunc(model, x, y)
    ps = params(model)
    (theta, grad) -> begin
        start = Ref(1)
        for l in model.layers
            set!(l, theta, start)
        end
        gs, l = gℓ(model, ps, x, y)
        if length(grad) > 0
            grad .= vcat([reshape(gs[p], :) for p in ps]...)
        end
        l
    end
end

function train!(model, x, y;
                N = 10, Ninner = 10^3, Nstart = 10,
                maxtime = 10^3, nlopt_threshold = 1e-7,
                opt = ADAM(), collect_history = true,
                nlopt_method = :LD_SLSQP)
    tic = time()
    eta0 = opt.eta
    loss = Float64[]
    t = Int[]
    ps = params(model)
    trace = collect_history ? [coords(model)] : []
    for i in 1:N
        loss_tmp = Float64[]
        for j in 1:Ninner
            gs, l = gℓ(model, ps, x, y)
            Flux.update!(opt, ps, gs)
            push!(loss_tmp, l)
            if i == 1 && (j % Nstart) == 0 && j < Ninner
                push!(loss, mean(@view loss_tmp[end-Nstart+1:end]))
                collect_history && push!(trace, coords(model))
                push!(t, j)
            end
        end
        println("$(i*Ninner) $(floor(time() - tic)) $(mean(loss_tmp))")
        push!(loss, mean(loss_tmp))
        collect_history && push!(trace, coords(model))
        push!(t, i*Ninner)
        opt.eta = eta0/(1 + i)
        if mean(loss_tmp) < nlopt_threshold || i == N
            ell = lossfunc(model, x, y)
            theta = vcat([reshape(p, :) for p in ps]...)
            nopt = Opt(nlopt_method, length(theta))
            nopt.min_objective = ell
            nopt.lower_bounds = theta .- 10.
            nopt.upper_bounds = theta .+ 10.
            nopt.maxtime = maxtime
            res = optimize(nopt, theta)
            collect_history && push!(trace, coords(model))
            push!(loss, res[1])
            return (model = model, loss = loss, opt = opt, trace = trace, iterations = t)
        end
    end
end

###
### Helper Functions for Inspection
###

activation(m) = m.layers[1].σ
hess(m, x, y) = hess(coords(m), x, y; activation = activation(m))
function hess(coords::Tuple, x, y; activation = exp)
    lossfunc = θ -> ℓ(model(θ; activation)(x), y)
    ForwardDiff.hessian(lossfunc, vcat(coords...))
end
grad(m, x, y) = grad(coords(m), x, y; activation = activation(m))
function grad(coords::Tuple, x, y; activation = exp)
    lossfunc = θ -> ℓ(model(θ; activation)(x), y)
    ForwardDiff.gradient(lossfunc, vcat(coords...))
end

function duplicate(m, p::Pair...)
    c = coords(m)
    for (j, v) in p
        @assert sum(v) ≈ 1 "Values have to sum to 1."
        for i in 1:2
            append!(c[i], fill(c[i][j], length(v)-1))
        end
        append!(c[3], v[2:end] * c[3][j])
        c[3][j] *= v[1]
    end
    model(c, activation = activation(m))
end

function perturb!(m; epsilon = 1e-3)
    for l in m.layers
        l.w .+= randn(size(l.w)) * epsilon
    end
    m
end

###
### Plotting
###

function transform_trace(trace)
    x, y, z = [hcat([t[i] for t in trace]...) for i in 1:3]
    [(x[i, :], y[i, :], z[i, :]) for i in 1:size(x, 1)]
end
function plot_traj(t, subsample = 1:length(t);
                   targetcoords = default_target().coords, kwargs...)
    @pgf Axis(merge({ytick = [], xtick = [], colorbar,
                   "colormap/jet", width = "5cm",
                   ylabel = raw"$w_1$", xlabel = raw"$w_2$",
                   ymin = -1.5, ymax = 1.5, xmin = -1.5, xmax = 1.5},
                    PGFPlotsX.Options(kwargs...)),
                  ["\\draw[gray] ($(-0*x), $(-0*y)) -- ($(10*x), $(10*y));
                    \\draw[gray,dashed] ($(-10*x), $(-10*y)) -- ($(0*x), $(0*y));"
                   for (x, y, z) in zip(targetcoords...)]...,
                  [Plot({no_marks, black}, Coordinates(x, y))
                   for (x, y, z) in transform_trace([t[subsample]; t[end]])]...,
              Plot({only_marks, scatter, mark_size = 1.5,
                    point_meta = "explicit",
                    point_meta_min = -1,
                    point_meta_max = 1},
                   Table({meta = "meta"}, ["x" => t[end][1],
                                           "y" => t[end][2],
                                           "meta" => (t[end][3])]))
             )
end
function plot_result(seed, n, l, t, i;
                     subsample = union(floor.(Int, 10 .^ (0:.2:3.9))))
    p1 = @pgf Axis({ymode = "log", xmode = "log", ymin = 1e-11, ymax = 1e2,
                    width = "5cm", ylabel = "MSE", xlabel = "step"},
                   Plot({no_marks, thick},
                        Coordinates([i[1]; [mean(i[subsample[j]+1:subsample[j+1]])
                                            for j in 1:length(subsample)-1]],
                                    l[subsample])))
    p2 = plot_traj(t, subsample)
    [raw"\begin{center}",
     "seed = $seed, n = $n, final loss = $(l[end])",
     raw"\end{center}",
     raw"\begin{tabular}{cc}",
     TikzPicture(p1), " & ", TikzPicture(p2),
     raw"\end{tabular}"]
end

function maximal_cosine_similarity(x, ys)
    i = argmax([abs(cosine_similarity(x, y)) for y in ys])
    cosine_similarity(x, ys[i])
end
cosine_similarity(x, y) = clamp(dot(x, y)/(norm(x) * norm(y)), -1., 1.)
weight_vectors(coords::Tuple) = weight_vectors(coords[1:2]...)
weight_vectors(coordsx, coordsy) = [[x, y] for (x, y) in zip(coordsx, coordsy)]
function alignment(traces, targetcoords)
    wt = weight_vectors(targetcoords)
    final_params = last.(traces)
    ws = weight_vectors.(final_params)
    maximal_cosine_similarities = [maximal_cosine_similarity.(w, Ref(wt))
                                   for w in ws]
    angles = [180/π * acos.(c) for c in maximal_cosine_similarities]
    a = last.(final_params)
    n = [norm.(w) for w in ws]
    (angles = angles,
     secondlayerweights = a,
     firstlayernorms = n)
end
function plot_alignment(traces, targetcoords)
    phi, a, n = alignment(traces, targetcoords)
    @pgf PolarAxis({only_marks, ymax = 1.5, xmin = -18, xmax = 18},
                   Plot(Coordinates(vcat(phi...), abs.(vcat(a...)))))
end

function cluster(x; th = 1e-1)
    clusters = Vector{Int}[]
    vals = Vector{Float64}[]
    for (i, v) in enumerate(x)
        newcluster = true
        for (c, cv) in enumerate(vals)
            if minimum(abs.(cv .- v)) < th
                push!(clusters[c], i)
                push!(vals[c], v)
                newcluster = false
                break
            end
        end
        if newcluster
            push!(clusters, [i])
            push!(vals, [v])
        end
    end
    (clusters, vals)
end
function categorize(phi, a; tha = 1e-10, thphi = 1e-3)
    zeroout = abs.(a) .< tha
    zerophi = phi .< thphi
    nzeroout = sum(zeroout)
    nzerophi = sum(zerophi)
    uncat = .!(zeroout .| zerophi)
    nonzerophi = phi[uncat]
    idxs, vals = cluster(nonzerophi, th = thphi)
    (nzeroout = nzeroout, nzerophi = nzerophi,
     uncat = uncat,
     sumuncat = [sum(a[uncat][idx]) .< tha for idx in idxs],
     stdphiuncat = std.(vals),
     zeroout = zeroout, zerophi = zerophi,
     nonzerophi = nonzerophi,
     nonzeroclusters = idxs,
     sa = round(sum(a), sigdigits = 3))
end


end # module
