using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DrWatson, SymmetrySaddles, PGFPlotsX, DataFrames, Flux, Statistics

results = collect_results(datadir("simulations"))
filenames = results.path .|> splitpath .|> last

trace1 = results[filenames .== "N=10_Ninner=100000_activation=sigmoid_gitcommit=v0.1.0_maxtime=1000_n=5_seed=35.bson", :trace][1]
trace2 = results[filenames .== "N=10_Ninner=100000_activation=sigmoid_gitcommit=v0.1.0_maxtime=1000_n=5_seed=38.bson", :trace][1]
trace3 = results[filenames .== "N=10_Ninner=100000_activation=sigmoid_gitcommit=v0.1.0_maxtime=1000_n=45_seed=1.bson", :trace][1]
pt1 = plot_traj(trace1, colorbar = false)
pt2 = plot_traj(trace2, colorbar = false)
pt3 = plot_traj(trace3, colorbar = true)
pgfsave(plotsdir("fig1a.tikz"), pt1)
pgfsave(plotsdir("fig1b.tikz"), pt2)
pgfsave(plotsdir("fig1c.tikz"), pt3)

function plot_loss(result; extract = last, ylabel = "loss at convergence")
    final_loss = transform(result, :loss => ByRow(extract) => :loss) |>
                 df -> select(df, :n, :loss) |> df -> groupby(df, :n)
    @pgf Axis({only_marks, ylabel = ylabel, xtick = [1, 2], ymode = "linear",
               xticklabels = union(result.n), xlabel = "width"},
              [Plot(Coordinates((1:length(r.loss))./(1.5*length(r.loss)) .+ i .- 1/3, r.loss))
               for (i, r) in enumerate(final_loss)])
end
pl1 = plot_loss(sort(results[results.activation .== :sigmoid, :], [:n, :seed]))
pgfsave(plotsdir("fig1d.tikz"), pl1)

pl2 = plot_loss(sort(results[(f -> match(r"layers=3", f) !== nothing).(filenames), :], [:n, :seed]))
pgfsave(plotsdir("fig1_3L.tikz"), pl2)

