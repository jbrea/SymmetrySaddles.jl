using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DrWatson, SymmetrySaddles, DataFrames, Flux
import SymmetrySaddles: alignment, categorize

results = collect_results(datadir("simulations"))
filenames = results.path .|> splitpath .|> last

resultsg = results[(map(x -> match(r"activation=g_", x) !== nothing,
                        filenames)), :]
g(x) = 1/(1 + exp(-4x)) + log(exp(x) + 1)
target = default_target(activation = g)

tmp = resultsg[resultsg.n .== 10, :]
@assert sum(last.(tmp.loss) .> 10^-20) == 0 "Not all students have loss < 10^-20"
phi, a, n = alignment(tmp.trace, target.coords);
res10 = categorize.(phi, a, thphi = 1, tha = 1e-6)
function tally(x)
    u = unique(x)
    [(length(findall(==(v), x)), v) for v in u]
end
cats10 = [[x.nzeroout > 0 ? (x.nzeroout, 1) : Tuple{Int, Int}[]; tally(length.(x.nonzeroclusters)); (x.nzerophi, 0)] for x in res10]
vcats10 = vcat(cats10...)
ucat = unique(last.(vcats10))
counts = [(v => (v == 0 ? 1 : v ) * sum(first.(vcats10[findall(x -> x[2] == v, vcats10)]))) for v in ucat]
sort!(counts)
println("copies: $(last(counts[1]))")
for (k, n) in counts[2:end]
    println("0-type with $(k-1) partner (group size $k): $n")
end
