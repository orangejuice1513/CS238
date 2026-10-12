"""
project1.jl: script to compute a graph based on the data

written by: julia jiang for CS238
"""

using Graphs
using Printf
using CSV
using DataFrames
using SpecialFunctions

include("score.jl")
include("parse_data.jl")
include("k2.jl")

function write_gph(dag::DiGraph, idx2names, filename)
    """
    input: dict of index to names 
    output: filename in gph form 
    """
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function compute(infile, outfile)
    """
    computes the graph using k2 algo and returns the graph and score 
    """
    vars, D, idx2names = load_data(infile)
    n = length(vars)
    method = K2Search(collect(1:n))

    seconds = @elapsed G = fit(method, vars, D) # runtime 

    write_gph(G, idx2names, outfile) # made the gph file 
    score = bayesian_score(vars, G, D) # score 

    G_baseline = SimpleDiGraph(n) # baseline is empty edge graph 
    baseline_score = bayesian_score(vars, G_baseline, D)
    @printf("Empty-edge baseline score for %s: %.10f\n", infile, baseline_score)
    if score > baseline_score
        println("huzzah")
    else
        println("try again u didn't beat the baseline")
    end
    
    @printf("Bayesian score for %s: %.10f\n", infile, score)
    @printf("K2 runtime: %.5f seconds\n", seconds)
    return G, score
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in ("small", "medium", "large")
        infile = joinpath(@__DIR__, "data", "$name.csv")
        outfile = joinpath(@__DIR__, "$name.gph")
        compute(infile, outfile)
    end
end
