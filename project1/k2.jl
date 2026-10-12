"""
textbook implementation of k2 search
"""

struct K2Search
    ordering::Vector{Int}
    max_parents::Int
end

K2Search(ordering::Vector{Int}) = K2Search(ordering, 5)

function fit(method::K2Search, vars, D)
    n = length(vars)
    G = SimpleDiGraph(n)

    for (k, i) in enumerate(method.ordering[2:end])
        # i is the variable we are choosing parents for
        # k is how many variables come before it in the ordering
        y = bayesian_score_i(vars, G, D, i)

        while length(inneighbors(G, i)) < method.max_parents
            y_best, j_best = -Inf, 0

            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y′ = bayesian_score_i(vars, G, D, i)

                    if y′ > y_best
                        y_best, j_best = y′, j
                    end

                    rem_edge!(G, j, i)
                end
            end

            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
    end

    return G
end
