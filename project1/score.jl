"""
functions to calculate bayesian score from textbook 
"""

# --------- getting the right data structures ------ #

struct Variable
    name::Symbol # variable name 
    r::Int  # number of possible values it can take on 
end

function n_parent_instantiations(vars, G, i)
    """
        returns the number of parent combinations, i.e.
        how many rows should the count table for node i have 

        args: 
            - vars: variables
            - G: the graph we are tryna calculate the score for 
            - i: the node we are on rn 
    """
    parents = inneighbors(G, i)
    return isempty(parents) ? 1 : prod(vars[j].r for j in parents)
end

function parental_index(vars, G, o, i)
    """
    converts the parents' values for one observation into a row number 
    args: 
        - vars: variables 
        - G: the graph we are tryna calculate the score for 
        - o: observation vector
        - i: the node we are on rn 
    return: the row number 
    """
    parents = inneighbors(G, i)
    j = 1
    for (position, node) in enumerate(parents)
        combs = 1 # how many combinations come b4 this parent's value 
        for p in 1:(p - 1) # loop over positions b4 current parent 
            combs *= vars[parents[p]].r
        end
        j += (o[node] - 1) * combs
    end
    return j 
end

function count_matrix_i(vars, G, D::AbstractMatrix{Int}, i::Int)
    """
    makes count matrix for variable i 
    """
    num_rows = n_parent_instantiations(vars, G, i) 
    num_cols = vars[i].r
    Mi = zeros(num_rows, num_cols)
    for o in eachcol(D) # loop thru observation cols 
        Mi[parental_index(vars, G, o, i), o[i]] += 1.0
    end
    return Mi
end

function count_matrices(vars, G, D::AbstractMatrix{Int})
    """
    count data for every variable 
    returns a list of count matrices (1 for each var)
    """
    n = length(vars)
    return [count_matrix_i(vars, G, D, i) for i in 1:n]
end

function prior_i(vars, G, i)
    """
    the uniform dirichlet pseudocounts for the ith var (matrix of all 1s :P)
    """
    return ones(n_parent_instantiations(vars, G, i), vars[i].r)
end

function prior(vars, G)
    """
    all da uniform dirichlet pseudocounts in a list 
    """
    n = length(vars)
    return [prior_i(vars, G, i) for i in 1:n]
end

# --------- score calculation ------- # 
function bayesian_score_component(M, alpha)
    p = sum(loggamma.(alpha + M))
    p -= sum(loggamma.(alpha))
    p += sum(loggamma.(sum(alpha, dims=2)))
    p -= sum(loggamma.(sum(alpha, dims=2) + sum(M, dims=2)))
    return p
end

function bayesian_score_i(vars, G, D, i)
    return bayesian_score_component(statistics_i(vars, G, D, i), prior_i(vars, G, i))
end

function bayesian_score(vars, G, D)
    n = length(vars)
    return sum(bayesian_score_i(vars, G, D, i) for i in 1:n)
end
