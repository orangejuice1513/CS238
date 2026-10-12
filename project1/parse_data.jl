"""
load CSVs and .gph files for structure learning 
"""

function load_data(path)
    """
    takes in the csv file 
    returns:
        - vector of the variables 
        - data matrix (variables x observations)
        - map from variable index to its name 
    """
    df = CSV.read(path, DataFrame)
    colnames = names(df)
    D = permutedims(Matrix{Int}(df))  # variables × observations : turn into ints then transpose 
    num_vars = size(D, 1)  
    vars = [Variable(Symbol(colnames[i]), maximum(D[i, :])) for i in 1:num_vars]
    idx2names = Dict(i => String(colnames[i]) for i in 1:num_vars) # idx2names[1] = "age"
    return vars, D, idx2names
end

function graph_from_gph(path, idx2names)
    """
    takes in .gph and turns it into graph object 
    """
    name2idx = Dict(name => i for (i, name) in idx2names)
    G = SimpleDiGraph(length(idx2names))
    for line in eachline(path) # add the nodes from the .gph file 
        s = strip(line) 
        isempty(s) && continue 
        a, b = split(s, ',') 
        add_edge!(G, name2idx[strip(String(a))], name2idx[strip(String(b))])
    end
    return G 
end
