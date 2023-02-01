using LinearAlgebra

export classifyKNN, classify_point_KNN, l1_distance, l2_distance, cosine_distance

"""
    l1_distance(x::Vector{<:Real},y::Vector{<:Real})

Compute l1 distance between vectors `x` and `y`

# Arguments
- `x`: Vector x (length,)
- `y`: Vector y (length,)
"""
l1_distance(x::Vector{<:Real},y::Vector{<:Real}) = sum(abs.(x-y))

"""
    l2_distance(x::Vector{<:Real},y::Vector{<:Real})

Compute l2 distance between vectors `x` and `y`

# Arguments
- `x`: Vector x (length,)
- `y`: Vector y (length,)
"""
l2_distance(x::Vector{<:Real},y::Vector{<:Real}) = norm(x-y)

"""
    cosine_distance(x::Vector{<:Real},y::Vector{<:Real})

Compute cosine distance between vectors `x` and `y`

# Arguments
- `x`: Vector x (length,)
- `y`: Vector y (length,)
"""
cosine_distance(x::Vector{<:Real},y::Vector{<:Real}) = dot(x,y)/(norm(x)*norm(y))
    
"""
    getNN(x::Vector{<:Real}, k::Integer, data::Vector{<:Real}; distance = l2_distance)

Get indices of `k` nearest neighbours for to vector `x` from `data`

# Arguments
- `x`: Datapoint to classify
- `k`: Number of neighbours to return
- `data`: Training data
- `distance`: Distance function
"""
function getNN(x::Vector{<:Real}, k::Integer, data::Matrix{<:Real}; distance = l2_distance)
    return sortperm(vec(mapslices(z -> distance(vec(x), vec(z)), data, dims=2)))[1:k]
end

"""
    classify_point_KNN(x::Vector{<:Real}, k::Integer, data::Matrix{<:Real}, labels::Vector{<:Integer}; kwargs...)

Classify point `x` using `k` nearest neighbours from `data`. Return label for `x`

# Arguments
- `x`: Datapoint to classify
- `k`: Number of neighbours to return
- `data`: Training data
"""
function classify_point_KNN(x::Vector{<:Real}, k::Integer, data::Matrix{<:Real}, labels::Vector{<:Integer}; kwargs...)
    nn_incidies = getNN(x, k, data; kwargs...)
    nn_labels = labels[nn_incidies]
    return Int64(sum(nn_labels) > k/2)
end

"""
    classifyKNN(X_test, X_train, y_train; k=3, kwargs...)

Classify matrix of data `X_test` using `k` nearest neighbours from `X_train`. Return vector of labels for `X`.

# Arguments
- `x`: Datapoint to classify
- `k`: Number of neighbours to return
- `data`: Training data
"""
function classifyKNN(X_test::Matrix{<:Real}, X_train::Matrix{<:Real}, y_train::Vector{<:Integer}; k::Integer=3, kwargs...)
    return mapslices(x -> classify_point_KNN(x, k, X_train, y_train; kwargs...), X_test; dims=2)
end

