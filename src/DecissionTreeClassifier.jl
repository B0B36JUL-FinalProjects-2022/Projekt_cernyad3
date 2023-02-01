export classifyDT, get_leaf_value, split_tree, get_best_split, predict, build_tree, information_gain

"""
    Node
    
Struct to represent a tree node.

# Fields
- `left`: Left subtree, nothing if this is a leaf node
- `right`: Right subtree, nothing if this is a leaf node
- `feature_index`: Index of feature that was used for split, -1 if this is a leaf node
- `threshold`: Threshold that was used for split, -1 if this is a leaf node
- `ig`: Information gain of the split, -1 if this is a leaf node
- `value`: Value of the node, only used for leaf nodes
"""
struct Node
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    feature_index::Int64
    threshold::Float64
    ig::Float64
    value::Union{Float64, Nothing}
end

leaf_node(value) = Node(nothing, nothing, -1, -1, -1, value)

"""
    build_tree(X::Matrix{<:Real}, y::Vector{<:Integer}, depth::Integer; min_samples::Integer=3, max_depth::Integer=typemax(Int))

Build a decission tree that maximizes information gain based on data `X` and labels `y`.
Return a `Node` stuct representing the decission tree. 

# Arguments
- `X`: Matrix of data. (num_samples, num_features)
- `y`: Vector of training labels. Shape (num_samples,)
- `depth`: Current depth, 0 if not specified
- `min_samples`: Minimal number of samples in a split to split further
- `max_depth`: Maximal depth of returned decission tree

See also `Node`
"""
function build_tree(X::Matrix{<:Real}, y::Vector{<:Integer}, depth::Integer; min_samples::Integer=3, max_depth::Integer=typemax(Int))
    sample_count, feature_count = size(X)

    if !(sample_count >= min_samples && depth <= max_depth)
        return leaf_node(get_leaf_value(y))
    end

    best_split = get_best_split(X, y, feature_count)

    if get(best_split, "ig", 0)  > 0
        left = build_tree(best_split["X_left"], best_split["y_left"], depth+1; min_samples=min_samples, max_depth=max_depth)
        right = build_tree(best_split["X_right"], best_split["y_right"], depth+1; min_samples=min_samples, max_depth=max_depth)
        return Node(left, right, best_split["feature_index"], best_split["threshold"], best_split["ig"], nothing)
    end
    
    return leaf_node(get_leaf_value(y))

end

build_tree(X::Matrix{<:Real}, y::Vector{<:Integer}; kwargs...) = build_tree(X, y, 0; kwargs...)


"""
    get_best_split(X::Matrix{<:Real}, y::Vector{<:Integer}, feature_count::Integer)

Split tree in the best way possible to maximize information gain. 
Return a dictionary with parameters of above mentioned split.
The dicitonary can be empty in case a split could not be made.
# Dictionary fields
- `ret["feature_index"]`: index of the feature that was used for split
- `ret["threshold"]`: threshold value used for split
- `ret["X_left"]`: matrix of data in the left subtree
- `ret["X_right"]`: matrix of data in the right subtree
- `ret["y_left"]`: vector of labels in the left subtree
- `ret["y_right"]`: vector of labels in the right subtree
- `ret["ig"]`: information gain achieved by best split

# Arguments
- `X`: Matrix of data. (num_samples, num_features)
- `y`: Vector of training labels. Shape (num_samples,)
- `feature_count`: Number of features

See also `split_tree`
"""
function get_best_split(X::Matrix{<:Real}, y::Vector{<:Integer}, feature_count::Integer)
    ret = Dict()
    max_ig = -Inf

    for feature_index in 1:feature_count
        feature_values = X[:, feature_index]
        possible_thresholds = unique(feature_values)

        for t in possible_thresholds
            left, right = split_tree(hcat(X,y), feature_index, t)

            if length(left) == 0 || length(right) == 0
                continue
            end
            

            y_left, y_right = left[:, size(left)[2]], right[:, size(right)[2]]
            X_left, X_right = left[:, 1:size(left)[2]-1], right[:, 1:size(right)[2]-1]
            ig = information_gain(y, y_left, y_right)
            
            

            if ig > max_ig
                max_ig = ig
                ret["feature_index"] = feature_index
                ret["threshold"] = t
                ret["X_left"] = X_left
                ret["X_right"] = X_right
                ret["y_left"] = y_left
                ret["y_right"] = y_right
                ret["ig"] = ig
            end
        end    
    end    

    return ret

end

"""
    get_leaf_value(y::Vector{<:Integer})

Get leaf value based on labels `y`. Return most frequent label from y.

# Arguments
- `y`: Vector of training labels. Shape (num_samples,)
"""
function get_leaf_value(y::Vector{<:Integer})
    classes = unique(y)
    max_count = -1
    ret = 0
    for c in classes
        n = count(i->(i == c), y)
        if n > max_count
            max_count = n
            ret = c
        end
    end

    return ret
end

"""
    split_tree(X::Matrix{<:Real}, feature_index::Integer, threshold::Real)

Split tree by feature at `feature_index` and value of `threshold`. 
Return tuple of matices (X_left, X_right).

# Arguments
- `X`: Vector of training labels. Shape (num_samples,)
- `feature_index`: index of the feature used for split
- `threshold`: threshold value used for split
"""
function split_tree(X::Matrix{<:Real}, feature_index::Integer, threshold::Real)
    left = X[ X[:, feature_index] .<= threshold, :]
    right = X[ X[:, feature_index] .> threshold, :]
    return left, right
end

"""
    information_gain(y::Vector{<:Integer}, y_left::Vector{<:Integer}, y_right::Vector{<:Integer})

Compute information gain given vectors of labels `y`, `y_left` and `y_right`.
Return information gain computed using information entropy.

# Arguments
- `y`: Vector of all training labels. Shape (num_samples,)
- `y_left`: Vector of  training labels in left subtree. Shape (num_samples,)
- `y_right`: Vector of training labels in right subtree. Shape (num_samples,)
"""
function information_gain(y::Vector{<:Integer}, y_left::Vector{<:Integer}, y_right::Vector{<:Integer})
    w_l, w_r = length(y_left) / length(y), length(y_right) / length(y)
    return entropy(y) - w_l*entropy(y_left) - w_r*entropy(y_right)
end


function entropy(y::Vector{<:Integer})
    e = 0
    for c in unique(y)
        
        ratio = sum(Int64.(y.==c)) / length(y)
        e -= ratio * log2(ratio)
    end
    return e
    
end

"""
    predict(x::Vector{<:Real}, tree::Node)

Classify single datapoint `x` using decission tree `tree`
Return predicted label for datapoint `x`.

# Arguments
- `x`: Datapoint to classify (num_features,)
- `tree`: Decission tree to use for classification
"""
function predict(x::Vector{<:Real}, tree::Node)
    if !isnothing(tree.value)
        return tree.value    
    end

    val = x[tree.feature_index]

    if val <= tree.threshold
        return predict(x, tree.left)
    else
        return predict(x, tree.right)
    end    

end

function print_tree(tree::Node, indent::String)

    if !isnothing(tree.value)
        println(tree.value)
    else
        println("X_$(tree.feature_index)<=$(tree.threshold)?$(tree.ig)")
        print("$(indent)left:")
        print_tree(tree.left, indent * indent)
        print("$(indent)right:")
        print_tree(tree.right, indent * indent)
    end

end

"""
    classifyDT(X_test::Matrix{<:Real}, tree::Node)

Classify a matrix of data `X` using decission tree `tree`
Return vector of predicted labels for data `X`.

# Arguments
- `X`: Matrix of data to classify (num_samples, num_features)
- `tree`: Decission tree to use for classification
"""
function classifyDT(X_test::Matrix{<:Real}, tree::Node)
    return mapslices(x -> predict(x, tree), X_test; dims=2)
end
