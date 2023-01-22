struct Node
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    feature_index::Int64
    threshold::Float64
    ig::Float64
    value::Union{Float64, Nothing}
end

leaf_node(value) = Node(nothing, nothing, -1, -1, -1, value)


function build_tree(X, y; depth=0, min_samples=3, max_depth=3)
    sample_count, feature_count = size(X)

    if !(sample_count >= min_samples && depth <= max_depth)
        return leaf_node(get_leaf_value(y))
    end

    best_split = get_best_split(X, y, feature_count)

    if best_split["ig"] > 0
        left = build_tree(best_split["X_left"], best_split["y_left"]; depth=depth+1)
        right = build_tree(best_split["X_right"], best_split["y_right"]; depth=depth+1)
        return Node(left, right, best_split["feature_index"], best_split["threshold"], best_split["ig"], nothing)
    end
    
    return leaf_node(get_leaf_value(y))

end

function get_best_split(X, y, feature_count)
    ret = Dict()
    max_ig = -Inf

    for feature_index in 1:feature_count
        feature_values = X[:, feature_index]
        possible_thresholds = unique(feature_values)

        for t in possible_thresholds
            left, right = split(hcat(X,y), feature_index, t)

            if length(left) == 0 || length(right) == 0
                continue
            end
            

            y_left, y_right = left[:, size(left)[2]], right[:, size(right)[2]]
            X_left, X_right = left[:, 1:size(left)[2]-1], right[:, 1:size(right)[2]-1]
            ig = information_gain(y, y_left, y_right)
            println(ig)
            

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

function get_leaf_value(y)
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

function split(X, feature_index, threshold)
    left = X[ X[:, feature_index] .<= threshold, :]
    right = X[ X[:, feature_index] .> threshold, :]
    return left, right
end

function information_gain(y, y_left, y_right)
    w_l, w_r = length(y_left) / length(y), length(y_right) / length(y)
    return entropy(y) - w_l*entropy(y_left) - w_r*entropy(y_right)
end


function entropy(y)
    e = 0
    for c in unique(y)
        
        ratio = sum(Int64.(y.==c)) / length(y)
        e -= ratio * log2(ratio)
    end
    return e
    
end

function predict(x, tree::Node)
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

function print_tree(tree, indent)

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
