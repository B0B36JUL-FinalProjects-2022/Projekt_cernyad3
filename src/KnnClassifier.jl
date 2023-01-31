using LinearAlgebra

export classifyKNN, classify_point_KNN, l1_distance, l2_distance, cosine_distance

l1_distance(x,y) = sum(abs.(x-y))

l2_distance(x,y) = norm(x-y)

cosine_distance(x,y) = dot(x,y)/(norm(x)*norm(y))
    
function getNN(x, k, data; distance = l2_distance)
    return sortperm(vec(mapslices(z -> distance(vec(x), vec(z)), data, dims=2)))[1:k]
end

function classify_point_KNN(x, k, data, labels; kwargs...)
    nn_incidies = getNN(x, k, data; kwargs...)
    nn_labels = labels[nn_incidies]
    return Int64(sum(nn_labels) > k/2)
end

function classifyKNN(X_test, X_train, y_train; k=3, kwargs...)
    return mapslices(x -> classify_point_KNN(x, k, X_train, y_train; distance = l2_distance), X_test; dims=2)
end

