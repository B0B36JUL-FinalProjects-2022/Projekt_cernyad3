export compute_accuracy

function compute_accuracy(labels, predictions)
    return sum(labels .== predictions) / length(labels)
end