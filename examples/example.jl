using TitanicClassifier
using CSV
using DataFrames

train = CSV.File("data/train_modified.csv") |> Tables.matrix
test = CSV.File("data/test_modified.csv") |> Tables.matrix

X_train = train[:, 1:size(train)[2]-1]
Y_train = train[:, size(train)[2]]

X_test = test[:, 1:size(train)[2]-1]
Y_test = test[:, size(train)[2]]

Y_pred = classifyKNN(X_test, X_train, Y_train; k=7, distance = l2_distance)
acc = compute_accuracy(Y_test, Y_pred)
println("Accuracy of 7-NN: $acc")

tree = build_tree(X_train, Y_train; max_depth=3)
Y_pred = classifyDT(X_test, tree)
acc = compute_accuracy(Y_test, Y_pred)
println("Accuracy of DT: $acc")

