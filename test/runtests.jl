using TitanicClassifier
using Test

@testset "TitanicClassifier.jl" begin
    data = [1 2 0;
        1.5 2.5 0;
        3 4 1; 
        1 5 0; 
        1 3 1; 
        3.5 5 1]

    y = data[:, 3]
    X = data[:, 1:2]
    @testset "KnnClassifier.jl" begin
        

        @test classify_point_KNN([0 0], 3, X, y; distance=l2_distance) == 0
        @test classify_point_KNN([4 4], 3, X, y; distance=l2_distance) == 1
        @test classifyKNN([0 0; 4 4], X, y) == [0 1]'
        
        @test l2_distance([1; 1], [1; 1]) == 0
        @test l2_distance([3; 1], [1; 1]) == 2
        @test l1_distance([3; 1], [5; 2]) == 3

    end

    @testset "DecissionTreeClassifier.jl" begin
        @test get_leaf_value([1 0 1 1 1]) == 1
        @test get_leaf_value([0 1]) == 0
    end

    @testset "ClassificationUtils.jl" begin
        predictions = [1 1 0 1 0]
        labels = [1 1 1 1 1]
        @test compute_accuracy(labels, predictions) == 0.6

        predictions = [1 1]
        labels = [1 1]
        @test compute_accuracy(labels, predictions) == 1

        @test split_tree(X, 2, 3.5) == ([1 2; 1.5 2.5; 1 3], [3 4; 1 5; 3.5 5])
        best_split = get_best_split(X, y, size(X)[2])
        @test best_split["feature_index"] == 1
        @test best_split["threshold"] == 1.5

        test_tree = build_tree(X, y)
        @test predict([1 1], test_tree) == 0
        @test classifyDT([0 0; 1 1], test_tree) == [0 0]'

        @test information_gain([1 1 1 1], [1], [1 1 1]) < information_gain([1 1 0 0], [1 1], [0 0])
        
    end

end
