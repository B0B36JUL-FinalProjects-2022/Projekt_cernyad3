using DataFrames
using CSV
using Statistics

function get_bins(data; bincount=4)
    binsize = (maximum(data) - minimum(data)) / bincount
    bins = [x*binsize for x in 0:bincount]
    bins[bincount+1] = Inf64  
    return bins
end

function get_quntile(x; n=4)
    q = quantile(x, LinRange(0, 1, n + 1))
    return map(v -> min(searchsortedlast(q, v)-1, n-1), x)
end


train_data = CSV.read("data/train.csv", DataFrame)
test_data = CSV.read("data/test.csv", DataFrame)
datasets = [train_data, test_data]



for data in datasets
    avg_age = median(collect(skipmissing(data[!, "Age"])))
    replace!(data.Age, missing => avg_age);
    data[!,:Age] = convert.(Float64, data[!,:Age])

    avg_fare = median(collect(skipmissing(data[!, "Fare"])))
    replace!(data.Fare, missing => avg_fare);
    data[!,:Fare] = convert.(Float64, data[!,:Fare])

    replace!(data.Embarked, "C" => "0");
    replace!(data.Embarked, "Q" => "1");
    replace!(data.Embarked, "S" => "2");
    replace!(data.Embarked, missing => "2"); # most frequent

    replace!(data.Sex, "male" => "0");
    replace!(data.Sex, "female" => "1");

    

    data[!,:Embarked] = parse.(Int64, data[!,:Embarked])
    data[!,:Sex] = parse.(Int64, data[!,:Sex])

    data[!,:FamilySize] = data[!,:SibSp] .+ data[!,:Parch] .+ 1
    data[!, :IsAlone] = Int.(data[!, :FamilySize] .== 1)
    
    data[!, :Title] = String.((x->x[1]).(split.((x->x[2]).(split.(data[!, :Name], ", ")), ".")))
    replace!(x-> (x!="Mr" && x!="Miss" && x!="Mrs" && x!="Master") ? "Misc" : x, data.Title)
    replace!(data.Title, "Mr" => "0");
    replace!(data.Title, "Mrs" => "1");
    replace!(data.Title, "Miss" => "2");
    replace!(data.Title, "Master" => "3");
    replace!(data.Title, "Misc" => "4");
    data[!,:Title] = parse.(Int64, data[!,:Title])

    bincount = 5
    bins = get_bins(data[!, :Age]; bincount=5)
    for i in 1:bincount
        replace!(x-> (x>=bins[i] && x<bins[i+1]) ? i-1 : x, data.Age)
    end
    data[!, :Age] = Int.(data[!, :Age])


    data[!, :Fare] = Int.(get_quntile(data[!, :Fare]))
    
    
    
    

    select!(data, Not(:Ticket))
    select!(data, Not(:Cabin))
    select!(data, Not(:Name))
    select!(data, Not(:PassengerId))
end



tmp = train_data[!, :Survived]
select!(train_data, Not(:Survived))
train_data[!,:Survived] = tmp

print(datasets)

CSV.write("data/train_modified.csv", datasets[1])


ground_truth = CSV.read("data/ground_truth.csv", DataFrame)
datasets[2][!, :Survived] = ground_truth[!, :Survived]
CSV.write("data/test_modified.csv", datasets[2])

