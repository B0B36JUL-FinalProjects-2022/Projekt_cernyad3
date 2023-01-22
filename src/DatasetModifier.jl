using DataFrames
using CSV
using Statistics


train_data = CSV.read("data/train.csv", DataFrame)
test_data = CSV.read("data/test.csv", DataFrame)
datasets = [train_data, test_data]

tmp = train_data[!, :Survived]
select!(train_data, Not(:Survived))
train_data[!,:Survived] = tmp

for data in datasets
    avg_age = mean(collect(skipmissing(data[!, "Age"])))
    replace!(data.Age, missing => avg_age);
    data[!,:Age] = convert.(Float64, data[!,:Age])

    avg_fare = mean(collect(skipmissing(data[!, "Fare"])))
    replace!(data.Fare, missing => avg_fare);
    data[!,:Fare] = convert.(Float64, data[!,:Fare])

    replace!(data.Embarked, "C" => "0");
    replace!(data.Embarked, "Q" => "1");
    replace!(data.Embarked, "S" => "2");
    replace!(data.Embarked, missing => "2");

    replace!(data.Sex, "male" => "0");
    replace!(data.Sex, "female" => "1");

    data[!,:Embarked] = parse.(Int64, data[!,:Embarked])
    data[!,:Sex] = parse.(Int64, data[!,:Sex])

    select!(data, Not(:Ticket))
    select!(data, Not(:Cabin))

    select!(data, Not(:Name))
    select!(data, Not(:PassengerId))
end

print(datasets)

CSV.write("data/train_modified.csv", datasets[1])


ground_truth = CSV.read("data/ground_truth.csv", DataFrame)
datasets[2][!, :Survived] = ground_truth[!, :Survived]
CSV.write("data/test_modified.csv", datasets[2])

