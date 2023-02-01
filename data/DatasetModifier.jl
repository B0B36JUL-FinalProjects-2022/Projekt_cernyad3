using DataFrames
using CSV
using Statistics

function get_bins(data; bincount=4)
    binsize = (maximum(data) - minimum(data)) / bincount
    bins = [x*binsize for x in 0:bincount]
    bins[bincount+1] = Inf64  
    return bins
end

function bin_equally!(df, colname; bincount=4)
    bins = get_bins(df[!, colname]; bincount=bincount)
    for i in 1:bincount
        replace!(x-> (x>=bins[i] && x<bins[i+1]) ? i-1 : x, df[!, colname])
    end
end

function get_quntile(x; n=4)
    q = quantile(x, LinRange(0, 1, n + 1))
    return map(v -> min(searchsortedlast(q, v)-1, n-1), x)
end

function replace_missing_with_median!(df, colname)
    med = median(collect(skipmissing(df[!, colname])))
    replace!(df[!, colname], missing => med);
    df[!, colname] = convert.(Float64, df[!, colname])
end

function convert_categorical_to_int!(df, colname, categories)

    for i in 1:length(categories)
        replace!(df[!, colname], categories[i] => string(i));
    end

    replace!(df[!, colname], missing => get_most_frequent(df, colname));
    df[!,colname] = parse.(Int64, df[!,colname])

    
end

function get_most_frequent(df, colname)
    counts_df = combine(groupby(df, [colname]), nrow => :count)
    idx = sortperm(counts_df, :count, rev=true)[1]
    return counts_df[idx, colname]
end




train_data = CSV.read("data/train.csv", DataFrame)
test_data = CSV.read("data/test.csv", DataFrame)
datasets = [train_data, test_data]



for data in datasets
    # ADD new features
    data[!, :Title] = String.((x->x[1]).(split.((x->x[2]).(split.(data[!, :Name], ", ")), ".")))
    replace!(x-> (x!="Mr" && x!="Miss" && x!="Mrs" && x!="Master") ? "Misc" : x, data.Title)
    data[!,:FamilySize] = data[!,:SibSp] .+ data[!,:Parch] .+ 1
    data[!, :IsAlone] = Int.(data[!, :FamilySize] .== 1)

    # deal with missing values
    replace_missing_with_median!(data, "Age")
    replace_missing_with_median!(data, "Fare")

    # deal with categorical features
    convert_categorical_to_int!(data, "Embarked", unique(data.Embarked))
    convert_categorical_to_int!(data, "Sex", unique(data.Sex))
    convert_categorical_to_int!(data, "Title", unique(data.Title))
    
    # BIN data
    bin_equally!(data, "Age"; bincount=5)
    data[!, :Age] = Int.(data[!, :Age])
    data[!, :Fare] = Int.(get_quntile(data[!, :Fare]))
    
    # DROP useless cols
    select!(data, Not(:Ticket))
    select!(data, Not(:Cabin))
    select!(data, Not(:Name))
    select!(data, Not(:PassengerId))
end


# push Survived to the end (so that the labels are the last col)
tmp = train_data[!, :Survived]
select!(train_data, Not(:Survived))
train_data[!,:Survived] = tmp

print(datasets)

CSV.write("data/train_modified.csv", datasets[1])

# add ground truth to test data for simpler testing
ground_truth = CSV.read("data/ground_truth.csv", DataFrame)
datasets[2][!, :Survived] = ground_truth[!, :Survived]
CSV.write("data/test_modified.csv", datasets[2])

