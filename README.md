# TitanicClassifier

This project implements KNN-classifier and decission tree
classifier and demonstrates their usage on Kaggle's Titanic dataset.

## Installation

The package can be installed as follows:
```julia
(@v1.4) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3
```

## Project structure
### Package code
[src/KnnClassifier.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/src/KnnClassifier.jl) contains all functions necessary to classify using KNN<br>
[src/DecissionTreeClassifier.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/src/DecissionTreeClassifier.jl) contains all functions necessary to classify using DT<br>
[src/ClassificationUtils.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/src/ClassificationUtils.jl) contains utility function for classification<br>
[src/TitanicClassifier.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/src/TitanicClassifier.jl) combines above mentioned code into a module

### Data analysis and feature engineering
[notebook/DataAnalysis.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/notebook/DataAnalysis.ipynb) contains basic data analysis that helped me understand the Titanic dataset<br>
[data/DatasetModifier.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/data/DatasetModifier.jl) contains all the code that I used to modify the Titanic dataset<br>

### Tests
[test/runtests.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/test/runtests.jl) contains several unit tests to show that the classifiers work properly
[notebook/TestDataVisualisation.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/notebook/TestDataVisualisation.ipynb) contains a visual representation of the data that I used in several unit tests<br>
### Examples
[examples/example.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_cernyad3/blob/main/examples/example.jl) contains an example that shows how the classifiers can be used on the Titanic dataset<br>



