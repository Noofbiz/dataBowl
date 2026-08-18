# Big Data Bowl 2026
my code for the [data bowl competitions](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

This repo holds the code for my submissions to the NFL Big Data Bowl 2026. This
will have multiple libraries related to the data bowl.

## Datasets

The `datasets` folder contains `train.Dataset` implementations of the data for the
bowl as well as selected datsets from the nflverse I used in my models. These
datasets are open source under the MIT license and can be used by anyone to
build their own models  using gomlx. 

### Data Pipeline
The data is in csv files available on the kaggle competition page 
for the [prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
as well as the [analytical](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data) data sets. They should be downloaded and placed in separate
folders in `datasets/assets`.

Those CSV files are then used to create a `train.Dataset` of the data. 
This requires implementing a few functions, but I also want to make
it compatible with my Monte Carlo simulations, that way I don't have to read
or write the data more than once. The datasets are pretty big, so I
load it in batches, saving a `gob` file to disc. This way the dataset
only needs to be read once, and the data set can be accessed as needed
rather than take up space in-imemory.

### Prediction Data

The goal for the prediction contest is to be able to predict the outcome
of a play based on the first few frames of that play. The plays are only
passing plays, and the frames given are up until the ball is in the air.
Once the ball is thrown, the data stops and the prediction model takes
over. 

The prediction data is lazily loaded from the csv files one game
at a time. The dataset produces both the go struct for the games and
plays as well as a flat tensor to use with machine learning models.
This will allow me to use the dataset for both the montecarlo simulations
as well as the machine learning of gomlx.

### Analytics Data

TBA

## Statistical Mechanics



## Machine Learning Models



## Front-End

cmd/compare - A command line tool to compare different models and their
predictions on the same data. This can be used to see how different models
perform on the same plays and to compare their predictions. It runs the simple
model and also the montecarlo simulations and outputs the results to an output
file for further analysis.
