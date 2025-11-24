# Big Data Bowl 2026
my code for the [data bowl competitions](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

This repo holds the code for my submissions to the NFL Big Data Bowl 2026. This
will have multiple libraries related to the data bowl.

## Datasets

The `datasets` folder contains `train.Dataset` implementations of the data for the
bowl as well as selected datsets from the nflverse I used in my models. These
datasets are open source under the MIT license and can be used by anyone to
build their own models  using gomlx. Each one corresponds to a different
dataset. They first check the assets folder for the data, either in a csv or
tarball. If the data is there, it loads it. If not it will automatically
download the data from the data url and then load it. The returned datasets
all implement train.Dataset and can be readily used to train models with gomlx.
The datasets are loaded lazily so they only take up memory when needed, and
also loading is parallelized where possible to speed up loading times and
cached as a gob file for faster future loads.

## Models

The `models` folder contains models used to make predictions for the NFL Big Data
Bowl 2026. The goal of this project is given an initial set of conditions from
when the ball is thrown, predict the way the play is going to go. The initial
conditions will include player stats, previous runs with similar setups, etc.
as well as the play-by-play data for the contest.
The model I used was from the gomlx library, the simple pure go model. It is a
feedforward neural network with relu activations. The model is trained to
minimize the mean squared error between the predicted and actual outcomes of the plays.
The models are trained using the datasets in the datasets folder. The models
are saved in the models folder as gob files and can be loaded and used to make
predictions. The models are trained using the gomlx library and can be
easily modified to try different architectures and hyperparameters.

## Stochastically

The `monte` folder contains methods for turning the data into statistics usable
in montecarlo simulations of the plays. This isn't part of the contest but it
is just something I want to see. The montecarlo simulations will take the
predicted outcomes from the models and use them to simulate the plays multiple
times to get a distribution of possible outcomes. This can be used to get a
better idea of the uncertainty in the predictions and to see how different
factors affect the outcomes. The simulations were implemented using pure go
and can be easily modified to try different simulation methods and parameters.
These simulations are run in paralel to speed up the process.

## Front-End

cmd/compare - A command line tool to compare different models and their
predictions on the same data. This can be used to see how different models
perform on the same plays and to compare their predictions. It runs the simple
model and also the montecarlo simulations and outputs the results to an output
file for further analysis.
