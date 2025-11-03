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

## Models

The `models` folder contains models used to make predictions for the NFL Big Data
Bowl 2026. The goal of this project is given an initial set of conditions from 
when the ball is thrown, predict the way the play is going to go. The initial 
conditions will include player stats, previous runs with similar setups, etc.
as well as the play-by-play data for the contest. `datasets` has a good 


## Stochastically

The `monte` folder contains methods for turning the data into statistics usable
in montecarlo simulations of the plays. This isn't part of the contest but it
is just something I want to see.

## Front-End

I want to write a front-end in for this that will show a little video
animation of the predicted plays.

