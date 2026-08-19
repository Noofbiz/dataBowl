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

## Statistical Mechanics

How it works:

  1. Index training data (lazy — only game_ids scanned at startup).
  2. Stream every training game through GameIter; extract empirical
     acceleration distributions into a ForceTable, grouped by
     (player_role, speed_bucket, dir_bucket).
  3. Load test rows from test_input.csv + submission template.
  4. For each test row, run -runs independent particle simulations
     starting from the player's last observed kinematic state.
  5. Take the mean trajectory and read off the position at output frame N
     (frame_id/num_frames_output encodes which future step is requested).
  6. Write out/submission.csv with columns id,x,y.

## Machine Learning Models

For Machine Learning, I built a Multi-Layer Perception (MLP) model to make
predictions about where the player is going based on previous frames.

### Architecture

	Input  [N, InputFeatureLen=9]
	   │
	Dense(→hiddenSizes[0]) + ReLU
	Dense(→hiddenSizes[1]) + ReLU
	   ⋮
	Dense(→hiddenSizes[-1]) + ReLU
	Dense(→2)               (linear output; predicts x and y)
 Loss:      Mean Squared Error
 Optimizer: Adam

## Results

To compare the results, I wrote a command to compare the two methods. It
uses the dataset to train a model using the `ml` package as well as runs a
montecarlo simulation using the `simulate` package. In order to gauge each
model's accuracy, I set aside some of the training data for use as 
test data. I then was able to use that to compare not only each result
against eachother, but the results against a separately known value.

I ran my comparison command using

```
$ go run ./cmd/compare/ ... -epochs 10 -infer-batch 256 -runs 100
```

```
=== Prediction Accuracy Comparison (Euclidean distance, yards) ===

Method                N       MAE      RMSE     Median
------                -       ---      ----     ------
ML (MLP)      [all]   31637   3.7473   4.7710   2.9560
Monte Carlo   [all]   31637   1.1727   1.9834   0.5900

--- Early frames (1–5) ---
ML (MLP)      13690   2.8791   3.2652   2.6428
Monte Carlo   13690   0.2289   0.3334   0.1408

--- Mid frames (6–15) ---
ML (MLP)      15119   3.2715   3.8331   2.8671
Monte Carlo   15119   1.5507   1.9963   1.2032

--- Late frames (16+) ---
ML (MLP)      2828   10.4933   11.1572   10.1764
Monte Carlo   2828   3.7207    4.7080    3.1274

Winner (by MAE):
  All     Monte Carlo wins (ΔMAE = 2.5746 yds)
  Early   Monte Carlo wins (ΔMAE = 2.6502 yds)
  Mid     Monte Carlo wins (ΔMAE = 1.7208 yds)
  Late    Monte Carlo wins (ΔMAE = 6.7727 yds)
```

This shows that using only 10 epochs and a simple machine learning method,
Monte Carlo makes better predictions.

Let's see what happens when I crank it up to 40 epochs.

```
=== Prediction Accuracy Comparison (Euclidean distance, yards) ===

Method                N       MAE      RMSE     Median
------                -       ---      ----     ------
ML (MLP)      [all]   31637   3.7473   4.7710   2.9560
Monte Carlo   [all]   31637   1.1727   1.9834   0.5900

--- Early frames (1–5) ---
ML (MLP)      13690   2.8791   3.2652   2.6428
Monte Carlo   13690   0.2289   0.3334   0.1408

--- Mid frames (6–15) ---
ML (MLP)      15119   3.2715   3.8331   2.8671
Monte Carlo   15119   1.5507   1.9963   1.2032

--- Late frames (16+) ---
ML (MLP)      2828   10.4933   11.1572   10.1764
Monte Carlo   2828   3.7207    4.7080    3.1274

Winner (by MAE):
  All     Monte Carlo wins (ΔMAE = 2.5746 yds)
  Early   Monte Carlo wins (ΔMAE = 2.6502 yds)
  Mid     Monte Carlo wins (ΔMAE = 1.7208 yds)
  Late    Monte Carlo wins (ΔMAE = 6.7727 yds)
```

This took 20 minutes to calculate and had very little improvement. Monte
Carlo appears to be winning every metric by a bit.
