package ml

import (
	"fmt"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
)

// TrainEpochs trains the model for the requested number of epochs over ds,
// printing the mean loss at the end of each epoch.
//
// One epoch iterates once through every game in ds (files are loaded lazily on
// first access). Each game becomes one training step whose batch contains all
// (player, output_frame) prediction targets for that game.
func (m *PlayerPositionModel) TrainEpochs(ds *datasets.PredictionDataset, epochs int) error {
	loop := train.NewLoop(m.trainer)

	// Print the moving-average loss (metrics[1]) at the end of every epoch.
	loop.OnEnd("log-loss", train.Priority(0), func(l *train.Loop, metrics []*tensors.Tensor) error {
		if len(metrics) > 1 {
			fmt.Printf("epoch %d  mean-loss=%.6f\n", l.Epoch, metrics[1].Value().(float32))
		}
		return nil
	})

	if _, err := loop.RunEpochs(ds, epochs); err != nil {
		return fmt.Errorf("training loop: %w", err)
	}
	return nil
}
