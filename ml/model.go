// Package ml builds and trains a multi-layer perceptron (MLP) that predicts
// the future (x, y) position of an NFL player using the PredictionDataset from
// the datasets package.
//
// Architecture
//
//	Input  [N, InputFeatureLen=9]
//	   │
//	Dense(→hiddenSizes[0]) + ReLU
//	Dense(→hiddenSizes[1]) + ReLU
//	   ⋮
//	Dense(→hiddenSizes[-1]) + ReLU
//	Dense(→2)               (linear output; predicts x and y)
//
// Loss:      Mean Squared Error
// Optimizer: Adam
package ml

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/Noofbiz/dataBowl/datasets"
)

// Config holds the hyperparameters for the player position model.
type Config struct {
	// HiddenSizes is the width of each hidden layer in the MLP.
	// Defaults to [128, 64, 32] if nil.
	HiddenSizes []int

	// LearningRate for Adam. Defaults to 1e-3.
	LearningRate float64
}

func (c *Config) withDefaults() Config {
	out := *c
	if len(out.HiddenSizes) == 0 {
		out.HiddenSizes = []int{128, 64, 32}
	}
	if out.LearningRate == 0 {
		out.LearningRate = 1e-3
	}
	return out
}

// PlayerPositionModel wraps a gomlx model that predicts future player positions.
type PlayerPositionModel struct {
	cfg     Config
	backend compute.Backend
	store   *model.Store
	trainer *train.Trainer

	// inferExec is a compiled graph for prediction-only (no gradient tracking).
	inferExec *model.Exec
}

// New creates a PlayerPositionModel with the given configuration and a freshly
// initialised compute backend.
func New(cfg Config) (*PlayerPositionModel, error) {
	cfg = cfg.withDefaults()

	backend, err := compute.New()
	if err != nil {
		return nil, fmt.Errorf("create backend: %w", err)
	}

	store := model.NewStore()

	hiddenSizes := cfg.HiddenSizes // captured in closures below

	// modelFn maps [N, 9] inputs → [N, 2] predictions.
	modelFn := func(scope *model.Scope, x *Node) *Node {
		return mlpForward(scope, x, hiddenSizes)
	}

	trainer := train.NewTrainer(
		backend,
		store,
		modelFn,
		loss.MeanSquaredError,
		optimizer.Adam().LearningRate(cfg.LearningRate).Done(),
		nil, // trainMetrics — use defaults (batch loss + moving-average loss)
		nil, // evalMetrics
	).WithMaxExecutors(2000) // each unique batch size needs its own compiled graph

	// Each game produces a batch of unique size [N, 9], so each unique N
	// triggers a new JIT compilation inside the Exec cache.  Raise the limit
	// to cover all games across all training epochs.
	trainer.OnExecCreation(func(e *model.Exec, _ train.GraphType) {
		e.SetMaxCache(2000)
	})

	// Build the inference executor once; it reuses the same store (variables).
	inferExec, err := model.NewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		return mlpForward(scope, x, hiddenSizes)
	})
	if err != nil {
		return nil, fmt.Errorf("create inference exec: %w", err)
	}

	return &PlayerPositionModel{
		cfg:       cfg,
		backend:   backend,
		store:     store,
		trainer:   trainer,
		inferExec: inferExec,
	}, nil
}

// Store returns the underlying model.Store (variables and hyperparameters).
func (m *PlayerPositionModel) Store() *model.Store { return m.store }

// Trainer returns the underlying train.Trainer.
func (m *PlayerPositionModel) Trainer() *train.Trainer { return m.trainer }

// mlpForward builds the MLP forward pass graph.
//
// x must be shape [N, inputDim].  The function stacks len(hiddenSizes) ReLU
// dense layers followed by a linear output layer of width 2.
func mlpForward(scope *model.Scope, x *Node, hiddenSizes []int) *Node {
	for i, sz := range hiddenSizes {
		x = denseLayer(scope.In("hidden%d", i), x, sz, activation.TypeRelu)
	}
	return denseLayer(scope.In("output"), x, 2, activation.TypeNone)
}

// denseLayer adds one fully-connected layer: y = activation(x W + b).
func denseLayer(scope *model.Scope, x *Node, outSize int, act activation.Type) *Node {
	g := x.Graph()
	inSize := x.Shape().Dimensions[x.Shape().Rank()-1]

	w := scope.VariableWithShape("w", shapes.Make(dtypes.Float32, inSize, outSize)).NodeValue(g)
	b := scope.VariableWithShape("b", shapes.Make(dtypes.Float32, outSize)).NodeValue(g)

	y := MatMul(x, w)
	y = Add(y, ExpandLeftToRank(b, y.Rank()))

	if act != activation.TypeNone {
		y = activation.Apply(act, y)
	}
	return y
}

// Predict returns (x, y) coordinates predicted by the model for each row in
// inputs.  inputs must be a flat slice of length N × datasets.InputFeatureLen.
//
// The returned slices have length N each.
func (m *PlayerPositionModel) Predict(inputs []float32) (xs, ys []float32, err error) {
	n := len(inputs) / datasets.InputFeatureLen
	if n == 0 || len(inputs)%datasets.InputFeatureLen != 0 {
		return nil, nil, fmt.Errorf("inputs length %d is not a multiple of InputFeatureLen=%d",
			len(inputs), datasets.InputFeatureLen)
	}

	inputTensor := tensors.FromFlatDataAndDimensions(inputs, n, datasets.InputFeatureLen)
	defer inputTensor.FinalizeAll() //nolint:errcheck

	results, err := m.inferExec.Call(inputTensor)
	if err != nil {
		return nil, nil, fmt.Errorf("inference: %w", err)
	}
	if len(results) == 0 {
		return nil, nil, fmt.Errorf("inference returned no outputs")
	}
	defer results[0].FinalizeAll() //nolint:errcheck

	flat, ok := results[0].Value().([][]float32)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected output type %T", results[0].Value())
	}

	xs = make([]float32, n)
	ys = make([]float32, n)
	for i, row := range flat {
		xs[i] = row[0]
		ys[i] = row[1]
	}
	return xs, ys, nil
}
