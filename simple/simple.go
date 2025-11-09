package simple

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

// Config holds configurable hyperparameters for the MLP model and training.
type Config struct {
	// HiddenSizes is the list of hidden layer sizes. Example: []int{64, 32}
	// If empty, a single hidden layer of size 64 will be used.
	HiddenSizes []int

	// InputDim is the dimensionality of the input feature vector. If zero,
	// a sensible default (6) will be used by NewModel.
	InputDim int

	// LearningRate used by the optimizer (SGD or Adam).
	LearningRate float64

	// Epochs to train for (default if 0 will be set by NewModel to 10).
	Epochs int

	// BatchSize for mini-batch updates (default if 0 will be set by NewModel to 8).
	BatchSize int

	// Seed controls RNG for weight init and shuffling. If zero, time-based seed is used.
	Seed int64

	// Optimizer selects the optimizer to use: "adam" or "sgd". Default: "adam".
	Optimizer string

	// Adam hyperparameters (used when Optimizer == "adam"; defaults below if zero).
	Beta1   float64
	Beta2   float64
	Epsilon float64

	// ClipNorm is the gradient clipping threshold. If zero a sensible default is used.
	ClipNorm float32
}

// Dataset is the minimal interface this package requires from a prediction dataset.
// This keeps simple decoupled from the concrete datasets package while allowing
// callers to pass the repository's PredictionDataset (it matches these methods).
type Dataset interface {
	Len() int
	// Batch returns inputs and labels for the provided global indices.
	// Inputs: [][]float32 where each input has dimension 6 (x,y,s,a,o,dir).
	// Labels: [][]float32 where each label is length 2 (ball_land_x, ball_land_y).
	Batch(indices []int) ([][]float32, [][]float32, error)
}

// Model is a small configurable MLP used for predicting landing coordinates
// from the `PredictionDataset` inputs. By default it uses a lightweight,
// self-contained trainer implemented in pure Go (no external deep-learning
// dependencies) so tests can run quickly and deterministically. A gomlx-based
// trainer can be added in a separate file and swapped in by the caller.
type Model struct {
	// Config used for training / initialization.
	Config Config

	// layerSizes includes input size, hidden sizes, then output size.
	layerSizes []int

	// weights[l] is a matrix of shape [out][in] for layer l -> l+1
	weights [][][]float32

	// biases[l] is a vector of length out for layer l -> l+1
	biases [][]float32

	// rng used for weight initialization and shuffling
	rng *rand.Rand
}

// NewModel creates a new Model instance with the provided configuration.
// It initializes weights (small random values) and is ready to train.
func NewModel(cfg Config) (*Model, error) {
	// defaults
	if len(cfg.HiddenSizes) == 0 {
		cfg.HiddenSizes = []int{64}
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.001
	}
	if cfg.Epochs == 0 {
		cfg.Epochs = 10
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 8
	}
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}

	m := &Model{
		Config: cfg,
		rng:    rand.New(rand.NewSource(cfg.Seed)),
	}

	// fixed input/output dims for the prediction dataset (input dim configurable)
	inputDim := cfg.InputDim
	if inputDim == 0 {
		inputDim = 6
	}
	const outputDim = 2

	// build layer sizes
	sizes := make([]int, 0, 2+len(cfg.HiddenSizes))
	sizes = append(sizes, inputDim)
	sizes = append(sizes, cfg.HiddenSizes...)
	sizes = append(sizes, outputDim)
	m.layerSizes = sizes

	// allocate weights and biases
	L := len(sizes) - 1
	m.weights = make([][][]float32, L)
	m.biases = make([][]float32, L)
	for l := 0; l < L; l++ {
		in := sizes[l]
		out := sizes[l+1]
		mat := make([][]float32, out)
		for j := 0; j < out; j++ {
			row := make([]float32, in)
			for i := 0; i < in; i++ {
				// Xavier/Glorot uniform initialization heuristic
				limit := float32(math.Sqrt(6.0 / float64(in+out)))
				row[i] = (m.rng.Float32()*2.0 - 1.0) * limit * 0.5
			}
			mat[j] = row
		}
		m.weights[l] = mat
		b := make([]float32, out)
		for j := 0; j < out; j++ {
			b[j] = 0.0
		}
		m.biases[l] = b
	}

	return m, nil
}

// activationReLU applies ReLU in-place over the slice.
func activationReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// activationReLUDeriv returns elementwise derivative of ReLU applied to preact.
// derivative is 1 where preact>0, else 0.
func activationReLUDeriv(preact []float32) []float32 {
	d := make([]float32, len(preact))
	for i := range preact {
		if preact[i] > 0 {
			d[i] = 1.0
		} else {
			d[i] = 0.0
		}
	}
	return d
}

// forwardSingle performs a forward pass for a single input vector, returning:
// - preActivations: list of pre-activation vectors per layer (len = L)
// - activations: list of activation vectors per layer (len = L+1, activations[0] = input)
// Note: L is number of layers (hidden+output)
func (m *Model) forwardSingle(input []float32) (preActs [][]float32, acts [][]float32, err error) {
	if len(input) != m.layerSizes[0] {
		return nil, nil, errors.New("input has incorrect dimension")
	}
	L := len(m.weights)
	acts = make([][]float32, L+1)
	acts[0] = make([]float32, len(input))
	copy(acts[0], input)

	preActs = make([][]float32, L)
	for l := 0; l < L; l++ {
		inVec := acts[l]
		outDim := len(m.biases[l])
		inDim := len(inVec)
		pre := make([]float32, outDim)
		W := m.weights[l]
		b := m.biases[l]
		for j := 0; j < outDim; j++ {
			sum := float32(0.0)
			row := W[j]
			for i := 0; i < inDim; i++ {
				sum += row[i] * inVec[i]
			}
			sum += b[j]
			pre[j] = sum
		}
		preActs[l] = pre

		// Activation: ReLU for hidden, linear for last layer
		act := make([]float32, outDim)
		copy(act, pre)
		if l < L-1 {
			activationReLU(act)
		}
		acts[l+1] = act
	}
	return preActs, acts, nil
}

// PredictBatch returns model predictions for a batch of inputs.
// It does a purely forward pass (no training). The returned [][]float32 has
// shape [batch][2] (land_x, land_y).
func (m *Model) PredictBatch(inputs [][]float32) ([][]float32, error) {
	out := make([][]float32, len(inputs))
	for i, in := range inputs {
		_, acts, err := m.forwardSingle(in)
		if err != nil {
			return nil, err
		}
		// last activation is output
		last := acts[len(acts)-1]
		pred := make([]float32, len(last))
		copy(pred, last)
		out[i] = pred
	}
	return out, nil
}

// TrainWithDataset trains the model using a small in-package SGD trainer that
// does not depend on any external deep-learning framework. This trainer is
// intentionally simple: it runs mini-batch SGD with ReLU activations and a
// mean-squared-error loss. SGD updates are applied per-example.
func (m *Model) TrainWithDataset(ds Dataset) error {
	if ds == nil {
		return errors.New("dataset is nil")
	}
	n := ds.Len()
	if n == 0 {
		return errors.New("dataset has no examples")
	}

	epochs := m.Config.Epochs
	if epochs <= 0 {
		epochs = 10
	}
	batchSize := m.Config.BatchSize
	if batchSize <= 0 {
		batchSize = 8
	}
	lr := float32(m.Config.LearningRate)
	if lr <= 0 {
		lr = 0.001
	}

	// Gradient clipping handled where appropriate (per-layer clipping in the optimizer).
	// Build initial index slice
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}

	// training loop
	for ep := 0; ep < epochs; ep++ {
		// shuffle indices
		m.rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// iterate minibatches (we will accumulate gradients over the minibatch and apply averaged SGD update)
		for bstart := 0; bstart < n; bstart += batchSize {
			bend := bstart + batchSize
			if bend > n {
				bend = n
			}
			batchIdx := indices[bstart:bend]

			// fetch the whole minibatch in one call
			inputs, labels, err := ds.Batch(batchIdx)
			if err != nil {
				return err
			}
			batchN := len(inputs)
			if batchN == 0 {
				continue
			}

			// Initialize gradient accumulators (same shape as weights / biases)
			L := len(m.weights)
			gradW := make([][][]float32, L)
			gradB := make([][]float32, L)
			for l := 0; l < L; l++ {
				outDim := len(m.biases[l])
				inDim := len(m.weights[l][0])
				gradW[l] = make([][]float32, outDim)
				for j := 0; j < outDim; j++ {
					gradW[l][j] = make([]float32, inDim)
				}
				gradB[l] = make([]float32, outDim)
			}

			// Accumulate gradients for each example in the batch
			for ex := 0; ex < batchN; ex++ {
				in := inputs[ex]
				la := labels[ex]

				preacts, acts, err := m.forwardSingle(in)
				if err != nil {
					return err
				}

				// dLoss/dOutput = 2*(pred - label)
				outAct := acts[len(acts)-1]
				delta := make([]float32, len(outAct))
				for j := 0; j < len(outAct); j++ {
					delta[j] = 2.0 * (outAct[j] - la[j])
				}

				// Backprop to compute gradients, accumulate into gradW/gradB
				for l := len(m.weights) - 1; l >= 0; l-- {
					inAct := acts[l]
					outDim := len(delta)
					inDim := len(inAct)

					// accumulate bias gradients and weight gradients
					for j := 0; j < outDim; j++ {
						gradB[l][j] += delta[j]
						for i := 0; i < inDim; i++ {
							gradW[l][j][i] += delta[j] * inAct[i]
						}
					}

					// propagate delta to previous layer if needed
					if l > 0 {
						prevLen := len(m.weights[l][0])
						newDelta := make([]float32, prevLen)
						for i := 0; i < prevLen; i++ {
							sum := float32(0.0)
							for j := 0; j < outDim; j++ {
								sum += m.weights[l][j][i] * delta[j]
							}
							newDelta[i] = sum
						}
						deriv := activationReLUDeriv(preacts[l-1])
						for i := 0; i < prevLen; i++ {
							newDelta[i] *= deriv[i]
						}
						delta = newDelta
					}
				}
			}

			// Apply averaged gradients (SGD) over the minibatch
			bInv := float32(1.0 / float64(batchN))
			for l := 0; l < L; l++ {
				outDim := len(m.biases[l])
				inDim := len(m.weights[l][0])
				for j := 0; j < outDim; j++ {
					db := gradB[l][j] * bInv
					m.biases[l][j] -= lr * db
					for i := 0; i < inDim; i++ {
						dw := gradW[l][j][i] * bInv
						m.weights[l][j][i] -= lr * dw
					}
				}
			}
		} // end batches
	} // end epochs

	return nil
}

// Helper: min of two ints
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
