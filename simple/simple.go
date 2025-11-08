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

	// LearningRate used by the default local trainer (SGD).
	LearningRate float64

	// Epochs to train for (default if 0 will be set by NewModel to 10).
	Epochs int

	// BatchSize for mini-batch SGD (default if 0 will be set by NewModel to 8).
	BatchSize int

	// Seed controls RNG for weight init and shuffling. If zero, time-based seed is used.
	Seed int64
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
// self-contained trainer implemented in pure Go (no external deep learning
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

	// fixed input/output dims for the prediction dataset
	const inputDim = 6
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
	for l := range L {
		in := sizes[l]
		out := sizes[l+1]
		mat := make([][]float32, out)
		for j := range out {
			row := make([]float32, in)
			for i := range in {
				// Xavier/Glorot uniform initialization heuristic
				limit := float32(math.Sqrt(6.0 / float64(in+out)))
				row[i] = (m.rng.Float32()*2.0 - 1.0) * limit * 0.5
			}
			mat[j] = row
		}
		m.weights[l] = mat
		b := make([]float32, out)
		for j := range out {
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
	for l := range L {
		inVec := acts[l]
		outDim := len(m.biases[l])
		pre := make([]float32, outDim)
		W := m.weights[l]
		b := m.biases[l]
		for j := range outDim {
			sum := float32(0.0)
			row := W[j]
			for i := range len(inVec) {
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

// meanSquaredErrorBatch computes MSE averaged over batch (for diagnostics).
// func meanSquaredErrorBatch(preds, labels [][]float32) float64 {
// 	if len(preds) == 0 {
// 		return 0.0
// 	}
// 	sum := 0.0
// 	n := 0
// 	for i := range preds {
// 		for j := range preds[i] {
// 			d := float64(preds[i][j] - labels[i][j])
// 			sum += d * d
// 			n++
// 		}
// 	}
// 	if n == 0 {
// 		return 0.0
// 	}
// 	return sum / float64(n)
// }

// TrainWithDataset trains the model using a small in-package SGD trainer that
// does not depend on any external deep-learning framework. This trainer is
// intentionally simple: it runs mini-batch SGD with ReLU activations and a
// mean-squared-error loss. Its purpose is to provide reproducible, fast tests.
// For production and more sophisticated training you can implement a gomlx
// trainer in a separate file and call that instead.
func (m *Model) TrainWithDataset(ds Dataset) error {
	if ds == nil {
		return errors.New("dataset is nil")
	}
	n := ds.Len()
	if n == 0 {
		return errors.New("dataset has no examples")
	}

	epochs := m.Config.Epochs
	batchSize := m.Config.BatchSize
	lr := float32(m.Config.LearningRate)

	// Build initial index slice
	indices := make([]int, n)
	for i := range n {
		indices[i] = i
	}

	// training loop
	for range epochs {
		// shuffle indices
		m.rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// iterate minibatches
		for bstart := 0; bstart < n; bstart += batchSize {
			bend := min(bstart+batchSize, n)
			batchIdx := indices[bstart:bend]

			inputs, labels, err := ds.Batch(batchIdx)
			if err != nil {
				return err
			}
			batchN := len(inputs)
			if batchN == 0 {
				continue
			}

			// Forward pass for all examples in batch: store preacts and acts per example
			preactsBatch := make([][][]float32, batchN)
			actsBatch := make([][][]float32, batchN)
			for i := range batchN {
				preacts, acts, err := m.forwardSingle(inputs[i])
				if err != nil {
					return err
				}
				preactsBatch[i] = preacts
				actsBatch[i] = acts
			}

			// Initialize gradients accumulators (same shape as weights / biases)
			L := len(m.weights)
			gradW := make([][][]float32, L)
			gradB := make([][]float32, L)
			for l := range L {
				outDim := len(m.biases[l])
				inDim := len(m.weights[l][0])
				gradW[l] = make([][]float32, outDim)
				for j := range outDim {
					gradW[l][j] = make([]float32, inDim)
				}
				gradB[l] = make([]float32, outDim)
			}

			// Backprop per example, accumulate gradients
			for ex := range batchN {
				acts := actsBatch[ex]
				preacts := preactsBatch[ex]
				// compute dLoss/dOutput = 2*(pred - label)
				outAct := acts[len(acts)-1]
				delta := make([]float32, len(outAct))
				for j := range outAct {
					delta[j] = 2 * (outAct[j] - labels[ex][j])
				}
				// backprop through layers from L-1 to 0
				for l := len(m.weights) - 1; l >= 0; l-- {
					inAct := acts[l]     // activation of previous layer (input to layer l)
					outDim := len(delta) // number of units in this layer
					// compute gradients w.r.t weights and biases for this layer
					for j := range outDim {
						// bias gradient
						gradB[l][j] += delta[j]
						// weight gradients
						for i := range len(inAct) {
							gradW[l][j][i] += delta[j] * inAct[i]
						}
					}
					// compute delta for previous layer if not input layer
					if l > 0 {
						prevLen := len(m.weights[l][0]) // inDim
						newDelta := make([]float32, prevLen)
						// propagate: newDelta[i] = sum_j W[j][i] * delta[j]
						for i := range prevLen {
							sum := float32(0.0)
							for j := range outDim {
								sum += m.weights[l][j][i] * delta[j]
							}
							newDelta[i] = sum
						}
						// multiply by activation derivative (ReLU) of preAct of previous layer
						deriv := activationReLUDeriv(preacts[l-1])
						for i := range prevLen {
							newDelta[i] *= deriv[i]
						}
						delta = newDelta
					}
				}
			}

			// Average gradients over batch and apply SGD update
			bInv := float32(1.0 / float64(batchN))
			for l := range len(m.weights) {
				outDim := len(m.biases[l])
				inDim := len(m.weights[l][0])
				for j := range outDim {
					// update bias
					db := gradB[l][j] * bInv
					m.biases[l][j] -= lr * db
					for i := range inDim {
						dw := gradW[l][j][i] * bInv
						m.weights[l][j][i] -= lr * dw
					}
				}
			}
		} // end batches
	} // end epochs

	return nil
}
