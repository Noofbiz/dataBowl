package datasets

import "github.com/gomlx/gomlx/pkg/core/tensors"

// This file provides two dataset implementations that load CSV data from the
// kaggle assets and present them as examples suitable for model training.
//
// Both datasets use lazy loading - they store file paths and only read the
// actual data when needed (during batch creation), minimizing memory usage.
//
// Notes on gomlx tensors:
//   - Converting batches into gomlx tensors is left as a small, well-defined
//     step. Different gomlx versions expose different helper constructors for
//     tensors; to avoid hard dependency on a particular gomlx API here we return
//     contiguous float32 buffers along with shape metadata. These are trivial to
//     convert into gomlx tensors (or any other tensor type) in your training
//     code. See the ToGomlx* conversion examples in comments below.
//
// Layout and intended usage:
//
// PredictionDataset
//   - Stores paths to CSV files matching a pattern
//   - Loads CSV rows on-demand that contain player state at the moment of the throw plus
//     the landing position of the ball.
//   - Inputs per example: x, y, s, a, o, dir  (in that order, float32)
//   - Labels per example: ball_land_x, ball_land_y (float32 vector length 2)
//
// The datasets implement this interface in order to interact with GoMLX
// training loops and batching utilities.
// The data sets should use lazy loading to save memory, as the CSV files can
// be large.
type Dataset interface {
	Len() int
	Example(i int) (inputs []float32, labels []float32, err error)
	Batch(indices []int) (inputs [][]float32, labels [][]float32, err error)
	Shuffle(seed int64)

	// To implement gomlx's train.Dataset interface
	Yield() (any, []*tensors.Tensor, []*tensors.Tensor, error)
}
