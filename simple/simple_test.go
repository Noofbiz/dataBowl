package simple

import (
	"math"
	"testing"

	"github.com/Noofbiz/dataBowl/datasets"
)

// mockDataset implements the minimal Dataset interface required by the trainer.
type mockDataset struct {
	inputs [][]float32
	labels [][]float32
}

func (m *mockDataset) Len() int { return len(m.inputs) }

func (m *mockDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	in := make([][]float32, len(indices))
	la := make([][]float32, len(indices))
	for i, idx := range indices {
		in[i] = m.inputs[idx]
		la[i] = m.labels[idx]
	}
	return in, la, nil
}

func mse(preds, labels [][]float32) float64 {
	if len(preds) == 0 {
		return 0.0
	}
	var sum float64
	var n int
	for i := range preds {
		for j := range preds[i] {
			d := float64(preds[i][j] - labels[i][j])
			sum += d * d
			n++
		}
	}
	if n == 0 {
		return 0.0
	}
	return sum / float64(n)
}

// TestModelTrainWithMockDataset verifies the pure-Go trainer reduces MSE on a
// simple synthetic regression dataset.
func TestModelTrainWithMockDataset(t *testing.T) {
	// synthesize a small dataset where label is a linear function of first two inputs.
	const N = 120
	inputs := make([][]float32, N)
	labels := make([][]float32, N)
	for i := 0; i < N; i++ {
		x := float32(i % 10)        // 0..9
		y := float32((i / 10) % 10) // 0..9 repeated
		in := make([]float32, 6)    // [x,y,s,a,o,dir]
		in[0] = x
		in[1] = y
		// remaining features left zero
		inputs[i] = in
		// label = [2*x + 0.5*y, x - y]
		labels[i] = []float32{2*x + 0.5*y, x - y}
	}

	ds := &mockDataset{inputs: inputs, labels: labels}

	cfg := Config{
		HiddenSizes:  []int{32, 16},
		LearningRate: 0.01,
		Epochs:       30,
		BatchSize:    16,
		Seed:         42,
	}

	model, err := NewModel(cfg)
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}

	// Evaluate baseline MSE on a holdout subset (first 20 examples)
	holdN := 20
	holdInputs := inputs[:holdN]
	holdLabels := labels[:holdN]

	predBefore, err := model.PredictBatch(holdInputs)
	if err != nil {
		t.Fatalf("PredictBatch(before) error: %v", err)
	}
	mseBefore := mse(predBefore, holdLabels)

	// Train
	if err := model.TrainWithDataset(ds); err != nil {
		t.Fatalf("TrainWithDataset error: %v", err)
	}

	predAfter, err := model.PredictBatch(holdInputs)
	if err != nil {
		t.Fatalf("PredictBatch(after) error: %v", err)
	}
	mseAfter := mse(predAfter, holdLabels)

	t.Logf("mse before=%.6f after=%.6f", mseBefore, mseAfter)

	// Expect MSE to have decreased (allow tiny tolerance)
	if !(mseAfter+1e-9 < mseBefore) {
		t.Fatalf("expected mse to decrease after training: before=%.6f after=%.6f", mseBefore, mseAfter)
	}

	// Ensure predictions are finite
	for i := range predAfter {
		for j := range predAfter[i] {
			if math.IsNaN(float64(predAfter[i][j])) || math.IsInf(float64(predAfter[i][j]), 0) {
				t.Fatalf("non-finite prediction at %d,%d: %v", i, j, predAfter[i][j])
			}
		}
	}
}

// subsetDataset wraps a PredictionDataset and exposes only a subset of examples
// via simple.Dataset interface so integration training runs quickly.
type subsetDataset struct {
	base    *datasets.PredictionDataset
	indices []int // global indices into base
}

func (s *subsetDataset) Len() int { return len(s.indices) }

func (s *subsetDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	if len(indices) == 0 {
		return [][]float32{}, [][]float32{}, nil
	}
	globals := make([]int, len(indices))
	for i, idx := range indices {
		// idx is relative to subset; map to global index
		if idx < 0 || idx >= len(s.indices) {
			return nil, nil, nil
		}
		globals[i] = s.indices[idx]
	}
	return s.base.Batch(globals)
}

// TestModelTrainWithRealPredictionDataset tries to load real prediction CSVs
// (if available) and runs a short training loop on a small subset to exercise
// integration with the PredictionDataset.
func TestModelTrainWithRealPredictionDataset(t *testing.T) {
	// common patterns to try
	patterns := []string{
		"assets/kaggle/prediction/train/*.csv",
		"../assets/kaggle/prediction/train/*.csv",
		"assets/prediction/*.csv",
		"../assets/prediction/*.csv",
	}

	var predDS *datasets.PredictionDataset
	var err error
	for _, p := range patterns {
		predDS, err = datasets.NewPredictionDataset(p, "")
		if err == nil {
			t.Logf("loaded prediction dataset with pattern: %s", p)
			break
		}
	}

	if err != nil || predDS == nil {
		t.Skipf("prediction CSVs not found; skipping integration test: last error: %v", err)
		return
	}

	// take a small subset of global examples to keep training fast
	total := predDS.Len()
	if total == 0 {
		t.Skip("prediction dataset empty; skipping integration test")
		return
	}
	limit := min(128, total)

	indices := make([]int, limit)
	for i := 0; i < limit; i++ {
		indices[i] = i
	}

	sub := &subsetDataset{base: predDS, indices: indices}

	cfg := Config{
		HiddenSizes:  []int{32},
		LearningRate: 0.005,
		Epochs:       2, // small epoch count for CI
		BatchSize:    16,
		Seed:         123,
	}

	model, err := NewModel(cfg)
	if err != nil {
		t.Fatalf("NewModel error: %v", err)
	}

	// Train on subset; ensure this completes without error.
	if err := model.TrainWithDataset(sub); err != nil {
		t.Fatalf("TrainWithDataset (real) error: %v", err)
	}

	// Run a small prediction batch using first few subset examples
	predInputs, predLabels, err := sub.Batch([]int{0, 1, 2, 3})
	if err != nil {
		t.Fatalf("subset Batch error: %v", err)
	}
	preds, err := model.PredictBatch(predInputs)
	if err != nil {
		t.Fatalf("PredictBatch error: %v", err)
	}
	if len(preds) != len(predInputs) {
		t.Fatalf("PredictBatch returned unexpected number of predictions: want %d got %d", len(predInputs), len(preds))
	}
	// basic sanity checks on returned values
	for i := range preds {
		if len(preds[i]) != 2 {
			t.Fatalf("unexpected prediction dimension: got %d want 2", len(preds[i]))
		}
		for j := 0; j < 2; j++ {
			if math.IsNaN(float64(preds[i][j])) || math.IsInf(float64(preds[i][j]), 0) {
				t.Fatalf("non-finite prediction at %d,%d: %v", i, j, preds[i][j])
			}
			// also verify labels are reasonable floats
			if math.IsNaN(float64(predLabels[i][j])) || math.IsInf(float64(predLabels[i][j]), 0) {
				t.Fatalf("non-finite label at %d,%d: %v", i, j, predLabels[i][j])
			}
		}
	}
}
