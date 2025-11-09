package monte

import (
	"math"
	"math/rand"
	"testing"

	"github.com/Noofbiz/dataBowl/datasets"
)

// TestIntegrationWithPredictionDataset is an integration-style test that will
// attempt to load local prediction CSVs and run the Monte simulator against a
// real example. If no CSVs are available at common locations, the test is
// skipped rather than failing. This enhanced version also computes simple
// statistics on sampled landings and checks neighbor index validity and
// sample diversity.
func TestIntegrationWithPredictionDataset(t *testing.T) {
	patterns := []string{
		"../assets/kaggle/prediction/train/*.csv",
		"assets/kaggle/prediction/train/*.csv",
		"../assets/prediction/*.csv",
		"assets/prediction/*.csv",
	}

	var predDS *datasets.PredictionDataset
	var err error
	for _, p := range patterns {
		predDS, err = datasets.NewPredictionDataset(p, "")
		if err == nil {
			t.Logf("Loaded prediction dataset using pattern: %s", p)
			break
		}
	}

	if predDS == nil || err != nil {
		t.Skipf("Prediction CSV files not found in repository assets (tried common patterns); skipping integration test: last error: %v", err)
		return
	}

	// Ensure dataset has at least one example
	if predDS.Len() == 0 {
		t.Skip("Prediction dataset loaded but contains zero examples; skipping integration test")
		return
	}

	// Use the first example as the starting condition.
	inputs, _, err := predDS.Example(0)
	if err != nil {
		t.Fatalf("failed to read first example from prediction dataset: %v", err)
	}
	if len(inputs) < 2 {
		t.Fatalf("unexpected input dimensionality from dataset example: got %d", len(inputs))
	}

	// Create Monte simulator and set deterministic RNG for reproducibility in CI runs.
	K := 20
	m, err := NewMonte(predDS, K)
	if err != nil {
		t.Fatalf("failed to create Monte simulator: %v", err)
	}
	m.rng = rand.New(rand.NewSource(42))

	const numSims = 50
	const steps = 24

	// Pass the global example index (0) so Monte can fetch frame players for influence.
	results, err := m.Simulate(0, inputs, numSims, steps)
	if err != nil {
		t.Fatalf("Simulate error: %v", err)
	}
	if len(results) != numSims {
		t.Fatalf("unexpected number of simulations returned: want %d got %d", numSims, len(results))
	}

	// Collect landing statistics and verify neighbor indices are valid.
	var sumX, sumY float64
	var sumSqX, sumSqY float64
	neighborSet := make(map[int]struct{})

	for si, res := range results {
		if len(res.Trajectory) != steps {
			t.Fatalf("simulation %d: unexpected trajectory length: want %d got %d", si, steps, len(res.Trajectory))
		}
		// Basic sanity checks: z >= 0 for all points
		for pi, p := range res.Trajectory {
			if math.IsNaN(float64(p.X)) || math.IsNaN(float64(p.Y)) || math.IsNaN(float64(p.Z)) {
				t.Fatalf("simulation %d point %d: NaN in trajectory", si, pi)
			}
			if p.Z < 0 {
				t.Fatalf("simulation %d point %d: negative Z value: %v", si, pi, p.Z)
			}
		}

		// Neighbor index must be valid
		if res.NeighborIdx < 0 || res.NeighborIdx >= predDS.Len() {
			t.Fatalf("simulation %d: neighbor index out of range: %d", si, res.NeighborIdx)
		}
		neighborSet[res.NeighborIdx] = struct{}{}

		// accumulate landing stats
		sumX += float64(res.LandX)
		sumY += float64(res.LandY)
		sumSqX += float64(res.LandX) * float64(res.LandX)
		sumSqY += float64(res.LandY) * float64(res.LandY)
	}

	meanX := sumX / float64(len(results))
	meanY := sumY / float64(len(results))
	varX := sumSqX/float64(len(results)) - meanX*meanX
	varY := sumSqY/float64(len(results)) - meanY*meanY
	stdX := math.Sqrt(math.Max(0, varX))
	stdY := math.Sqrt(math.Max(0, varY))

	t.Logf("Landing mean: (%.3f, %.3f), std: (%.3f, %.3f), unique neighbors sampled: %d",
		meanX, meanY, stdX, stdY, len(neighborSet))

	// Assertions on statistics & diversity:
	// - variance should be non-zero (i.e., sampling produces diversity) unless dataset is degenerate
	if stdX <= 1e-6 && stdY <= 1e-6 {
		// If dataset has only one example, std may be zero; in that case ensure dataset size is 1.
		if predDS.Len() > 1 && len(neighborSet) <= 1 {
			t.Fatalf("expected sampling diversity but got near-zero std and only %d unique neighbor(s)", len(neighborSet))
		}
	}

	// - we sampled at least one neighbor (obvious) and not more than K unique neighbors unless dataset is small
	if len(neighborSet) == 0 {
		t.Fatalf("no neighbors were recorded in simulation results")
	}
	if len(neighborSet) > K && predDS.Len() >= K {
		// It's fine to sample more than K unique neighbors across draws if K >= dataset size,
		// but if predDS.Len() >= K we expect number unique <= predDS.Len() always, so this is just a sanity check.
		t.Logf("sampled %d unique neighbors (K=%d, dataset size=%d)", len(neighborSet), K, predDS.Len())
	}

	// Basic bounds check for landings to detect gross outliers
	for si, res := range results {
		if math.Abs(float64(res.LandX)) > 100000 || math.Abs(float64(res.LandY)) > 100000 {
			t.Fatalf("simulation %d: unrealistic landing coordinates: %v,%v", si, res.LandX, res.LandY)
		}
	}

	t.Logf("Integration simulation completed: ran %d simulations, %d steps each", len(results), steps)
}
