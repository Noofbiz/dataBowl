package monte

import (
	"math"
	"math/rand"
	"testing"
)

// mockDS is a small in-memory dataset implementing the Dataset interface used by Monte.
type mockDS struct {
	inputs [][]float32
	labels [][]float32
}

func (m *mockDS) Len() int { return len(m.inputs) }

func (m *mockDS) Example(i int) ([]float32, []float32, error) {
	if i < 0 || i >= len(m.inputs) {
		return nil, nil, nil
	}
	return m.inputs[i], m.labels[i], nil
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestSimulateReturnsTrajectories(t *testing.T) {
	// Build mock dataset with several nearby examples.
	ds := &mockDS{
		inputs: [][]float32{
			{0, 0, 10, 0, 0, 0},
			{1, 1, 12, 0, 0, 0},
			{2, 2, 8, 0, 0, 0},
		},
		labels: [][]float32{
			{50, 50},
			{60, 40},
			{40, 60},
		},
	}

	initial, _, _ := ds.Example(0)
	m := &Monte{
		DS:  ds,
		K:   3,
		rng: rand.New(rand.NewSource(12345)),
	}

	numSims := 10
	steps := 6
	results, err := m.Simulate(initial, numSims, steps)
	if err != nil {
		t.Fatalf("Simulate returned error: %v", err)
	}
	if len(results) != numSims {
		t.Fatalf("expected %d results, got %d", numSims, len(results))
	}

	// Allowed tolerance for exact label matches (landings should come from dataset, possibly exactly).
	const tol = 1e-6
	// collect all labels for membership checks
	labelSet := make(map[[2]float32]bool)
	for _, lab := range ds.labels {
		labelSet[[2]float32{lab[0], lab[1]}] = true
	}

	for i, r := range results {
		if len(r.Trajectory) != steps {
			t.Fatalf("result %d: expected trajectory length %d, got %d", i, steps, len(r.Trajectory))
		}
		// NeighborIdx must be a valid dataset index
		if r.NeighborIdx < 0 || r.NeighborIdx >= ds.Len() {
			t.Fatalf("result %d: neighbor idx out of range: %d", i, r.NeighborIdx)
		}
		// Landings should be one of the dataset labels (sampled neighbor label, possibly with small float noise)
		found := false
		for key := range labelSet {
			if approxEqual(float64(r.LandX), float64(key[0]), tol) && approxEqual(float64(r.LandY), float64(key[1]), tol) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("result %d: landing (%v,%v) not found among dataset labels", i, r.LandX, r.LandY)
		}
	}
}

func TestSimulateK1Deterministic(t *testing.T) {
	// With K=1 the simulator should always pick the single nearest neighbor.
	ds := &mockDS{
		inputs: [][]float32{
			{0, 0, 10, 0, 0, 0},     // nearest
			{100, 100, 10, 0, 0, 0}, // far
			{-100, -50, 9, 0, 0, 0}, // far
		},
		labels: [][]float32{
			{99.5, -3.2},
			{200, 200},
			{-50, -50},
		},
	}

	initial, _, _ := ds.Example(0)
	m := &Monte{
		DS:  ds,
		K:   1,
		rng: rand.New(rand.NewSource(999)), // deterministic RNG
	}

	numSims := 5
	steps := 4
	results, err := m.Simulate(initial, numSims, steps)
	if err != nil {
		t.Fatalf("Simulate returned error: %v", err)
	}
	if len(results) != numSims {
		t.Fatalf("expected %d results, got %d", numSims, len(results))
	}

	expectedX := float64(ds.labels[0][0])
	expectedY := float64(ds.labels[0][1])
	const tol = 1e-6

	for i, r := range results {
		if len(r.Trajectory) != steps {
			t.Fatalf("result %d: expected trajectory length %d, got %d", i, steps, len(r.Trajectory))
		}
		if r.NeighborIdx != 0 {
			t.Fatalf("result %d: expected neighbor idx 0 with K=1, got %d", i, r.NeighborIdx)
		}
		if !approxEqual(float64(r.LandX), expectedX, tol) || !approxEqual(float64(r.LandY), expectedY, tol) {
			t.Errorf("result %d: landing mismatch, expected (%v,%v), got (%v,%v)", i, expectedX, expectedY, r.LandX, r.LandY)
		}
	}
}
