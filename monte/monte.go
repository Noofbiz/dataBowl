package monte

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// Point3 is a 3D point (x,y,z).
type Point3 struct {
	X float32
	Y float32
	Z float32
}

// SimulationResult holds the simulated trajectory for a single Monte Carlo draw.
type SimulationResult struct {
	// Trajectory is a sequence of 3D points describing the ball while in the air.
	// Length is equal to Steps used for simulation.
	Trajectory []Point3

	// LandX, LandY is the sampled landing point used for this simulation.
	LandX float32
	LandY float32

	// NeighborIdx is the index of the dataset example that was sampled.
	NeighborIdx int
}

// Dataset is a minimal interface the Monte package needs from the prediction
// dataset. Using an interface here avoids importing the concrete package type
// and makes the monte package easier to compile/use in different module setups.
type Dataset interface {
	// Len returns the number of examples in the dataset.
	Len() int

	// Example returns the inputs and labels for the example at global index.
	// It matches the signature used by the datasets.PredictionDataset Example method.
	Example(idx int) (inputs []float32, labels []float32, err error)
}

// Monte runs Monte Carlo simulations using a Dataset as the source of empirical
// landing outcomes. It finds K nearest neighbors in input-feature space and
// samples their landing labels to produce trajectories.
type Monte struct {
	DS Dataset
	K  int

	// rng is used for sampling neighbors and adding stochasticity to trajectories.
	rng *rand.Rand
}

// NewMonte creates a new Monte object.
// ds must be a non-nil PredictionDataset. k must be >= 1.
func NewMonte(ds Dataset, k int) (*Monte, error) {
	if ds == nil {
		return nil, errors.New("dataset cannot be nil")
	}
	if k < 1 {
		return nil, fmt.Errorf("k must be >= 1, got %d", k)
	}
	return &Monte{
		DS:  ds,
		K:   k,
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// Simulate runs `numSims` Monte Carlo simulations starting from `initial`
// and returns the resulting trajectories. `initial` must be a float32 slice
// containing the 6 features in the same order the PredictionDataset expects:
// [x, y, s, a, o, dir]. `steps` controls how many points per trajectory are
// returned (must be >= 2).
//
// The simulation algorithm:
//  1. Find the K nearest dataset examples to `initial` in Euclidean distance
//     over the 6-dimensional input features.
//  2. Randomly sample one of the K neighbors with probability proportional to
//     the inverse distance (closer neighbors are more likely).
//  3. Use the neighbor's landing label (ball_land_x, ball_land_y) as the
//     landing point for the simulation.
//  4. Generate a smooth arc (quadratic Bézier) from initial (x,y) to landing
//     (x,y) with a control point elevated in z (height) to produce a realistic
//     in-air trajectory. Add small stochastic perturbations to peak height and
//     lateral offset to reflect variability.
//
// Returns a slice of SimulationResult of length `numSims`.
func (m *Monte) Simulate(initial []float32, numSims, steps int) ([]SimulationResult, error) {
	if m == nil {
		return nil, errors.New("Monte object is nil")
	}
	if initial == nil || len(initial) != 6 {
		return nil, fmt.Errorf("initial must be length-6 slice: [x,y,s,a,o,dir]")
	}
	if numSims <= 0 {
		return nil, fmt.Errorf("numSims must be > 0")
	}
	if steps < 2 {
		return nil, fmt.Errorf("steps must be >= 2")
	}

	// 1) KNN search
	neighbors, err := m.knnNeighbors(initial, m.K)
	if err != nil {
		return nil, err
	}
	if len(neighbors) == 0 {
		return nil, fmt.Errorf("no neighbors found in dataset")
	}

	// Prepare weights inverse to distance (with epsilon)
	eps := float64(1e-6)
	weights := make([]float64, len(neighbors))
	var totalWeight float64
	for i, nb := range neighbors {
		w := 1.0 / (float64(nb.distance) + eps)
		weights[i] = w
		totalWeight += w
	}

	results := make([]SimulationResult, 0, numSims)
	for sim := 0; sim < numSims; sim++ {
		// 2) Sample neighbor index according to weights
		target := m.rng.Float64() * totalWeight
		acc := 0.0
		choice := 0
		for i, w := range weights {
			acc += w
			if target <= acc {
				choice = i
				break
			}
		}
		chosen := neighbors[choice]

		// 3) create trajectory
		traj := make([]Point3, steps)

		x0 := float64(initial[0])
		y0 := float64(initial[1])
		x2 := float64(chosen.labels[0])
		y2 := float64(chosen.labels[1])

		// Estimate peak height using speed feature (initial[2]) and some randomness.
		speed := float64(initial[2])
		basePeak := clampFloat64(speed*0.5+6.0, 1.5, 40.0) // empirical heuristic
		peakNoise := (m.rng.Float64()*2.0 - 1.0) * 1.5     // +/-1.5 units
		peakHeight := basePeak + peakNoise

		// Lateral offset: perpendicular to the straight line between P0 and P2.
		dx := x2 - x0
		dy := y2 - y0
		perpX := -dy
		perpY := dx
		perpLen := math.Hypot(perpX, perpY)
		var ux, uy float64
		if perpLen > 1e-9 {
			ux = perpX / perpLen
			uy = perpY / perpLen
		} else {
			ux = 0
			uy = 0
		}
		// lateral magnitude influenced by orientation feature (initial[4]) and some randomness
		orient := float64(initial[4])
		lateralBase := clampFloat64(orient*0.12, -6.0, 6.0)
		lateralNoise := (m.rng.Float64()*2.0 - 1.0) * 1.0
		lateralOffset := lateralBase + lateralNoise

		// Control point (P1) is midpoint offset by perpendicular vector and elevated in Z
		midX := (x0 + x2) / 2.0
		midY := (y0 + y2) / 2.0
		p1x := midX + ux*lateralOffset
		p1y := midY + uy*lateralOffset
		p1z := peakHeight

		// Build quadratic Bézier for x,y,z: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
		for i := 0; i < steps; i++ {
			var t float64
			if steps == 1 {
				t = 0
			} else {
				t = float64(i) / float64(steps-1)
			}
			it := 1.0 - t
			b0 := it * it
			b1 := 2.0 * it * t
			b2 := t * t

			x := b0*x0 + b1*p1x + b2*x2
			y := b0*y0 + b1*p1y + b2*y2
			z := b0*0.0 + b1*p1z + b2*0.0

			// small jitter to avoid overly deterministic paths
			jitterScale := 0.02 * math.Hypot(x2-x0, y2-y0)
			x += (m.rng.Float64()*2.0 - 1.0) * jitterScale * 0.5
			y += (m.rng.Float64()*2.0 - 1.0) * jitterScale * 0.5
			z += (m.rng.Float64()*2.0 - 1.0) * 0.2

			traj[i] = Point3{X: float32(x), Y: float32(y), Z: float32(math.Max(0.0, z))}
		}

		results = append(results, SimulationResult{
			Trajectory:  traj,
			LandX:       float32(x2),
			LandY:       float32(y2),
			NeighborIdx: chosen.idx,
		})
	}

	return results, nil
}

// neighbor holds a dataset neighbor candidate.
type neighbor struct {
	idx      int
	distance float32
	labels   []float32 // [land_x, land_y]
}

// knnNeighbors performs a simple linear scan KNN search over the dataset.
// It returns up to k neighbors sorted by increasing distance.
func (m *Monte) knnNeighbors(initial []float32, k int) ([]neighbor, error) {
	n := m.DS.Len()
	if n == 0 {
		return nil, fmt.Errorf("prediction dataset is empty")
	}
	// collect all distances
	candidates := make([]neighbor, 0, n)
	for i := 0; i < n; i++ {
		inp, labels, err := m.DS.Example(i)
		if err != nil {
			// skip entries we can't read
			continue
		}
		if len(inp) != 6 {
			continue
		}
		dist := euclideanDistanceSquared(initial, inp)
		candidates = append(candidates, neighbor{
			idx:      i,
			distance: float32(math.Sqrt(float64(dist))),
			labels:   labels,
		})
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no readable examples in dataset")
	}

	// sort by increasing distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	if k > len(candidates) {
		k = len(candidates)
	}
	return candidates[:k], nil
}

// euclideanDistanceSquared computes squared Euclidean distance between two equal-length float32 slices.
func euclideanDistanceSquared(a, b []float32) float64 {
	sum := 0.0
	for i := 0; i < len(a) && i < len(b); i++ {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return sum
}

// clampFloat64 clamps v to [minVal, maxVal].
func clampFloat64(v, minVal, maxVal float64) float64 {
	if v < minVal {
		return minVal
	}
	if v > maxVal {
		return maxVal
	}
	return v
}
