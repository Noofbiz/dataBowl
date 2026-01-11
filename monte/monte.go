package monte

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Noofbiz/dataBowl/datasets"
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

	// Simulation tunables (exported so callers can set them)
	ForceEps           float64 // small epsilon to avoid divide-by-zero in force calcs
	ForceScale         float64 // how strongly player forces move control point
	DefenseWeight      float64 // multiplier for defensive repulsion
	MaxPerPlayer       float64 // clamp per-player influence
	LandingNudgeFactor float64 // how much landing point is nudged by net force

	// Legacy per-role heuristics (kept for backward compatibility)
	RoleTargetedWeight float64 // targeted receivers have larger influence (legacy)
	RolePasserWeight   float64 // passer influence multiplier (legacy)

	// Per-category tunables introduced for finer control:
	// RoleCoverageWeight: magnitude applied for defensive coverage players (repulsive)
	// RoleRouteRunnerWeight: magnitude applied for other route-running players (offense slight attractive)
	RoleCoverageWeight    float64
	RoleRouteRunnerWeight float64

	// RoleWeights is an optional map from role keyword -> multiplier used to
	// compute per-player role influence. If populated, it takes precedence over
	// the legacy heuristics and per-category fields.
	RoleWeights map[string]float64

	// rng is used for sampling neighbors and adding stochasticity to trajectories.
	rng *rand.Rand
}

// NewMonte creates a new Monte object.
// ds must be a non-nil PredictionDataset. k must be >= 1.
//
// The returned Monte includes sensible defaults for the simulation tunables.
// Callers can override these via the setter methods provided on Monte, or by
// using ConfigureDatasetFrameIndex to try to configure the dataset frame index
// cache when the underlying dataset implements the FrameIndexConfigurer interface.
func NewMonte(ds Dataset, k int) (*Monte, error) {
	if ds == nil {
		return nil, errors.New("dataset cannot be nil")
	}
	if k < 1 {
		return nil, fmt.Errorf("k must be >= 1, got %d", k)
	}
	return &Monte{
		DS:                 ds,
		K:                  k,
		ForceEps:           1e-3,
		ForceScale:         1.2,
		DefenseWeight:      1.0,
		MaxPerPlayer:       10.0,
		LandingNudgeFactor: 0.02,
		RoleTargetedWeight: 2.0,
		RolePasserWeight:   1.1,
		// Per-category defaults
		RoleCoverageWeight:    1.3,
		RoleRouteRunnerWeight: 1.15,
		// Default RoleWeights provides an explicit map for common role keywords.
		RoleWeights: map[string]float64{
			"target":   2.0,
			"receiver": 2.0,
			"pass":     1.1,
		},
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// FrameIndexConfigurer can be implemented by datasets that want to expose
// runtime configuration of their frame index cache (TTL and max entries).
// Monte provides a convenience helper to call these methods on the underlying
// dataset when available.
type FrameIndexConfigurer interface {
	SetFrameIndexTTL(d time.Duration)
	SetFrameIndexMaxEntries(n int)
}

// ConfigureDatasetFrameIndex attempts to configure the underlying dataset's
// frame-index cache (TTL and capacity) if it implements FrameIndexConfigurer.
func (m *Monte) ConfigureDatasetFrameIndex(ttl time.Duration, maxEntries int) {
	if m == nil {
		return
	}
	if cfg, ok := m.DS.(FrameIndexConfigurer); ok {
		cfg.SetFrameIndexTTL(ttl)
		cfg.SetFrameIndexMaxEntries(maxEntries)
	}
}

// Setter helper methods for tunables. These let callers (or CLI wiring code)
// adjust Monte behavior after creation.
func (m *Monte) SetForceEps(v float64) {
	if m == nil {
		return
	}
	m.ForceEps = v
}

func (m *Monte) SetForceScale(v float64) {
	if m == nil {
		return
	}
	m.ForceScale = v
}

func (m *Monte) SetDefenseWeight(v float64) {
	if m == nil {
		return
	}
	m.DefenseWeight = v
}

func (m *Monte) SetMaxPerPlayer(v float64) {
	if m == nil {
		return
	}
	m.MaxPerPlayer = v
}

func (m *Monte) SetLandingNudgeFactor(v float64) {
	if m == nil {
		return
	}
	m.LandingNudgeFactor = v
}

func (m *Monte) SetRoleTargetedWeight(v float64) {
	if m == nil {
		return
	}
	m.RoleTargetedWeight = v
}

func (m *Monte) SetRolePasserWeight(v float64) {
	if m == nil {
		return
	}
	m.RolePasserWeight = v
}

// SetRoleCoverageWeight sets the multiplier used for players classified as
// defensive coverage (e.g., CB, safety, zone, man). This controls the magnitude
// of their per-player contribution (repulsion when on defense).
func (m *Monte) SetRoleCoverageWeight(v float64) {
	if m == nil {
		return
	}
	m.RoleCoverageWeight = v
}

// SetRoleRouteRunnerWeight sets the multiplier used for other route-running
// players (offense slightly attractive, defense slightly repulsive when
// combined with the team sign).
func (m *Monte) SetRoleRouteRunnerWeight(v float64) {
	if m == nil {
		return
	}
	m.RoleRouteRunnerWeight = v
}

// SetRoleWeights replaces the role keyword -> multiplier mapping used by Monte
// when computing role-based multipliers. If set, this map will be consulted
// before falling back to the legacy heuristics.
func (m *Monte) SetRoleWeights(v map[string]float64) {
	if m == nil {
		return
	}
	m.RoleWeights = v
}

// LoadRoleConfig reads a JSON configuration file that describes role categories,
// their associated tokens and per-category weights. The JSON format supports an
// array of category objects of the form:
//
//	{
//	  "category": "coverage",
//	  "tokens": ["cb","safety"],
//	  "weight": 1.3
//	}
//
// Recognized category names (case-insensitive): "coverage", "target", "pass",
// "route". For each token in a category, the token is added to m.RoleWeights
// (token -> weight). Additionally, known category names will set the per-
// category fields (RoleCoverageWeight, RoleRouteRunnerWeight, RoleTargetedWeight,
// RolePasserWeight) to allow CLI-style tuning or heuristics to reflect config.
func (m *Monte) LoadRoleConfig(path string) error {
	if m == nil {
		return fmt.Errorf("Monte is nil")
	}
	if path == "" {
		return fmt.Errorf("empty path")
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read role config: %w", err)
	}

	// Support either an object with optional "categories" and "tunables" fields
	// or a top-level array of category objects (backwards compatibility).
	var raw struct {
		Categories []struct {
			Category string   `json:"category"`
			Tokens   []string `json:"tokens"`
			Weight   float64  `json:"weight"`
		} `json:"categories"`
		Tunables *struct {
			ForceEps              *float64 `json:"force_eps"`
			ForceScale            *float64 `json:"force_scale"`
			DefenseWeight         *float64 `json:"defense_weight"`
			MaxPerPlayer          *float64 `json:"max_per_player"`
			LandingNudgeFactor    *float64 `json:"landing_nudge_factor"`
			RoleTargetedWeight    *float64 `json:"role_targeted_weight"`
			RolePasserWeight      *float64 `json:"role_passer_weight"`
			RoleCoverageWeight    *float64 `json:"role_coverage_weight"`
			RoleRouteRunnerWeight *float64 `json:"role_route_runner_weight"`
		} `json:"tunables"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		// Also accept top-level array form for backward compatibility
		var arr []struct {
			Category string   `json:"category"`
			Tokens   []string `json:"tokens"`
			Weight   float64  `json:"weight"`
		}
		if jerr := json.Unmarshal(data, &arr); jerr != nil {
			return fmt.Errorf("unmarshal role config: %w (json error: %v)", err, jerr)
		}
		raw.Categories = arr
		raw.Tunables = nil
	}

	// Ensure RoleWeights map exists
	if m.RoleWeights == nil {
		m.RoleWeights = make(map[string]float64)
	}
	// Apply category token weights and set per-category fields when provided.
	for _, c := range raw.Categories {
		cat := strings.ToLower(strings.TrimSpace(c.Category))
		weight := c.Weight
		if weight == 0 {
			// skip zero-weight entries (no effect)
			continue
		}
		// assign category-level fields for known categories
		switch cat {
		case "coverage", "defense", "defensive":
			m.RoleCoverageWeight = weight
		case "route", "route_runner", "runners":
			m.RoleRouteRunnerWeight = weight
		case "target", "targeted", "receiver":
			m.RoleTargetedWeight = weight
		case "pass", "passer", "quarterback", "qb":
			m.RolePasserWeight = weight
		}
		// register each token into RoleWeights
		for _, tok := range c.Tokens {
			tk := strings.ToLower(strings.TrimSpace(tok))
			if tk == "" {
				continue
			}
			m.RoleWeights[tk] = weight
		}
	}

	// If tunables block present, apply provided numeric tunables.
	// Use the setter methods so behavior is consistent and any validation is centralized.
	if raw.Tunables != nil {
		t := raw.Tunables
		if t.ForceEps != nil {
			m.SetForceEps(*t.ForceEps)
		}
		if t.ForceScale != nil {
			m.SetForceScale(*t.ForceScale)
		}
		if t.DefenseWeight != nil {
			m.SetDefenseWeight(*t.DefenseWeight)
		}
		if t.MaxPerPlayer != nil {
			m.SetMaxPerPlayer(*t.MaxPerPlayer)
		}
		if t.LandingNudgeFactor != nil {
			m.SetLandingNudgeFactor(*t.LandingNudgeFactor)
		}
		if t.RoleTargetedWeight != nil {
			m.SetRoleTargetedWeight(*t.RoleTargetedWeight)
		}
		if t.RolePasserWeight != nil {
			m.SetRolePasserWeight(*t.RolePasserWeight)
		}
		if t.RoleCoverageWeight != nil {
			m.SetRoleCoverageWeight(*t.RoleCoverageWeight)
		}
		if t.RoleRouteRunnerWeight != nil {
			m.SetRoleRouteRunnerWeight(*t.RoleRouteRunnerWeight)
		}
	}

	return nil
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
func (m *Monte) Simulate(globalIdx int, initial []float32, numSims, steps int) ([]SimulationResult, error) {
	if m == nil {
		return nil, errors.New("Monte object is nil")
	}
	// Allow richer initial payloads: require at least the 6 core features.
	if initial == nil || len(initial) < 6 {
		return nil, fmt.Errorf("initial must be length >= 6 slice: [x, y, s, a, o, dir, ...optional per-frame players]")
	}
	if numSims <= 0 {
		return nil, fmt.Errorf("numSims must be > 0")
	}
	if steps < 2 {
		return nil, fmt.Errorf("steps must be >= 2")
	}

	// Fetch frame players once for this global example index (if supported by the dataset).
	var players []datasets.FramePlayer
	if provider, ok := m.DS.(interface {
		FramePlayersForExample(int) ([]datasets.FramePlayer, error)
	}); ok {
		if fps, err := provider.FramePlayersForExample(globalIdx); err == nil {
			players = fps
		} else {
			players = nil
		}
	} else {
		players = nil
	}

	// 1) KNN search
	println("Finding KNN neighbors...")
	neighbors, err := m.knnNeighbors(initial, m.K)
	if err != nil {
		return nil, err
	}
	if len(neighbors) == 0 {
		return nil, fmt.Errorf("no neighbors found in dataset")
	}

	// Prepare weights inverse to distance (with epsilon)
	println("Preparing neighbor weights...")
	eps := float64(1e-6)
	weights := make([]float64, len(neighbors))
	var totalWeight float64
	for i, nb := range neighbors {
		w := 1.0 / (float64(nb.distance) + eps)
		weights[i] = w
		totalWeight += w
	}

	// Blend landing points of neighbors as the target
	var blendedLandX, blendedLandY float64
	for i, nb := range neighbors {
		blendedLandX += float64(nb.labels[0]) * weights[i]
		blendedLandY += float64(nb.labels[1]) * weights[i]
	}
	if totalWeight > 0 {
		blendedLandX /= totalWeight
		blendedLandY /= totalWeight
	} else {
		blendedLandX = float64(initial[0])
		blendedLandY = float64(initial[1])
	}

	// Simulation-level tuning constants
	forceEps := m.ForceEps
	if forceEps == 0 {
		forceEps = 1e-3
	}
	forceScale := m.ForceScale
	if forceScale == 0 {
		forceScale = 1.2
	}
	defenseWeight := m.DefenseWeight
	if defenseWeight == 0 {
		defenseWeight = 1.0
	}
	maxPerPlayer := m.MaxPerPlayer
	if maxPerPlayer == 0 {
		maxPerPlayer = 10.0
	}
	roleTargetedWeight := m.RoleTargetedWeight
	if roleTargetedWeight == 0 {
		roleTargetedWeight = 2.0
	}
	rolePasserWeight := m.RolePasserWeight
	if rolePasserWeight == 0 {
		rolePasserWeight = 1.1
	}

	gravity := -9.8 // m/s^2, downward acceleration

	results := make([]SimulationResult, numSims)
	seeds := make([]int64, numSims)
	for i := 0; i < numSims; i++ {
		seeds[i] = m.rng.Int63()
	}

	workerCount := runtime.NumCPU()
	if workerCount > numSims {
		workerCount = numSims
	}
	jobs := make(chan int, numSims)
	var wg sync.WaitGroup
	wg.Add(workerCount)

	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			for sim := range jobs {
				rng := rand.New(rand.NewSource(seeds[sim]))

				// Use the players fetched once earlier for this global example (if any).
				framePlayers := players

				// Initial position
				x0 := float64(initial[0])
				y0 := float64(initial[1])
				z0 := 0.0

				// Target (blended landing)
				xT := blendedLandX
				yT := blendedLandY

				// Estimate time of flight based on distance and initial speed
				speed := float64(initial[2])
				distXY := math.Hypot(xT-x0, yT-y0)
				// Add some noise to speed for realism
				speedNoise := (rng.Float64()*2.0 - 1.0) * 1.5
				launchSpeed := clampFloat64(speed+speedNoise, 1.0, 60.0)

				// Choose a reasonable launch angle (e.g., 35-55 degrees), add noise
				angleDeg := 45.0 + (rng.Float64()*2.0-1.0)*8.0
				angleRad := angleDeg * math.Pi / 180.0

				// Compute initial velocity components
				v0 := launchSpeed
				v0z := v0 * math.Sin(angleRad)
				v0xy := v0 * math.Cos(angleRad)
				dir := math.Atan2(yT-y0, xT-x0)
				v0x := v0xy * math.Cos(dir)
				v0y := v0xy * math.Sin(dir)

				// Time step (dt) and total time
				totalTime := (2 * v0z) / -gravity
				if totalTime <= 0 || math.IsNaN(totalTime) || math.IsInf(totalTime, 0) {
					totalTime = distXY / (v0 + 1e-6)
				}
				if totalTime < 0.5 {
					totalTime = 0.5
				}
				dt := totalTime / float64(steps-1)

				// State variables
				x, y, z := x0, y0, z0
				vx, vy, vz := v0x, v0y, v0z

				traj := make([]Point3, steps)
				landed := false

				for i := 0; i < steps; i++ {
					// Compute net player force at this step
					forceX := 0.0
					forceY := 0.0
					if len(framePlayers) > 0 {
						for _, fp := range framePlayers {
							px := float64(fp.X)
							py := float64(fp.Y)
							dpx := px - x
							dpy := py - y
							dist := math.Hypot(dpx, dpy)
							if dist < forceEps {
								continue
							}
							uxp := dpx / dist
							uyp := dpy / dist

							sign := 1.0
							if strings.EqualFold(fp.Side, "Defense") || strings.EqualFold(fp.Side, "D") {
								sign = -defenseWeight
							}

							roleMul := 1.0
							lrole := strings.ToLower(strings.TrimSpace(fp.Role))
							if m.RoleWeights != nil && len(m.RoleWeights) > 0 {
								found := false
								for k, v := range m.RoleWeights {
									kl := strings.ToLower(strings.TrimSpace(k))
									if kl == "" {
										continue
									}
									if strings.Contains(lrole, kl) {
										roleMul = v
										found = true
										break
									}
								}
								if !found {
								}
							} else {
								if strings.Contains(lrole, "coverage") || strings.Contains(lrole, "defend") || strings.Contains(lrole, "defense") {
									roleMul = 1.3
								} else if strings.Contains(lrole, "target") || strings.Contains(lrole, "receiver") || strings.Contains(lrole, "targeted") {
									roleMul = roleTargetedWeight
								} else if strings.Contains(lrole, "pass") || strings.Contains(lrole, "quarterback") || strings.Contains(lrole, "qb") {
									roleMul = rolePasserWeight
								} else {
									if strings.EqualFold(fp.Side, "Defense") || strings.EqualFold(fp.Side, "D") {
										roleMul = 1.05
									} else {
										roleMul = 1.15
									}
								}
							}

							w := roleMul * sign * (1.0 / (dist*dist + forceEps))
							if w > maxPerPlayer {
								w = maxPerPlayer
							}
							if w < -maxPerPlayer {
								w = -maxPerPlayer
							}
							forceX += w * uxp
							forceY += w * uyp
						}
					}

					// Apply player force as acceleration (scaled)
					ax := forceScale * forceX
					ay := forceScale * forceY
					az := gravity

					// Euler integration for velocity and position
					vx += ax * dt
					vy += ay * dt
					vz += az * dt

					x += vx * dt
					y += vy * dt
					z += vz * dt

					// Add small random jitter for realism
					jitterScale := 0.02 * distXY
					x += (rng.Float64()*2.0 - 1.0) * jitterScale * 0.5
					y += (rng.Float64()*2.0 - 1.0) * jitterScale * 0.5
					z += (rng.Float64()*2.0 - 1.0) * 0.2

					if z < 0 {
						z = 0
						landed = true
					}

					traj[i] = Point3{X: float32(x), Y: float32(y), Z: float32(z)}

					if landed {
						// Fill remaining steps with landing point
						for j := i + 1; j < steps; j++ {
							traj[j] = Point3{X: float32(x), Y: float32(y), Z: 0}
						}
						break
					}
				}

				results[sim] = SimulationResult{
					Trajectory:  traj,
					LandX:       float32(x),
					LandY:       float32(y),
					NeighborIdx: -1, // No single neighbor, since we blended
				}
			}
		}()
	}

	for i := 0; i < numSims; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	return results, nil
}

// PredictNextFrame computes a Monte-style estimate for the next-frame position
// of the object described by `initial`. It requires that the underlying dataset
// implement a method:
//
//	NextFramePositionForExample(int) (float32, float32, bool, error)
//
// The method samples `numSims` neighbors (using the same weighted sampling as
// Simulate) and queries each chosen neighbor's next-frame position. It returns
// the mean next-frame coordinates across successful samples.
func (m *Monte) PredictNextFrame(globalIdx int, initial []float32, numSims int) (float64, float64, error) {
	if m == nil {
		return 0, 0, fmt.Errorf("Monte object is nil")
	}
	if initial == nil || len(initial) < 6 {
		return 0, 0, fmt.Errorf("initial must be length >= 6 slice: [x, y, s, a, o, dir]")
	}
	if numSims <= 0 {
		return 0, 0, fmt.Errorf("numSims must be > 0")
	}

	// Dataset provider must expose NextFramePositionForExample.
	provider, ok := m.DS.(interface {
		NextFramePositionForExample(int) (float32, float32, bool, error)
	})
	if !ok {
		return 0, 0, fmt.Errorf("underlying dataset does not implement NextFramePositionForExample")
	}

	// Find neighbors using the existing KNN routine.
	neighbors, err := m.knnNeighbors(initial, m.K)
	if err != nil {
		return 0, 0, err
	}
	if len(neighbors) == 0 {
		return 0, 0, fmt.Errorf("no neighbors found in dataset")
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

	// Use Monte RNG to sample neighbors repeatedly.
	seeds := make([]int64, numSims)
	for i := 0; i < numSims; i++ {
		seeds[i] = m.rng.Int63()
	}

	sumX := 0.0
	sumY := 0.0
	success := 0

	for s := 0; s < numSims; s++ {
		rng := rand.New(rand.NewSource(seeds[s]))
		target := rng.Float64() * totalWeight
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

		// Query neighbor's next-frame position via provider.
		nx, ny, found, nerr := provider.NextFramePositionForExample(chosen.idx)
		if nerr != nil {
			// skip problematic neighbors
			continue
		}
		if !found {
			// neighbor has no next-frame label; skip
			continue
		}
		sumX += float64(nx)
		sumY += float64(ny)
		success++
	}

	if success == 0 {
		return 0, 0, fmt.Errorf("no next-frame positions available from sampled neighbors")
	}
	return sumX / float64(success), sumY / float64(success), nil
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

	// Use a worker pool to compute distances concurrently.
	jobs := make(chan int, n)
	resultsCh := make(chan neighbor, n)

	workerCount := runtime.NumCPU()
	if workerCount > n {
		workerCount = n
	}

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			for i := range jobs {
				inp, labels, err := m.DS.Example(i)
				if err != nil {
					// skip entries we can't read
					continue
				}
				if len(inp) != 6 {
					continue
				}
				dist := euclideanDistanceSquared(initial, inp)
				resultsCh <- neighbor{
					idx:      i,
					distance: float32(math.Sqrt(dist)),
					labels:   labels,
				}
			}
		}()
	}

	// enqueue jobs
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)

	// wait for workers to finish and then close results
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// collect candidates from channel
	candidates := make([]neighbor, 0, n)
	for nb := range resultsCh {
		candidates = append(candidates, nb)
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

// --- Automated Parameter Tuning ---

// TuneParameters performs random search to optimize simulation parameters for lowest RMSE on landing points.
// It prints the best parameter set found and its RMSE, and returns a slice of all tried parameter sets and their RMSE for visualization.
func (m *Monte) TuneParameters(
	exampleIndices []int, // indices of examples to use for tuning
	numSims int,          // number of simulations per example
	steps int,            // number of steps per simulation
	numTrials int,        // number of random parameter sets to try
) []struct {
	ForceScale    float64
	DefenseWeight float64
	MaxPerPlayer  float64
	RoleTargeted  float64
	RolePasser    float64
	RMSE          float64
} {
	type Params struct {
		ForceScale    float64
		DefenseWeight float64
		MaxPerPlayer  float64
		RoleTargeted  float64
		RolePasser    float64
	}

	bestRMSE := 1e12
	var bestParams Params
	var results []struct {
		ForceScale    float64
		DefenseWeight float64
		MaxPerPlayer  float64
		RoleTargeted  float64
		RolePasser    float64
		RMSE          float64
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	for trial := 0; trial < numTrials; trial++ {
		// Sample random parameters within reasonable ranges
		params := Params{
			ForceScale:    0.5 + rng.Float64()*2.0,    // [0.5, 2.5]
			DefenseWeight: 0.5 + rng.Float64()*2.0,    // [0.5, 2.5]
			MaxPerPlayer:  5.0 + rng.Float64()*15.0,   // [5, 20]
			RoleTargeted:  1.0 + rng.Float64()*3.0,    // [1, 4]
			RolePasser:    0.5 + rng.Float64()*2.0,    // [0.5, 2.5]
		}

		// Set parameters
		m.SetForceScale(params.ForceScale)
		m.SetDefenseWeight(params.DefenseWeight)
		m.SetMaxPerPlayer(params.MaxPerPlayer)
		m.SetRoleTargetedWeight(params.RoleTargeted)
		m.SetRolePasserWeight(params.RolePasser)

		// Evaluate RMSE over the examples
		var sumSq float64
		var count int

		for _, idx := range exampleIndices {
			inputs, labels, err := m.DS.Example(idx)
			if err != nil || len(inputs) < 6 || len(labels) < 2 {
				continue
			}
			resultsSim, err := m.Simulate(idx, inputs, numSims, steps)
			if err != nil || len(resultsSim) == 0 {
				continue
			}
			// Use mean landing point over all sims
			var meanX, meanY float64
			for _, r := range resultsSim {
				meanX += float64(r.LandX)
				meanY += float64(r.LandY)
			}
			meanX /= float64(len(resultsSim))
			meanY /= float64(len(resultsSim))

			dx := meanX - float64(labels[0])
			dy := meanY - float64(labels[1])
			sumSq += dx*dx + dy*dy
			count++
		}

		if count == 0 {
			continue
		}
		rmse := math.Sqrt(sumSq / float64(count))
		results = append(results, struct{
			ForceScale   float64
			DefenseWeight float64
			MaxPerPlayer  float64
			RoleTargeted  float64
			RolePasser    float64
			RMSE          float64
		}{
			ForceScale:    params.ForceScale,
			DefenseWeight: params.DefenseWeight,
			MaxPerPlayer:  params.MaxPerPlayer,
			RoleTargeted:  params.RoleTargeted,
			RolePasser:    params.RolePasser,
			RMSE:          rmse,
		})
		if rmse < bestRMSE {
			bestRMSE = rmse
			bestParams = params
		}
	}

	fmt.Printf("Best RMSE: %.4f\n", bestRMSE)
	fmt.Printf("Best Params: ForceScale=%.3f, DefenseWeight=%.3f, MaxPerPlayer=%.3f, RoleTargeted=%.3f, RolePasser=%.3f\n",
		bestParams.ForceScale, bestParams.DefenseWeight, bestParams.MaxPerPlayer, bestParams.RoleTargeted, bestParams.RolePasser)

	return results
}
