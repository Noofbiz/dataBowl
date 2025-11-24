package main

import (
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/Noofbiz/dataBowl/monte"
	"github.com/Noofbiz/dataBowl/simple"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var (
	// cacheVersion is incremented when the on-disk precompute format changes.
	cacheVersion = 1
	// precomputeWorkers is populated from CLI flag (if >0, overrides NumCPU).
	precomputeWorkers = 0
	// precomputeForce indicates whether to force recompute even if cache exists.
	precomputeForce = false

	// precomputeProgressIntervalSeconds controls how often Precompute logs progress.
	// Default is 3 seconds and can be overridden via JSON or CLI flag.
	precomputeProgressIntervalSeconds = 3
)

// defaultRoleConfigJSON is the embedded JSON used to create cmd/compare/data.json
// when the user did not provide a --monte-role-config path. We write this file as
// a convenience so the default configuration is available on disk; however when
// no JSON option is set the CLI flags are preferred (we do not load this file
// automatically into the Monte instance).
const defaultRoleConfigJSON = `{
  "monte": {
    "k": 8,
    "sims": 60,
    "steps": 24,
    "force_eps": 0.001,
    "force_scale": 1.2,
    "defense_weight": 1.0,
    "role_targeted_weight": 2.0,
    "role_passer_weight": 1.1,
    "role_coverage_weight": 1.3,
    "role_route_runner_weight": 1.15,
    "max_per_player": 10.0,
    "landing_nudge_factor": 0.02
  },
  "tunables": {
    "precompute": {
      "cache_path": "output/cache.gob",
      "workers": 0,
      "force": false,
      "progress_interval_seconds": 3
    },
    "training": {
      "optimizer": "adam",
      "learning_rate": 0.005,
      "epochs": 8,
      "batch_size": 32,
      "adam_beta1": 0.9,
      "adam_beta2": 0.999,
      "adam_eps": 1e-8,
      "clip_norm": 5.0
    },
    "dataset_frame_index": {
      "cache_ttl_seconds": 300,
      "cache_max_entries": 2000
    }
  },
   "categories": [
     {
       "category": "coverage",
       "tokens": ["cb","corner","safety","db","coverage","zone","man","defense"],
       "weight": 1.3
     },
     {
       "category": "target",
       "tokens": ["target","targeted","receiver","wr","te","slot"],
       "weight": 2.0
     },
     {
       "category": "pass",
       "tokens": ["pass","passer","qb","quarterback"],
       "weight": 1.1
     },
     {
       "category": "route",
       "tokens": ["route","route_runner","runner","slot","wr","te"],
       "weight": 1.15
     }
   ]
 }
`

// subsetDataset adapts datasets.PredictionDataset to the simple.Dataset
// interface but exposing only a subset of global indices for faster training.
type subsetDataset struct {
	base    *datasets.PredictionDataset
	indices []int // global indices into base
}

// trajLine is a small helper to store a trajectory (used by the plotting code).
type trajLine struct {
	xys plotter.XYs
}

// forecastDataset wraps PredictionDataset and emits inputs that concatenate:
// [current_features(6), neighbors_features(K*5)] and labels as next-frame x,y.
// It can optionally precompute all inputs/labels into memory for faster training.
type forecastDataset struct {
	base    *datasets.PredictionDataset
	indices []int
	K       int
	Workers int

	// precomputation fields
	precomputed bool
	inputs      [][]float32
	labels      [][]float32
}

// Len implements Dataset
func (f *forecastDataset) Len() int { return len(f.indices) }

// Precompute builds all inputs/labels in memory for the current index set.
// This avoids repeated CSV reads and neighbor computation during training.
//
// The implementation runs a small worker pool to parallelize per-example
// processing (reading Example, neighbor features, next-frame lookup). It
// writes results directly into the pre-allocated f.inputs/f.labels slices at
// the index corresponding to the training position.
func (f *forecastDataset) Precompute() error {
	if f.precomputed {
		return nil
	}

	n := len(f.indices)
	// allocate containers up front so workers can safely write distinct indices.
	f.inputs = make([][]float32, n)
	f.labels = make([][]float32, n)

	if n == 0 {
		f.precomputed = true
		return nil
	}

	// Determine worker pool size using precedence:
	// 1) CLI flag value (package-level precomputeWorkers) if > 0
	// 2) dataset instance field f.Workers if > 0
	// 3) runtime.NumCPU() (fallback)
	workers := precomputeWorkers
	if workers <= 0 && f.Workers > 0 {
		workers = f.Workers
	}
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers <= 0 {
		workers = 1
	}
	if workers > n {
		workers = n
	}

	jobs := make(chan int, n)
	errCh := make(chan error, workers)

	var wg sync.WaitGroup
	wg.Add(workers)

	// atomic counter for progress
	var done int64

	// progress ticker: logs periodically until all examples processed or an error occurs
	ticker := time.NewTicker(time.Duration(precomputeProgressIntervalSeconds) * time.Second)
	stopProgress := make(chan struct{})
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				d := atomic.LoadInt64(&done)
				percent := (float64(d) / float64(n)) * 100.0
				log.Printf("[Precompute] progress: %d/%d (%.1f%%)", d, n, percent)
			case <-stopProgress:
				// final log
				d := atomic.LoadInt64(&done)
				log.Printf("[Precompute] completed: %d/%d (100%%)", d, n)
				return
			}
		}
	}()

	// worker goroutines
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for pos := range jobs {
				globalIdx := f.indices[pos]

				// base features (6)
				baseInp, _, err := f.base.Example(globalIdx)
				if err != nil {
					// report error and exit worker
					errCh <- fmt.Errorf("read example %d: %w", globalIdx, err)
					return
				}

				// neighbor features flattened (K*5)
				neigh, err := f.base.FrameKNearestPlayersFeatures(globalIdx, f.K)
				if err != nil {
					errCh <- fmt.Errorf("neighbors for %d: %w", globalIdx, err)
					return
				}

				// concat inputs
				inp := make([]float32, 0, len(baseInp)+len(neigh))
				inp = append(inp, baseInp...)
				inp = append(inp, neigh...)

				// next-frame label
				nx, ny, found, err := f.base.NextFramePositionForExample(globalIdx)
				if err != nil {
					errCh <- fmt.Errorf("next-frame for %d: %w", globalIdx, err)
					return
				}
				if !found {
					// fallback: use current position as pseudo-label
					if len(baseInp) >= 2 {
						nx = baseInp[0]
						ny = baseInp[1]
					} else {
						nx = 0
						ny = 0
					}
				}

				// store results at the position corresponding to this job
				f.inputs[pos] = inp
				f.labels[pos] = []float32{nx, ny}

				// increment progress counter
				atomic.AddInt64(&done, 1)
			}
		}()
	}

	// enqueue jobs (positions into f.indices)
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)

	// wait for workers to finish
	wg.Wait()

	// stop progress logger and emit final status
	close(stopProgress)

	close(errCh)

	// If any worker reported an error, return the first one.
	select {
	case e := <-errCh:
		return e
	default:
	}

	f.precomputed = true
	return nil
}

// cacheFormat is the on-disk representation of precomputed inputs/labels.
// It includes metadata to validate cache integrity and support upgrades.
type cacheFormat struct {
	Version   int   // format version
	K         int   // K used for neighbor features when cache was built
	Indices   []int // training global indices covered by the cache
	CreatedAt int64 // unix timestamp when cache was created
	Inputs    [][]float32
	Labels    [][]float32
}

// SaveCache writes the precomputed inputs/labels into the provided file path
// using encoding/gob. It performs an atomic write (create temp file then rename).
// If the dataset has not been precomputed, SaveCache will call Precompute first.
func (f *forecastDataset) SaveCache(path string) error {
	if path == "" {
		return fmt.Errorf("empty cache path")
	}
	if !f.precomputed {
		if err := f.Precompute(); err != nil {
			return fmt.Errorf("precompute before save: %w", err)
		}
	}

	// Ensure parent directory exists
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("mkdir %s: %w", dir, err)
		}
	}

	// create a temp file in the same directory for atomicity
	tmpFile, err := os.CreateTemp(dir, filepath.Base(path)+".tmp.*")
	if err != nil {
		return fmt.Errorf("create temp cache file: %w", err)
	}
	tmpName := tmpFile.Name()
	// ensure cleanup on error
	defer func() {
		tmpFile.Close()
		// if temp file still exists and something failed, attempt to remove it.
		_ = os.Remove(tmpName)
	}()

	enc := gob.NewEncoder(tmpFile)
	now := time.Now().Unix()
	pc := cacheFormat{
		Version:   cacheVersion,
		K:         f.K,
		Indices:   f.indices,
		CreatedAt: now,
		Inputs:    f.inputs,
		Labels:    f.labels,
	}
	if err := enc.Encode(&pc); err != nil {
		return fmt.Errorf("encode cache to temp file: %w", err)
	}
	// ensure data is flushed to disk
	if err := tmpFile.Sync(); err != nil {
		// non-fatal but warn
		log.Printf("warning: sync temp cache file: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("close temp cache file: %w", err)
	}
	// atomic rename to target path
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename temp cache to target: %w", err)
	}
	return nil
}

// LoadCache attempts to read a precomputed cache from disk and populate the
// in-memory buffers. On success it sets f.precomputed = true. The function
// validates metadata (version, K, indices) to avoid accidental reuse.
func (f *forecastDataset) LoadCache(path string) error {
	if path == "" {
		return fmt.Errorf("empty cache path")
	}
	fh, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open cache file %s: %w", path, err)
	}
	defer fh.Close()
	dec := gob.NewDecoder(fh)
	var pc cacheFormat
	if err := dec.Decode(&pc); err != nil {
		return fmt.Errorf("decode cache %s: %w", path, err)
	}
	// validate format version
	if pc.Version != cacheVersion {
		return fmt.Errorf("cache version mismatch: cache=%d expected=%d", pc.Version, cacheVersion)
	}
	// validate K
	if pc.K != f.K {
		return fmt.Errorf("cache K mismatch: cache=%d expected=%d", pc.K, f.K)
	}
	// validate indices length and exact match
	if len(pc.Indices) != len(f.indices) {
		return fmt.Errorf("cache indices length mismatch: cache=%d expected=%d", len(pc.Indices), len(f.indices))
	}
	for i := range pc.Indices {
		if pc.Indices[i] != f.indices[i] {
			return fmt.Errorf("cache index mismatch at pos %d: cache=%d expected=%d", i, pc.Indices[i], f.indices[i])
		}
	}
	// validate sizes of inputs/labels
	if len(pc.Inputs) != len(f.indices) || len(pc.Labels) != len(f.indices) {
		return fmt.Errorf("cache size mismatch: inputs=%d labels=%d expected=%d", len(pc.Inputs), len(pc.Labels), len(f.indices))
	}
	// adopt cached buffers
	f.inputs = pc.Inputs
	f.labels = pc.Labels
	f.precomputed = true
	return nil
}

// Batch implements Dataset.Batch where the provided indices are positions inside f.indices.
func (f *forecastDataset) Batch(idxs []int) ([][]float32, [][]float32, error) {
	// If precomputed, serve directly from memory (fast path)
	if f.precomputed {
		inputs := make([][]float32, len(idxs))
		labels := make([][]float32, len(idxs))
		for bi, pos := range idxs {
			if pos < 0 || pos >= len(f.indices) {
				return nil, nil, fmt.Errorf("batch index %d out of range", pos)
			}
			// pos indexes into the precomputed arrays (aligned with f.indices)
			inputs[bi] = f.inputs[pos]
			labels[bi] = f.labels[pos]
		}
		return inputs, labels, nil
	}

	// Fallback: compute on-the-fly (existing behavior)
	inputs := make([][]float32, len(idxs))
	labels := make([][]float32, len(idxs))
	for bi, pos := range idxs {
		if pos < 0 || pos >= len(f.indices) {
			return nil, nil, fmt.Errorf("batch index %d out of range", pos)
		}
		globalIdx := f.indices[pos]
		// base features (6)
		baseInp, _, err := f.base.Example(globalIdx)
		if err != nil {
			return nil, nil, fmt.Errorf("read example %d: %w", globalIdx, err)
		}
		// neighbor features flattened (K*5)
		neigh, err := f.base.FrameKNearestPlayersFeatures(globalIdx, f.K)
		if err != nil {
			return nil, nil, fmt.Errorf("neighbors for %d: %w", globalIdx, err)
		}
		// concat
		inp := make([]float32, 0, len(baseInp)+len(neigh))
		inp = append(inp, baseInp...)
		inp = append(inp, neigh...)
		// next-frame label
		nx, ny, found, err := f.base.NextFramePositionForExample(globalIdx)
		if err != nil {
			return nil, nil, fmt.Errorf("next-frame for %d: %w", globalIdx, err)
		}
		if !found {
			// fallback: use current position as pseudo-label
			if len(baseInp) >= 2 {
				nx = baseInp[0]
				ny = baseInp[1]
			} else {
				nx = 0
				ny = 0
			}
		}
		inputs[bi] = inp
		labels[bi] = []float32{nx, ny}
	}
	return inputs, labels, nil
}

func (s *subsetDataset) Len() int { return len(s.indices) }

func (s *subsetDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	if len(indices) == 0 {
		return [][]float32{}, [][]float32{}, nil
	}
	globals := make([]int, len(indices))
	for i, idx := range indices {
		if idx < 0 || idx >= len(s.indices) {
			return nil, nil, fmt.Errorf("index %d out of range for subset length %d", idx, len(s.indices))
		}
		globals[i] = s.indices[idx]
	}
	return s.base.Batch(globals)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	// CLI flags
	patternFlag := flag.String("pattern", "../../datasets/assets/kaggle/prediction/train/input*.csv", "glob pattern for prediction CSV files (inputs)")
	outputPatternFlag := flag.String("output-pattern", "../../datasets/assets/kaggle/prediction/train/output*.csv", "glob pattern for output CSV files (labels). If empty, input pattern will be used")
	outDir := flag.String("out", "plots", "output directory for generated plots")
	outCSV := flag.String("out-csv", "output/out.csv", "if set, write evaluation CSV to this path and skip plotting")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	useCache := flag.Bool("cache", true, "enable in-memory CSV cache (faster random reads)")
	// Optional path for precomputed forecast dataset cache (gob). If provided,
	// compare will attempt to load this cache and fall back to computing it and
	// saving it back to this path on success.
	precomputeCache := flag.String("precompute-cache", "output/cache.gob", "path to gob file for saving/loading precomputed forecast inputs/labels")
	precomputeForceFlag := flag.Bool("precompute-force", false, "if true, force recompute and overwrite existing cache")
	precomputeWorkersFlag := flag.Int("precompute-workers", 0, "number of workers to use for precompute (0 = NumCPU)")
	precomputeProgressFlag := flag.Int("precompute-progress-interval", precomputeProgressIntervalSeconds, "progress logging interval for precompute in seconds (overrides JSON if provided)")

	// Dataset frame-index cache tunables
	cacheTTL := flag.Duration("cache-ttl", 5*time.Minute, "TTL for frame-index entries (e.g., 5m)")
	cacheMaxEntries := flag.Int("cache-max", 2000, "maximum number of entries in frame-index LRU cache")

	// Monte parameters and tunables
	monteK := flag.Int("monte-k", 8, "number of nearest neighbors (K) used by Monte")
	monteSims := flag.Int("monte-sims", 60, "number of Monte Carlo draws per initial")
	monteSteps := flag.Int("monte-steps", 24, "number of trajectory steps to generate per simulation")
	monteForceEps := flag.Float64("monte-force-eps", 1e-3, "small epsilon used in force calculations to avoid divide-by-zero")
	monteForceScale := flag.Float64("monte-force-scale", 1.2, "Monte Computer player force scale (control point influence)")
	monteDefenseWeight := flag.Float64("monte-defense-weight", 1.0, "Monte defensive repulsion weight")
	monteRoleTargetedWeight := flag.Float64("monte-role-targeted-weight", 2.0, "Monte weight for targeted receiver roles (legacy)")
	monteRolePasserWeight := flag.Float64("monte-role-passer-weight", 1.1, "Monte weight for passer roles (legacy)")
	monteRoleWeights := flag.String("monte-role-weights", "", "comma-separated role:weight pairs, e.g. 'target:2,receiver:2,pass:1.1' (takes precedence over legacy role weights when provided)")
	monteMaxPerPlayer := flag.Float64("monte-max-per-player", 10.0, "Monte clamp for per-player influence magnitude")
	monteLandingNudge := flag.Float64("monte-landing-nudge", 0.02, "Monte landing nudge factor (how much landing point is moved by net force)")
	// Per-category role tuning flags (new): coverage players (defensive coverage) and route runners (other offensive route runners)
	monteRoleCoverageWeight := flag.Float64("monte-role-coverage-weight", 1.3, "Monte weight magnitude for defensive coverage players (repulsive magnitude when on defense)")
	monteRoleRouteWeight := flag.Float64("monte-role-route-weight", 1.15, "Monte weight magnitude for other route-running players (offense slight attractive / defense slight repulsive)")
	// Optional JSON role configuration file to define categories/tokens/weights
	monteRoleConfig := flag.String("monte-role-config", "", "path to JSON role configuration file (optional). If provided, will be loaded and applied to Monte.")

	// Optimizer and training tunables for the simple model
	optimizer := flag.String("optimizer", "adam", "optimizer to use for training: 'adam' or 'sgd'")
	trainLearningRate := flag.Float64("learning-rate", 0.005, "learning rate for training (overrides JSON if provided)")
	trainEpochs := flag.Int("epochs", 8, "number of training epochs (overrides JSON if provided)")
	trainBatchSize := flag.Int("batch-size", 32, "training batch size (overrides JSON if provided)")
	adamBeta1 := flag.Float64("adam-beta1", 0.9, "Adam beta1 hyperparameter")
	adamBeta2 := flag.Float64("adam-beta2", 0.999, "Adam beta2 hyperparameter")
	adamEps := flag.Float64("adam-eps", 1e-8, "Adam epsilon hyperparameter")
	clipNorm := flag.Float64("clip-norm", 5.0, "gradient clipping norm (float32)")

	// Print merged effective configuration and exit (dry-run)
	printEffectiveConfig := flag.Bool("print-effective-config", false, "print the effective (JSON+CLI merged) configuration and exit")

	// Evaluation flag
	evalN := flag.Int("eval-n", 300, "number of examples to evaluate for RMSE when writing CSV")

	flag.Parse()

	// apply CLI flags to package-level precompute controls
	precomputeWorkers = *precomputeWorkersFlag
	precomputeForce = *precomputeForceFlag
	// track explicit CLI progress flag (override JSON if user provided)
	precomputeProgressIntervalSeconds = *precomputeProgressFlag

	// apply CLI flags to package-level precompute controls
	precomputeWorkers = *precomputeWorkersFlag
	precomputeForce = *precomputeForceFlag

	_ = rand.New(rand.NewSource(*seed))

	// Resolve pattern (for nicer log message)
	globPaths, _ := filepath.Glob(*patternFlag)
	log.Printf("Using CSV pattern: %s (found %d files)", *patternFlag, len(globPaths))

	// Load prediction dataset
	predDS, err := datasets.NewPredictionDataset(*patternFlag, *outputPatternFlag)
	if err != nil {
		log.Fatalf("failed to open prediction dataset: %v", err)
	}
	log.Printf("Prediction dataset loaded: total examples=%d", predDS.Len())

	// Configure dataset frame-index cache parameters from CLI flags.
	// Even if the caller doesn't plan to enable the CSV in-memory cache, the
	// frame-index is useful for Monte and can be tuned.
	predDS.SetFrameIndexTTL(*cacheTTL)
	predDS.SetFrameIndexMaxEntries(*cacheMaxEntries)

	if *useCache {
		log.Printf("Enabling in-memory cache for prediction dataset...")
		// Call EnableCache with no arguments for now. If EnableCache gains an
		// optional parallelism parameter later, it can be added here.
		if err := predDS.EnableCache(); err != nil {
			log.Fatalf("failed to enable dataset cache: %v", err)
		}
		log.Printf("Dataset cache enabled")
	}

	// Placeholder for training dataset. We'll construct a forecastDataset (with K neighbors)
	// and assign it to `sub` below before training.
	var sub simple.Dataset

	// Prepare Monte simulator early so role JSON tunables can influence model config.
	// We create a Monte instance now (using the CLI -monte-k as an initial value).
	monteSim, err := monte.NewMonte(predDS, *monteK)
	if err != nil {
		log.Fatalf("failed to create monte simulator: %v", err)
	}

	// Apply a default dataset frame-index configuration (keep existing CLI values).
	monteSim.ConfigureDatasetFrameIndex(*cacheTTL, *cacheMaxEntries)

	// Role configuration behavior:
	// - If the user supplied -monte-role-config, load that JSON file and apply it.
	// - If no JSON option was provided, ensure a default cmd/compare/data.json exists
	//   on disk (created from embedded defaults) AND attempt to load it. After
	//   loading JSON, we'll merge tunables from the JSON into runtime defaults,
	//   but explicit CLI flags always override JSON values.
	var effectiveRoleConfigPath string
	if strings.TrimSpace(*monteRoleConfig) != "" {
		effectiveRoleConfigPath = *monteRoleConfig
		if err := monteSim.LoadRoleConfig(*monteRoleConfig); err != nil {
			log.Fatalf("failed to load monte role config %s: %v", *monteRoleConfig, err)
		}
		log.Printf("Loaded monte role config from %s", *monteRoleConfig)
	} else {
		// Ensure default data.json exists (create from embedded defaults).
		defaultPath := filepath.Join("data.json")
		needWrite := false
		if _, err := os.Stat(defaultPath); os.IsNotExist(err) {
			needWrite = true
		} else if err != nil {
			log.Printf("warning: could not stat default role config %s: %v", defaultPath, err)
			needWrite = true
		}
		if needWrite {
			if err := os.MkdirAll(filepath.Dir(defaultPath), 0755); err != nil {
				log.Printf("warning: failed to create dir for default role config: %v", err)
			} else {
				if werr := os.WriteFile(defaultPath, []byte(defaultRoleConfigJSON), 0644); werr != nil {
					log.Printf("warning: failed to write default role config to %s: %v", defaultPath, werr)
				} else {
					log.Printf("Wrote default role config to %s", defaultPath)
				}
			}
		} else {
			log.Printf("Default role config present at %s", defaultPath)
		}
		// Attempt to load the default JSON; if loading fails, log a warning and continue.
		if lerr := monteSim.LoadRoleConfig(defaultPath); lerr != nil {
			log.Printf("warning: failed to load default monte role config %s: %v", defaultPath, lerr)
		} else {
			log.Printf("Loaded monte role config from %s", defaultPath)
		}
		effectiveRoleConfigPath = defaultPath
	}

	// Try to read additional tunables from the same JSON file (if it exists).
	// The JSON structure may include a top-level "tunables" object with subblocks
	// for precompute, training, and dataset_frame_index. We apply JSON values
	// only when the corresponding CLI flag was left at its default.
	if effectiveRoleConfigPath != "" {
		if data, err := os.ReadFile(effectiveRoleConfigPath); err == nil {
			var raw struct {
				Monte *struct {
					K                     *int     `json:"k"`
					Sims                  *int     `json:"sims"`
					Steps                 *int     `json:"steps"`
					ForceEps              *float64 `json:"force_eps"`
					ForceScale            *float64 `json:"force_scale"`
					DefenseWeight         *float64 `json:"defense_weight"`
					RoleTargetedWeight    *float64 `json:"role_targeted_weight"`
					RolePasserWeight      *float64 `json:"role_passer_weight"`
					RoleCoverageWeight    *float64 `json:"role_coverage_weight"`
					RoleRouteRunnerWeight *float64 `json:"role_route_runner_weight"`
					MaxPerPlayer          *float64 `json:"max_per_player"`
					LandingNudgeFactor    *float64 `json:"landing_nudge_factor"`
				} `json:"monte"`
				Tunables *struct {
					Precompute *struct {
						CachePath               string `json:"cache_path"`
						Workers                 *int   `json:"workers"`
						Force                   *bool  `json:"force"`
						ProgressIntervalSeconds *int   `json:"progress_interval_seconds"`
					} `json:"precompute"`
					Training *struct {
						Optimizer    *string  `json:"optimizer"`
						LearningRate *float64 `json:"learning_rate"`
						Epochs       *int     `json:"epochs"`
						BatchSize    *int     `json:"batch_size"`
						AdamBeta1    *float64 `json:"adam_beta1"`
						AdamBeta2    *float64 `json:"adam_beta2"`
						AdamEps      *float64 `json:"adam_eps"`
						ClipNorm     *float64 `json:"clip_norm"`
					} `json:"training"`
					DatasetFrameIndex *struct {
						CacheTTLSeconds *int `json:"cache_ttl_seconds"`
						CacheMaxEntries *int `json:"cache_max_entries"`
					} `json:"dataset_frame_index"`
				} `json:"tunables"`
			}
			if jerr := json.Unmarshal(data, &raw); jerr == nil {
				// Apply Monte-level numeric tunables from top-level "monte" block if present
				if raw.Monte != nil {
					if raw.Monte.K != nil && *raw.Monte.K > 0 {
						// only set monteSim.K if CLI monte-k equals default (8) OR user did not override CLI
						if *monteK == 8 {
							monteSim.K = *raw.Monte.K
						}
					}
					if raw.Monte.ForceEps != nil && *monteForceEps == 1e-3 {
						monteSim.SetForceEps(*raw.Monte.ForceEps)
					}
					if raw.Monte.ForceScale != nil && *monteForceScale == 1.2 {
						monteSim.SetForceScale(*raw.Monte.ForceScale)
					}
					if raw.Monte.DefenseWeight != nil && *monteDefenseWeight == 1.0 {
						monteSim.SetDefenseWeight(*raw.Monte.DefenseWeight)
					}
					if raw.Monte.MaxPerPlayer != nil && *monteMaxPerPlayer == 10.0 {
						monteSim.SetMaxPerPlayer(*raw.Monte.MaxPerPlayer)
					}
					if raw.Monte.LandingNudgeFactor != nil && *monteLandingNudge == 0.02 {
						monteSim.SetLandingNudgeFactor(*raw.Monte.LandingNudgeFactor)
					}
					// role-specific tunables
					if raw.Monte.RoleTargetedWeight != nil && *monteRoleTargetedWeight == 2.0 {
						monteSim.SetRoleTargetedWeight(*raw.Monte.RoleTargetedWeight)
					}
					if raw.Monte.RolePasserWeight != nil && *monteRolePasserWeight == 1.1 {
						monteSim.SetRolePasserWeight(*raw.Monte.RolePasserWeight)
					}
					if raw.Monte.RoleCoverageWeight != nil && *monteRoleCoverageWeight == 1.3 {
						monteSim.SetRoleCoverageWeight(*raw.Monte.RoleCoverageWeight)
					}
					if raw.Monte.RoleRouteRunnerWeight != nil && *monteRoleRouteWeight == 1.15 {
						monteSim.SetRoleRouteRunnerWeight(*raw.Monte.RoleRouteRunnerWeight)
					}
				}
				// Precompute tunables
				if raw.Tunables != nil && raw.Tunables.Precompute != nil {
					p := raw.Tunables.Precompute
					if p.CachePath != "" && *precomputeCache == "output/cache.gob" {
						// only override CLI default if user did not pass a custom value
						*precomputeCache = p.CachePath
					}
					if p.Workers != nil && *precomputeWorkersFlag == 0 {
						*precomputeWorkersFlag = *p.Workers
						precomputeWorkers = *p.Workers
					}
					if p.Force != nil && !*precomputeForceFlag {
						precomputeForce = *p.Force
					}
					if p.ProgressIntervalSeconds != nil && *precomputeProgressFlag == precomputeProgressIntervalSeconds {
						precomputeProgressIntervalSeconds = *p.ProgressIntervalSeconds
					}
				}
				// Training tunables
				if raw.Tunables != nil && raw.Tunables.Training != nil {
					t := raw.Tunables.Training
					if t.Optimizer != nil && *optimizer == "adam" {
						*optimizer = *t.Optimizer
					}
					if t.LearningRate != nil && *trainLearningRate == 0.005 {
						*trainLearningRate = *t.LearningRate
					}
					if t.Epochs != nil && *trainEpochs == 8 {
						*trainEpochs = *t.Epochs
					}
					if t.BatchSize != nil && *trainBatchSize == 32 {
						*trainBatchSize = *t.BatchSize
					}
					if t.AdamBeta1 != nil && *adamBeta1 == 0.9 {
						*adamBeta1 = *t.AdamBeta1
					}
					if t.AdamBeta2 != nil && *adamBeta2 == 0.999 {
						*adamBeta2 = *t.AdamBeta2
					}
					if t.AdamEps != nil && *adamEps == 1e-8 {
						*adamEps = *t.AdamEps
					}
					if t.ClipNorm != nil && *clipNorm == 5.0 {
						*clipNorm = *t.ClipNorm
					}
				}
				// Dataset frame-index tunables
				if raw.Tunables != nil && raw.Tunables.DatasetFrameIndex != nil {
					df := raw.Tunables.DatasetFrameIndex
					if df.CacheMaxEntries != nil && *cacheMaxEntries == 2000 {
						*cacheMaxEntries = *df.CacheMaxEntries
					}
					if df.CacheTTLSeconds != nil && *cacheTTL == 5*time.Minute {
						*cacheTTL = time.Duration(*df.CacheTTLSeconds) * time.Second
					}
				}
			} else {
				log.Printf("warning: failed to parse tunables from %s: %v", effectiveRoleConfigPath, jerr)
			}
		}
	}

	// After merging JSON -> runtime defaults, re-apply CLI flags to ensure they take precedence.
	// If the user specified CLI flags different from the defaults, they should override JSON.
	monteSim.SetForceEps(*monteForceEps)
	monteSim.SetForceScale(*monteForceScale)
	monteSim.SetDefenseWeight(*monteDefenseWeight)
	monteSim.SetRoleTargetedWeight(*monteRoleTargetedWeight)
	monteSim.SetRolePasserWeight(*monteRolePasserWeight)
	monteSim.SetRoleCoverageWeight(*monteRoleCoverageWeight)
	monteSim.SetRoleRouteRunnerWeight(*monteRoleRouteWeight)
	monteSim.SetMaxPerPlayer(*monteMaxPerPlayer)
	monteSim.SetLandingNudgeFactor(*monteLandingNudge)

	// Parse explicit role weights if provided (format: key:val,key2:val2) and apply.
	if strings.TrimSpace(*monteRoleWeights) != "" {
		roleMap := make(map[string]float64)
		tokens := strings.Split(*monteRoleWeights, ",")
		for _, tok := range tokens {
			tok = strings.TrimSpace(tok)
			if tok == "" {
				continue
			}
			kv := strings.SplitN(tok, ":", 2)
			if len(kv) != 2 {
				log.Printf("warning: ignoring invalid monte-role-weights token: %q", tok)
				continue
			}
			k := strings.TrimSpace(kv[0])
			vstr := strings.TrimSpace(kv[1])
			if k == "" || vstr == "" {
				log.Printf("warning: ignoring invalid monte-role-weights token: %q", tok)
				continue
			}
			v, err := strconv.ParseFloat(vstr, 64)
			if err != nil {
				log.Printf("warning: invalid weight for role %s: %v", k, err)
				continue
			}
			roleMap[k] = v
		}
		if len(roleMap) > 0 {
			monteSim.SetRoleWeights(roleMap)
			log.Printf("Applied monte role weights: %v", roleMap)
		}
	}

	// Optionally attempt to configure dataset frame-index via Monte helper
	// (works if the underlying dataset implements the corresponding interface).
	monteSim.ConfigureDatasetFrameIndex(*cacheTTL, *cacheMaxEntries)

	// If the user requested a dry-run printout, output the merged effective config and exit.
	if *printEffectiveConfig {
		fmt.Printf("Effective Monte configuration:\n")
		fmt.Printf("  K: %d\n", monteSim.K)
		fmt.Printf("  ForceEps: %f\n", monteSim.ForceEps)
		fmt.Printf("  ForceScale: %f\n", monteSim.ForceScale)
		fmt.Printf("  DefenseWeight: %f\n", monteSim.DefenseWeight)
		fmt.Printf("  MaxPerPlayer: %f\n", monteSim.MaxPerPlayer)
		fmt.Printf("  LandingNudgeFactor: %f\n", monteSim.LandingNudgeFactor)
		fmt.Printf("  RoleTargetedWeight: %f\n", monteSim.RoleTargetedWeight)
		fmt.Printf("  RolePasserWeight: %f\n", monteSim.RolePasserWeight)
		fmt.Printf("  RoleCoverageWeight: %f\n", monteSim.RoleCoverageWeight)
		fmt.Printf("  RoleRouteRunnerWeight: %f\n", monteSim.RoleRouteRunnerWeight)
		fmt.Printf("Precompute settings:\n")
		fmt.Printf("  cache_path: %s\n", *precomputeCache)
		fmt.Printf("  workers: %d\n", precomputeWorkers)
		fmt.Printf("  force: %v\n", precomputeForce)
		fmt.Printf("  progress_interval_seconds: %d\n", precomputeProgressIntervalSeconds)
		fmt.Printf("Training settings:\n")
		fmt.Printf("  optimizer: %s\n", *optimizer)
		fmt.Printf("  learning_rate: %f\n", *trainLearningRate)
		fmt.Printf("  epochs: %d\n", *trainEpochs)
		fmt.Printf("  batch_size: %d\n", *trainBatchSize)
		fmt.Printf("  adam_beta1: %f\n", *adamBeta1)
		fmt.Printf("  adam_beta2: %f\n", *adamBeta2)
		fmt.Printf("  adam_eps: %f\n", *adamEps)
		fmt.Printf("  clip_norm: %f\n", *clipNorm)
		os.Exit(0)
	}

	// Apply CLI-configured Monte tunables so flags override JSON values.
	monteSim.SetForceEps(*monteForceEps)
	monteSim.SetForceScale(*monteForceScale)
	monteSim.SetDefenseWeight(*monteDefenseWeight)
	monteSim.SetRoleTargetedWeight(*monteRoleTargetedWeight)
	monteSim.SetRolePasserWeight(*monteRolePasserWeight)
	monteSim.SetRoleCoverageWeight(*monteRoleCoverageWeight)
	monteSim.SetRoleRouteRunnerWeight(*monteRoleRouteWeight)
	monteSim.SetMaxPerPlayer(*monteMaxPerPlayer)
	monteSim.SetLandingNudgeFactor(*monteLandingNudge)

	// Parse explicit role weights if provided (format: key:val,key2:val2) and apply.
	if strings.TrimSpace(*monteRoleWeights) != "" {
		roleMap := make(map[string]float64)
		tokens := strings.Split(*monteRoleWeights, ",")
		for _, tok := range tokens {
			tok = strings.TrimSpace(tok)
			if tok == "" {
				continue
			}
			kv := strings.SplitN(tok, ":", 2)
			if len(kv) != 2 {
				log.Printf("warning: ignoring invalid monte-role-weights token: %q", tok)
				continue
			}
			k := strings.TrimSpace(kv[0])
			vstr := strings.TrimSpace(kv[1])
			if k == "" || vstr == "" {
				log.Printf("warning: ignoring invalid monte-role-weights token: %q", tok)
				continue
			}
			v, err := strconv.ParseFloat(vstr, 64)
			if err != nil {
				log.Printf("warning: invalid weight for role %s: %v", k, err)
				continue
			}
			roleMap[k] = v
		}
		if len(roleMap) > 0 {
			monteSim.SetRoleWeights(roleMap)
			log.Printf("Applied monte role weights: %v", roleMap)
		}
	}

	// Create and train simple model
	// Configure model input dimension to include K-nearest player features:
	// base features (6) + K * 5 features per neighbor (rel_x, rel_y, dist, side_sign, role_weight)
	inputDim := 6 + (monteSim.K)*5

	cfg := simple.Config{
		HiddenSizes:  []int{64, 32},
		LearningRate: *trainLearningRate,
		Epochs:       *trainEpochs,
		BatchSize:    *trainBatchSize,
		Seed:         *seed,
		Optimizer:    *optimizer,
		Beta1:        *adamBeta1,
		Beta2:        *adamBeta2,
		Epsilon:      *adamEps,
		ClipNorm:     float32(*clipNorm),
		InputDim:     inputDim,
	}
	model, err := simple.NewModel(cfg)
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	// collect training examples: indices where player_to_predict is true and next-frame exists
	maxTrain := predDS.Len()
	if maxTrain > 1024 {
		maxTrain = 1024
	}
	trainingIdxs := make([]int, 0, maxTrain)
	for i := 0; i < predDS.Len() && len(trainingIdxs) < maxTrain; i++ {
		ok, err := predDS.IsPlayerToPredict(i)
		if err != nil || !ok {
			continue
		}
		_, _, found, err := predDS.NextFramePositionForExample(i)
		if err != nil || !found {
			continue
		}
		trainingIdxs = append(trainingIdxs, i)
	}
	if len(trainingIdxs) == 0 {
		// fallback: use first maxTrain examples
		for i := 0; i < maxTrain && i < predDS.Len(); i++ {
			trainingIdxs = append(trainingIdxs, i)
		}
	}
	// instantiate forecastDataset and assign to sub
	fd := &forecastDataset{
		base:    predDS,
		indices: trainingIdxs,
		K:       monteSim.K,
	}
	// If a cache path was provided, try to load it; otherwise precompute and
	// optionally save the computed cache back to disk. Respect the force flag.
	if *precomputeCache != "" {
		if precomputeForce {
			log.Printf("Precompute force enabled: computing and overwriting cache at %s", *precomputeCache)
			if err := fd.Precompute(); err != nil {
				log.Fatalf("failed to precompute forecast dataset: %v", err)
			}
			if serr := fd.SaveCache(*precomputeCache); serr != nil {
				log.Printf("warning: failed to save precomputed cache to %s: %v", *precomputeCache, serr)
			} else {
				log.Printf("Saved precomputed forecast cache to %s", *precomputeCache)
			}
		} else {
			if err := fd.LoadCache(*precomputeCache); err == nil {
				log.Printf("Loaded precomputed forecast cache from %s (examples=%d)", *precomputeCache, fd.Len())
			} else {
				log.Printf("Precomputed cache load failed (%v). Computing and will attempt to save to %s", err, *precomputeCache)
				if err := fd.Precompute(); err != nil {
					log.Fatalf("failed to precompute forecast dataset: %v", err)
				}
				if serr := fd.SaveCache(*precomputeCache); serr != nil {
					log.Printf("warning: failed to save precomputed cache to %s: %v", *precomputeCache, serr)
				} else {
					log.Printf("Saved precomputed forecast cache to %s", *precomputeCache)
				}
			}
		}
	} else {
		// No cache path: compute in-memory for this run.
		log.Printf("Precomputing %d forecast inputs into memory...", fd.Len())
		if err := fd.Precompute(); err != nil {
			log.Fatalf("failed to precompute training set: %v", err)
		}
		log.Printf("Precomputation completed: inputs=%d labels=%d", len(fd.inputs), len(fd.labels))
	}
	sub = fd
	start := time.Now()
	log.Printf("Training forecast model on %d examples (epochs=%d, batch=%d)...", sub.Len(), cfg.Epochs, cfg.BatchSize)
	if err := model.TrainWithDataset(sub); err != nil {
		log.Fatalf("training failed: %v", err)
	}
	log.Printf("Training completed in %v", time.Since(start))

	// Optionally attempt to configure dataset frame-index via Monte helper
	// (works if the underlying dataset implements the corresponding interface).
	monteSim.ConfigureDatasetFrameIndex(*cacheTTL, *cacheMaxEntries)

	// For plotting, collect a modest number of ground-truth landing points from dataset.
	numPlotPoints := min(600, predDS.Len())
	groundXY := make(plotter.XYs, 0, numPlotPoints)
	for i := 0; i < numPlotPoints; i++ {
		_, lab, err := predDS.Example(i)
		if err != nil {
			// skip problematic rows
			continue
		}
		groundXY = append(groundXY, plotter.XY{X: float64(lab[0]), Y: float64(lab[1])})
	}

	// Run the model's predictions for the same set (or a subset) for visual comparison.
	numPred := min(300, predDS.Len())
	predInputs := make([][]float32, 0, numPred)
	for i := 0; i < numPred; i++ {
		// Build the full model input: base features + K nearest neighbor features
		baseInp, _, err := predDS.Example(i)
		if err != nil || len(baseInp) < 2 {
			continue
		}
		neigh, nerr := predDS.FrameKNearestPlayersFeatures(i, monteSim.K)
		if nerr != nil {
			// If neighbor extraction fails for plotting, fall back to a zeroed neighbor vector
			neigh = make([]float32, (monteSim.K)*5)
		}
		inp := make([]float32, 0, len(baseInp)+len(neigh))
		inp = append(inp, baseInp...)
		inp = append(inp, neigh...)
		predInputs = append(predInputs, inp)
	}
	modelPreds, err := model.PredictBatch(predInputs)
	if err != nil {
		log.Fatalf("model prediction failed: %v", err)
	}
	modelXY := make(plotter.XYs, 0, len(modelPreds))
	for _, p := range modelPreds {
		if len(p) >= 2 {
			modelXY = append(modelXY, plotter.XY{X: float64(p[0]), Y: float64(p[1])})
		}
	}

	// Run Monte Carlo simulations for a few example initial states and collect trajectories and landings.
	// We'll pick a small number of initial examples to simulate many draws each.
	numInits := 6
	numSims := *monteSims
	steps := *monteSteps

	var monteLandingXY plotter.XYs
	// we'll also collect a few representative trajectories to draw
	var trajs []trajLine

	// pick initial indices spaced across the dataset (or first ones if small)
	println("Running Monte Carlo simulations...")
	for k := 0; k < numInits; k++ {
		println("Monte Carlo simulation #", k+1)
		idx := (k * predDS.Len()) / numInits
		inp, _, err := predDS.Example(idx)
		if err != nil {
			continue
		}
		// Pass the global example index so Monte can fetch frame players for influence.
		sims, err := monteSim.Simulate(idx, inp, numSims, steps)
		if err != nil {
			log.Printf("monte simulate error for init %d: %v", idx, err)
			continue
		}
		// collect landing points and a few trajectories
		for i, s := range sims {
			monteLandingXY = append(monteLandingXY, plotter.XY{X: float64(s.LandX), Y: float64(s.LandY)})
			// sample some example trajectories (a few per initial)
			if i < 4 {
				xys := make(plotter.XYs, 0, len(s.Trajectory))
				for _, pt := range s.Trajectory {
					xys = append(xys, plotter.XY{X: float64(pt.X), Y: float64(pt.Y)})
				}
				trajs = append(trajs, trajLine{xys: xys})
			}
		}
	}

	// If outCSV is provided, run a numerical evaluation for forecast RMSE and write results.
	// We evaluate model predictions for next-frame (t+1) positions and compare to Monte
	// next-frame predictions (using Monte.PredictNextFrame). If outCSV is empty we fall
	// back to the plotting behavior.
	if *outCSV != "" {
		// Create CSV file
		f, err := os.Create(*outCSV)
		if err != nil {
			log.Fatalf("failed to create output CSV %s: %v", *outCSV, err)
		}
		w := csv.NewWriter(f)
		// header: ground-next, model-pred, model-err, monte-mean, monte-err, monte_samples
		_ = w.Write([]string{
			"idx",
			"ground_next_x",
			"ground_next_y",
			"model_x",
			"model_y",
			"model_error",
			"monte_mean_x",
			"monte_mean_y",
			"monte_mean_error",
			"monte_samples",
		})

		// Evaluate up to evalN examples (cap at dataset length)
		evalCount := min(*evalN, predDS.Len())
		var sumSqModel float64
		var sumSqMonte float64
		valid := 0
		// keep a separate counter for Monte samples that were actually valid
		monteValid := 0

		for i := 0; i < evalCount; i++ {
			// Fetch next-frame ground truth for this example (skip if absent)
			nx, ny, found, err := predDS.NextFramePositionForExample(i)
			if err != nil {
				// skip on read/parse errors
				continue
			}
			if !found {
				// no next-frame label -> skip this example
				continue
			}

			// Build model input: current features + K-nearest flattened neighbor features
			baseInp, _, err := predDS.Example(i)
			if err != nil || len(baseInp) < 2 {
				// skip if we cannot read base inputs
				continue
			}
			neigh, err := predDS.FrameKNearestPlayersFeatures(i, monteSim.K)
			if err != nil {
				continue
			}
			modelInput := make([]float32, 0, len(baseInp)+len(neigh))
			modelInput = append(modelInput, baseInp...)
			modelInput = append(modelInput, neigh...)

			// Model prediction (next-frame)
			mpreds, err := model.PredictBatch([][]float32{modelInput})
			if err != nil || len(mpreds) == 0 {
				continue
			}
			mp := mpreds[0]
			modelErr := math.Hypot(float64(mp[0]-nx), float64(mp[1]-ny))

			// If model produced a non-finite error (NaN/Inf), skip counting this example
			if math.IsNaN(modelErr) || math.IsInf(modelErr, 0) {
				// still attempt to get Monte stats for the row to log, but do not include in RMSE accumulators
				monteMeanX, monteMeanY, merr := monteSim.PredictNextFrame(i, baseInp, *monteSims)
				monteSamples := *monteSims
				monteErr := math.NaN()
				if merr == nil {
					monteErr = math.Hypot(monteMeanX-float64(nx), monteMeanY-float64(ny))
				} else {
					monteMeanX = math.NaN()
					monteMeanY = math.NaN()
					monteErr = math.NaN()
					monteSamples = 0
				}
				if monteSamples > 0 && !math.IsNaN(monteErr) {
					sumSqMonte += monteErr * monteErr
					monteValid++
				}
				row := []string{
					strconv.Itoa(i),
					strconv.FormatFloat(float64(nx), 'f', 6, 64),
					strconv.FormatFloat(float64(ny), 'f', 6, 64),
					strconv.FormatFloat(float64(mp[0]), 'f', 6, 64),
					strconv.FormatFloat(float64(mp[1]), 'f', 6, 64),
					strconv.FormatFloat(math.NaN(), 'f', 6, 64),
					strconv.FormatFloat(monteMeanX, 'f', 6, 64),
					strconv.FormatFloat(monteMeanY, 'f', 6, 64),
					strconv.FormatFloat(monteErr, 'f', 6, 64),
					strconv.Itoa(monteSamples),
				}
				_ = w.Write(row)
				// skip incrementing valid/model accumulators
				continue
			}

			// Monte next-frame prediction (uses neighbor-based empirical next-frame lookups)
			monteMeanX, monteMeanY, merr := monteSim.PredictNextFrame(i, baseInp, *monteSims)
			monteSamples := *monteSims
			monteErr := math.NaN()
			if merr == nil {
				monteErr = math.Hypot(monteMeanX-float64(nx), monteMeanY-float64(ny))
			} else {
				// treat as skipped Monte result
				monteMeanX = math.NaN()
				monteMeanY = math.NaN()
				monteErr = math.NaN()
				monteSamples = 0
			}

			// accumulate RMSE sums for valid comparisons
			sumSqModel += modelErr * modelErr
			if monteSamples > 0 && !math.IsNaN(monteErr) {
				sumSqMonte += monteErr * monteErr
				monteValid++
			}
			valid++

			row := []string{
				strconv.Itoa(i),
				strconv.FormatFloat(float64(nx), 'f', 6, 64),
				strconv.FormatFloat(float64(ny), 'f', 6, 64),
				strconv.FormatFloat(float64(mp[0]), 'f', 6, 64),
				strconv.FormatFloat(float64(mp[1]), 'f', 6, 64),
				strconv.FormatFloat(modelErr, 'f', 6, 64),
				strconv.FormatFloat(monteMeanX, 'f', 6, 64),
				strconv.FormatFloat(monteMeanY, 'f', 6, 64),
				strconv.FormatFloat(monteErr, 'f', 6, 64),
				strconv.Itoa(monteSamples),
			}
			_ = w.Write(row)
		}

		w.Flush()
		_ = f.Close()

		if valid == 0 {
			log.Printf("No valid examples found for forecast evaluation")
		} else {
			rmseModel := math.Sqrt(sumSqModel / float64(valid))
			// Monte RMSE only meaningful if we had valid monte samples; if none, report NaN
			var rmseMonte float64
			if monteValid == 0 {
				rmseMonte = math.NaN()
			} else {
				rmseMonte = math.Sqrt(sumSqMonte / float64(monteValid))
			}
			fmt.Printf("Forecast evaluation over %d examples: RMSE Model = %f, RMSE Monte(mean) = %f\n", valid, rmseModel, rmseMonte)
		}
	} else {
		// Plot overlay: ground truth, model preds, monte landings, and some trajectories.
		if err := plotCompare(*outDir, groundXY, modelXY, monteLandingXY, trajs); err != nil {
			log.Fatalf("failed to generate plot: %v", err)
		}

		log.Printf("Comparison plots written to %s", *outDir)
	}
}

// plotCompare writes a PNG visualizing dataset labels (grey), model predictions (blue),
// monte landings (red), and sample trajectories (sem-transparent lines).
func plotCompare(outDir string, ground, model, monte plotter.XYs, trajs []trajLine) error {
	p := plot.New()
	p.Title.Text = "Landing points: dataset (grey), model (blue), monte (red)"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	// Ground truth scatter (small grey points)
	gr, err := plotter.NewScatter(ground)
	if err != nil {
		return err
	}
	gr.GlyphStyle.Color = color.RGBA{R: 120, G: 120, B: 120, A: 180}
	gr.GlyphStyle.Radius = vg.Points(1.8)
	p.Add(gr)
	p.Legend.Add("dataset", gr)

	// Model predictions scatter (blue)
	mp, err := plotter.NewScatter(model)
	if err != nil {
		return err
	}
	mp.GlyphStyle.Color = color.RGBA{R: 20, G: 80, B: 200, A: 220}
	mp.GlyphStyle.Radius = vg.Points(2.8)
	p.Add(mp)
	p.Legend.Add("model", mp)

	// Monte landings scatter (red, semi-transparent)
	ml, err := plotter.NewScatter(monte)
	if err != nil {
		return err
	}
	ml.GlyphStyle.Color = color.RGBA{R: 200, G: 30, B: 30, A: 180}
	ml.GlyphStyle.Radius = vg.Points(1.8)
	p.Add(ml)
	p.Legend.Add("monte", ml)

	// Trajectories: draw a few lines with faint alpha
	for i, t := range trajs {
		if len(t.xys) == 0 {
			continue
		}
		line, err := plotter.NewLine(t.xys)
		if err != nil {
			return err
		}
		// cycle colors a bit
		col := color.RGBA{R: 40, G: 120, B: 40, A: uint8(100 + (i%3)*30)}
		line.Color = col
		line.Width = vg.Points(0.8)
		p.Add(line)
		if i == 0 {
			p.Legend.Add("trajectories (sample)", line)
		}
	}

	// Auto-range with a small padding
	grid := plotter.NewGrid()
	p.Add(grid)
	all := append(append(ground, model...), monte...)
	xmin, xmax, ymin, ymax := autoRange(all)
	p.X.Min = xmin
	p.X.Max = xmax
	p.Y.Min = ymin
	p.Y.Max = ymax

	// Ensure output directory exists
	if err := ensureDir(outDir); err != nil {
		return err
	}
	outPath := filepath.Join(outDir, "compare_landing.png")
	if err := p.Save(8*vg.Inch, 6*vg.Inch, outPath); err != nil {
		return err
	}
	return nil
}

// autoRange computes padded min/max for X and Y for a set of points.
func autoRange(xs plotter.XYs) (xmin, xmax, ymin, ymax float64) {
	if len(xs) == 0 {
		return -1, 1, -1, 1
	}
	xmin = math.Inf(1)
	xmax = math.Inf(-1)
	ymin = math.Inf(1)
	ymax = math.Inf(-1)
	for _, p := range xs {
		if p.X < xmin {
			xmin = p.X
		}
		if p.X > xmax {
			xmax = p.X
		}
		if p.Y < ymin {
			ymin = p.Y
		}
		if p.Y > ymax {
			ymax = p.Y
		}
	}
	padx := (xmax - xmin) * 0.06
	pady := (ymax - ymin) * 0.06
	if padx == 0 {
		padx = 1.0
	}
	if pady == 0 {
		pady = 1.0
	}
	return xmin - padx, xmax + padx, ymin - pady, ymax + pady
}

func ensureDir(path string) error {
	// Attempt to create directory if it doesn't exist (silently succeed if present).
	if path == "" {
		return nil
	}
	return os.MkdirAll(path, 0755)
}
