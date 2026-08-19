// Command montecarlo uses data-driven Monte Carlo simulation to predict future
// NFL player positions and writes a submission.csv identical in format to the
// one produced by cmd/train.
//
// Usage:
//
//	go run ./cmd/montecarlo [flags]
//
// Flags:
//
//	-train-input   glob for training input CSVs  (default: datasets/assets/prediction/train/input*.csv)
//	-test-input    path to test_input.csv         (default: datasets/assets/prediction/test_input.csv)
//	-test-template path to test.csv template      (default: datasets/assets/prediction/test.csv)
//	-out           output directory               (default: out)
//	-runs          Monte Carlo runs per row        (default: 200)
//	-seed          RNG seed for force table        (default: 42)
package main

import (
	"encoding/csv"
	"flag"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/Noofbiz/dataBowl/simulate"
)

func main() {
	trainInput := flag.String("train-input", "datasets/assets/prediction/train/input*.csv", "glob for training input CSVs")
	testInput := flag.String("test-input", "datasets/assets/prediction/test_input.csv", "test input CSV")
	testTmpl := flag.String("test-template", "datasets/assets/prediction/test.csv", "submission template CSV")
	outDir := flag.String("out", "out", "output directory")
	runs := flag.Int("runs", 200, "Monte Carlo runs per test row")
	seed := flag.Int64("seed", 42, "RNG seed for the force table")
	flag.Parse()

	// ── 1. Index training dataset (no CSVs loaded yet) ─────────────────────
	log.Println("indexing training data…")
	// We pass an empty output pattern — the Monte Carlo approach only needs
	// input frames (no ground-truth labels required for force extraction).
	ds, err := datasets.NewPredictionDataset(*trainInput, "")
	if err != nil {
		log.Fatalf("index dataset: %v", err)
	}
	log.Printf("indexed %d games", ds.NumGames())

	// ── 2. Build ForceTable by streaming every training game ───────────────
	log.Println("building force table from training games…")
	var trainingGames []*datasets.Game
	for g := range ds.GameIter() {
		trainingGames = append(trainingGames, g)
	}
	log.Printf("loaded %d games for force extraction", len(trainingGames))

	ft := simulate.BuildForceTable(trainingGames, *seed)
	log.Println("force table ready")

	// ── 3. Load test rows ──────────────────────────────────────────────────
	log.Println("loading test data…")
	testRows, err := datasets.LoadTestData(*testInput, *testTmpl)
	if err != nil {
		log.Fatalf("load test data: %v", err)
	}
	log.Printf("test rows: %d", len(testRows))
	if len(testRows) == 0 {
		log.Fatal("no test rows resolved; check -test-input and -test-template paths")
	}

	// ── 4. Simulate — one run per unique starting state ───────────────────
	// Pre-scan to find the maximum OutFrameID needed per unique initial state
	// so each cached simulation covers all future frames for that player-play.
	type simKey struct {
		x, y, speed, dir float64
		role             string
	}

	maxFrames := make(map[simKey]int)
	for _, row := range testRows {
		key := simKey{
			x:     round6(row.InitX),
			y:     round6(row.InitY),
			speed: round6(row.InitSpeed),
			dir:   round6(row.InitDir),
			role:  row.Role,
		}
		if row.OutFrameID > maxFrames[key] {
			maxFrames[key] = row.OutFrameID
		}
	}

	sim := simulate.NewSimulator(ft)
	cache := make(map[simKey]simulate.Trajectory)

	log.Printf("simulating %d test rows (%d runs each)…", len(testRows), *runs)

	xs := make([]float64, len(testRows))
	ys := make([]float64, len(testRows))

	for i, row := range testRows {
		key := simKey{
			x:     round6(row.InitX),
			y:     round6(row.InitY),
			speed: round6(row.InitSpeed),
			dir:   round6(row.InitDir),
			role:  row.Role,
		}

		mean, ok := cache[key]
		if !ok {
			initState := simulate.State{
				Pos:   simulate.Vec2{X: row.InitX, Y: row.InitY},
				Speed: row.InitSpeed,
				Dir:   row.InitDir,
			}
			nSteps := maxFrames[key]
			results := sim.Run(row.Role, initState, *runs, nSteps)
			mean = simulate.MeanTrajectory(results)
			cache[key] = mean
		}

		// mean[0] = initial position, mean[k] = position after k steps.
		idx := row.OutFrameID
		if idx >= len(mean) {
			idx = len(mean) - 1
		}
		if idx < 0 {
			idx = 0
		}
		xs[i] = mean[idx].X
		ys[i] = mean[idx].Y
	}

	// ── 5. Write submission CSV ────────────────────────────────────────────
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("create output dir %q: %v", *outDir, err)
	}
	outPath := filepath.Join(*outDir, "submission.csv")
	if err := writeSubmission(outPath, testRows, xs, ys); err != nil {
		log.Fatalf("write submission: %v", err)
	}
	log.Printf("predictions written to %s", outPath)
}

// writeSubmission writes id,x,y to path.
func writeSubmission(path string, rows []datasets.TestRow, xs, ys []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write([]string{"id", "x", "y"}); err != nil {
		return err
	}
	for i, row := range rows {
		if err := w.Write([]string{
			row.ID,
			strconv.FormatFloat(xs[i], 'f', 6, 64),
			strconv.FormatFloat(ys[i], 'f', 6, 64),
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// round6 rounds to 6 decimal places for use as a map key.
func round6(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
