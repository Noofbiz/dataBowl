// Command compare evaluates both the ML (MLP) and Monte Carlo simulation
// methods against ground-truth training data and prints a side-by-side
// accuracy report.
//
// Usage:
//
//	go run ./cmd/compare [flags]
//
// Flags:
//
//	-input       glob for training input CSVs   (default: datasets/assets/prediction/train/input*.csv)
//	-output      glob for training output CSVs  (default: datasets/assets/prediction/train/output*.csv)
//	-eval-frac   fraction of games held out for evaluation (default: 0.2)
//	-max-games   cap total games used (0 = all); useful for quick runs (default: 0)
//	-epochs      ML training epochs             (default: 10)
//	-lr          Adam learning rate             (default: 0.001)
//	-hidden      hidden layer widths            (default: 128,64,32)
//	-infer-batch ML inference batch size        (default: 256)
//	-runs        Monte Carlo runs per example   (default: 200)
//	-seed        RNG seed for force table       (default: 42)
//
// How it works:
//
//  1. Load the full dataset and split game IDs: the last (eval-frac) fraction
//     of sorted games is held out; the remainder is used for training.
//  2. ML path: train an MLP on the train-split games, then run inference in
//     small fixed-size batches (infer-batch rows each) to avoid OOM.
//  3. Monte Carlo path: build a ForceTable from the train-split games (raw
//     *Game structs via GameIter), then simulate each eval example.
//  4. Score both sets of predictions against TrueX/TrueY using:
//       - MAE    (mean absolute Euclidean distance, yards)
//       - RMSE   (root mean squared Euclidean distance)
//       - Median error
//     broken down by output-frame bucket (early: 1–5, mid: 6–15, late: 16+).
//  5. Print a formatted comparison table to stdout.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/Noofbiz/dataBowl/ml"
	"github.com/Noofbiz/dataBowl/simulate"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	inputGlob  := flag.String("input",      "datasets/assets/prediction/train/input*.csv",  "glob for training input CSVs")
	outputGlob := flag.String("output",     "datasets/assets/prediction/train/output*.csv", "glob for training output CSVs")
	evalFrac   := flag.Float64("eval-frac",  0.2,       "fraction of games held out for evaluation")
	maxGames   := flag.Int("max-games",      0,         "cap total games (0 = all); use for quick runs")
	epochs     := flag.Int("epochs",         10,        "ML training epochs")
	lr         := flag.Float64("lr",         0.001,     "Adam learning rate")
	hiddenStr  := flag.String("hidden",      "128,64,32","comma-separated hidden layer widths")
	inferBatch := flag.Int("infer-batch",    256,       "ML inference batch size (rows)")
	runs       := flag.Int("runs",           200,       "Monte Carlo runs per example")
	seed       := flag.Int64("seed",         42,        "RNG seed for force table")
	flag.Parse()

	hidden, err := parseHidden(*hiddenStr)
	if err != nil {
		log.Fatalf("invalid -hidden %q: %v", *hiddenStr, err)
	}

	// ── 1. Index and split ────────────────────────────────────────────────
	log.Println("indexing dataset…")
	ds, err := datasets.NewPredictionDataset(*inputGlob, *outputGlob)
	if err != nil {
		log.Fatalf("index dataset: %v", err)
	}

	allGames := ds.GameOrder()
	if *maxGames > 0 && *maxGames < len(allGames) {
		allGames = allGames[:*maxGames]
	}
	nEval    := max(1, int(math.Round(float64(len(allGames))**evalFrac)))
	trainIDs := allGames[:len(allGames)-nEval]
	evalIDs  := allGames[len(allGames)-nEval:]
	log.Printf("games: %d train, %d eval", len(trainIDs), len(evalIDs))

	// trainDS is a lazy view restricted to the training split.
	trainDS := ds.Subset(trainIDs)

	// ── 2. Train ML model ─────────────────────────────────────────────────
	log.Printf("training ML model (%d epochs)…", *epochs)
	model, err := ml.New(ml.Config{HiddenSizes: hidden, LearningRate: *lr})
	if err != nil {
		log.Fatalf("build model: %v", err)
	}
	if err := model.TrainEpochs(trainDS, *epochs); err != nil {
		log.Fatalf("train: %v", err)
	}
	log.Println("ML training complete")

	// ── 3. Build ForceTable from training games ───────────────────────────
	log.Println("building force table…")
	var trainGames []*datasets.Game
	for g := range trainDS.GameIter() {
		trainGames = append(trainGames, g)
	}
	ft  := simulate.BuildForceTable(trainGames, *seed)
	sim := simulate.NewSimulator(ft)
	log.Println("force table ready")

	// ── 4. Collect labelled eval examples ─────────────────────────────────
	log.Printf("collecting eval examples from %d games…", len(evalIDs))
	var evalExs []datasets.EvalExample
	for _, gid := range evalIDs {
		exs, err := ds.GameExamples(gid)
		if err != nil {
			log.Printf("warning: %v", err)
			continue
		}
		evalExs = append(evalExs, exs...)
	}
	log.Printf("eval examples: %d", len(evalExs))
	if len(evalExs) == 0 {
		log.Fatal("no eval examples — widen -eval-frac or check data paths")
	}

	// ── 5. ML predictions (fixed-size batches to avoid OOM) ───────────────
	log.Printf("running ML inference (batch=%d)…", *inferBatch)
	mlPreds := mlPredict(model, evalExs, *inferBatch)

	// ── 6. Monte Carlo predictions ────────────────────────────────────────
	log.Printf("running Monte Carlo simulation (%d runs/example)…", *runs)
	mcPreds := mcPredict(sim, evalExs, *runs)

	// ── 7. Score and report ───────────────────────────────────────────────
	fmt.Println()
	printReport(evalExs, mlPreds, mcPreds)
}

// ── ML inference ─────────────────────────────────────────────────────────────

// mlPredict runs inference in chunks of batchSize rows to bound memory use.
// All rows in one chunk share the same tensor shape, so XLA compiles only
// O(ceil(N/batchSize)) distinct graphs instead of O(N).
func mlPredict(m *ml.PlayerPositionModel, exs []datasets.EvalExample, batchSize int) [][2]float64 {
	preds := make([][2]float64, len(exs))
	for start := 0; start < len(exs); start += batchSize {
		end := start + batchSize
		if end > len(exs) {
			end = len(exs)
		}
		chunk := exs[start:end]
		n := len(chunk)

		// Pad the chunk to exactly batchSize so every call uses the same shape,
		// letting XLA reuse its compiled graph after the first two calls
		// (batchSize and the final smaller batch).
		flat := make([]float32, batchSize*datasets.InputFeatureLen)
		for i, ex := range chunk {
			copy(flat[i*datasets.InputFeatureLen:], ex.Inputs)
		}

		xs, ys, err := m.Predict(flat)
		if err != nil {
			log.Fatalf("ML predict chunk [%d,%d): %v", start, end, err)
		}
		for i := range n {
			preds[start+i] = [2]float64{float64(xs[i]), float64(ys[i])}
		}
	}
	return preds
}

// ── Monte Carlo inference ─────────────────────────────────────────────────────

// mcPredict runs Monte Carlo simulation for each eval example.
// Trajectories are cached by initial kinematic state+role so that all output
// frames of the same player-play share one simulation.
func mcPredict(sim *simulate.Simulator, exs []datasets.EvalExample, runs int) [][2]float64 {
	type key struct {
		x, y, speed, dir float64
		role              string
		numFrames         int
	}

	// First pass: find the max FrameID needed per starting state so one
	// simulation covers the full trajectory.
	maxFor := make(map[key]int)
	for _, ex := range exs {
		k := key{r6(ex.InitX), r6(ex.InitY), r6(ex.InitSpeed), r6(ex.InitDir), ex.Role, ex.NumFrames}
		if ex.FrameID > maxFor[k] {
			maxFor[k] = ex.FrameID
		}
	}

	cache := make(map[key]simulate.Trajectory)
	preds := make([][2]float64, len(exs))

	for i, ex := range exs {
		k := key{r6(ex.InitX), r6(ex.InitY), r6(ex.InitSpeed), r6(ex.InitDir), ex.Role, ex.NumFrames}
		traj, ok := cache[k]
		if !ok {
			state := simulate.State{
				Pos:   simulate.Vec2{X: ex.InitX, Y: ex.InitY},
				Speed: ex.InitSpeed,
				Dir:   ex.InitDir,
			}
			results := sim.Run(ex.Role, state, runs, maxFor[k])
			traj = simulate.MeanTrajectory(results)
			cache[k] = traj
		}
		idx := ex.FrameID
		if idx >= len(traj) {
			idx = len(traj) - 1
		}
		preds[i] = [2]float64{traj[idx].X, traj[idx].Y}
	}
	return preds
}

// ── Metrics and reporting ─────────────────────────────────────────────────────

type metrics struct {
	errors []float64
}

func (m *metrics) add(predX, predY, trueX, trueY float64) {
	dx := predX - trueX
	dy := predY - trueY
	m.errors = append(m.errors, math.Sqrt(dx*dx+dy*dy))
}

func (m *metrics) mae() float64 {
	if len(m.errors) == 0 {
		return math.NaN()
	}
	var s float64
	for _, e := range m.errors {
		s += e
	}
	return s / float64(len(m.errors))
}

func (m *metrics) rmse() float64 {
	if len(m.errors) == 0 {
		return math.NaN()
	}
	var s float64
	for _, e := range m.errors {
		s += e * e
	}
	return math.Sqrt(s / float64(len(m.errors)))
}

func (m *metrics) median() float64 {
	if len(m.errors) == 0 {
		return math.NaN()
	}
	sorted := make([]float64, len(m.errors))
	copy(sorted, m.errors)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

func (m *metrics) n() int { return len(m.errors) }

type methodMetrics struct {
	all   metrics
	early metrics // frames 1–5
	mid   metrics // frames 6–15
	late  metrics // frames 16+
}

func (mm *methodMetrics) add(frameID int, predX, predY, trueX, trueY float64) {
	mm.all.add(predX, predY, trueX, trueY)
	switch {
	case frameID <= 5:
		mm.early.add(predX, predY, trueX, trueY)
	case frameID <= 15:
		mm.mid.add(predX, predY, trueX, trueY)
	default:
		mm.late.add(predX, predY, trueX, trueY)
	}
}

func printReport(exs []datasets.EvalExample, mlPreds, mcPreds [][2]float64) {
	var mlM, mcM methodMetrics
	for i, ex := range exs {
		tx, ty := float64(ex.TrueX), float64(ex.TrueY)
		mlM.add(ex.FrameID, mlPreds[i][0], mlPreds[i][1], tx, ty)
		mcM.add(ex.FrameID, mcPreds[i][0], mcPreds[i][1], tx, ty)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)

	fmt.Fprintln(w, "=== Prediction Accuracy Comparison (Euclidean distance, yards) ===")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Method\tN\tMAE\tRMSE\tMedian")
	fmt.Fprintln(w, "------\t-\t---\t----\t------")
	printRow(w, "ML (MLP)      [all]", &mlM.all)
	printRow(w, "Monte Carlo   [all]", &mcM.all)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "--- Early frames (1–5) ---")
	printRow(w, "ML (MLP)",    &mlM.early)
	printRow(w, "Monte Carlo", &mcM.early)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "--- Mid frames (6–15) ---")
	printRow(w, "ML (MLP)",    &mlM.mid)
	printRow(w, "Monte Carlo", &mcM.mid)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "--- Late frames (16+) ---")
	printRow(w, "ML (MLP)",    &mlM.late)
	printRow(w, "Monte Carlo", &mcM.late)
	fmt.Fprintln(w)
	w.Flush()

	fmt.Println("Winner (by MAE):")
	printWinner("All",   &mlM.all,   &mcM.all)
	printWinner("Early", &mlM.early, &mcM.early)
	printWinner("Mid",   &mlM.mid,   &mcM.mid)
	printWinner("Late",  &mlM.late,  &mcM.late)
}

func printRow(w *tabwriter.Writer, name string, m *metrics) {
	if m.n() == 0 {
		fmt.Fprintf(w, "%s\t0\t–\t–\t–\n", name)
		return
	}
	fmt.Fprintf(w, "%s\t%d\t%.4f\t%.4f\t%.4f\n",
		name, m.n(), m.mae(), m.rmse(), m.median())
}

func printWinner(label string, ml, mc *metrics) {
	if ml.n() == 0 {
		fmt.Printf("  %-6s  no data\n", label)
		return
	}
	diff := ml.mae() - mc.mae()
	const tieThreshold = 0.01
	switch {
	case math.Abs(diff) < tieThreshold:
		fmt.Printf("  %-6s  tie             (ΔMAE = %.4f yds)\n", label, math.Abs(diff))
	case diff > 0:
		fmt.Printf("  %-6s  Monte Carlo wins (ΔMAE = %.4f yds)\n", label, diff)
	default:
		fmt.Printf("  %-6s  ML (MLP) wins    (ΔMAE = %.4f yds)\n", label, -diff)
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func r6(v float64) float64 { return math.Round(v*1e6) / 1e6 }

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parseHidden(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	sizes := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("expected positive integer, got %q", p)
		}
		sizes = append(sizes, n)
	}
	if len(sizes) == 0 {
		return nil, fmt.Errorf("must specify at least one hidden layer width")
	}
	return sizes, nil
}


