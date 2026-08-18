// Command train trains an MLP on the NFL Big Data Bowl prediction dataset and
// writes test-set predictions to a CSV submission file.
//
// Usage:
//
//	go run ./cmd/train [flags]
//
// Flags:
//
//	-train-input   glob pattern for training input CSVs  (default: datasets/assets/prediction/train/input*.csv)
//	-train-output  glob pattern for training output CSVs (default: datasets/assets/prediction/train/output*.csv)
//	-test-input    path to test_input.csv                (default: datasets/assets/prediction/test_input.csv)
//	-test-template path to test.csv submission template  (default: datasets/assets/prediction/test.csv)
//	-out           output directory                       (default: out)
//	-epochs        number of training epochs              (default: 10)
//	-lr            Adam learning rate                     (default: 0.001)
//	-hidden        comma-separated hidden layer widths    (default: 128,64,32)
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/Noofbiz/dataBowl/ml"
	_ "github.com/gomlx/gomlx/backends/default" // register xla/go compute backends
)

func main() {
	trainInput  := flag.String("train-input",   "datasets/assets/prediction/train/input*.csv", "glob for training input CSVs")
	trainOutput := flag.String("train-output",  "datasets/assets/prediction/train/output*.csv", "glob for training output CSVs")
	testInput   := flag.String("test-input",    "datasets/assets/prediction/test_input.csv", "test input CSV")
	testTmpl    := flag.String("test-template", "datasets/assets/prediction/test.csv", "test submission template CSV")
	outDir      := flag.String("out",           "out", "directory to write predictions")
	epochs      := flag.Int("epochs",           10, "number of training epochs")
	lr          := flag.Float64("lr",           0.001, "Adam learning rate")
	hiddenStr   := flag.String("hidden",        "128,64,32", "comma-separated hidden layer widths")
	flag.Parse()

	// Parse hidden layer widths.
	hidden, err := parseHidden(*hiddenStr)
	if err != nil {
		log.Fatalf("invalid -hidden %q: %v", *hiddenStr, err)
	}

	// ── 1. Build dataset ───────────────────────────────────────────────────
	log.Println("indexing training data…")
	ds, err := datasets.NewPredictionDataset(*trainInput, *trainOutput)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	log.Printf("dataset ready: %d games", ds.NumGames())

	// ── 2. Build model ─────────────────────────────────────────────────────
	log.Println("building model…")
	m, err := ml.New(ml.Config{
		HiddenSizes:  hidden,
		LearningRate: *lr,
	})
	if err != nil {
		log.Fatalf("build model: %v", err)
	}

	// ── 3. Train ───────────────────────────────────────────────────────────
	log.Printf("training for %d epoch(s)…", *epochs)
	if err := m.TrainEpochs(ds, *epochs); err != nil {
		log.Fatalf("training: %v", err)
	}
	log.Println("training complete")

	// ── 4. Load test data ──────────────────────────────────────────────────
	log.Println("loading test data…")
	testRows, err := datasets.LoadTestData(*testInput, *testTmpl)
	if err != nil {
		log.Fatalf("load test data: %v", err)
	}
	log.Printf("test rows: %d", len(testRows))
	if len(testRows) == 0 {
		log.Fatal("no test rows resolved; check test input/template paths")
	}

	// ── 5. Run inference ───────────────────────────────────────────────────
	log.Println("running inference…")
	flatInputs := make([]float32, len(testRows)*datasets.InputFeatureLen)
	for i, row := range testRows {
		copy(flatInputs[i*datasets.InputFeatureLen:], row.Inputs)
	}

	xs, ys, err := m.Predict(flatInputs)
	if err != nil {
		log.Fatalf("predict: %v", err)
	}

	// ── 6. Write submission CSV ────────────────────────────────────────────
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("create output dir %q: %v", *outDir, err)
	}
	outPath := filepath.Join(*outDir, "submission.csv")
	if err := writeSubmission(outPath, testRows, xs, ys); err != nil {
		log.Fatalf("write submission: %v", err)
	}
	log.Printf("predictions written to %s", outPath)
}

// writeSubmission writes id,x,y rows to path.
func writeSubmission(path string, rows []datasets.TestRow, xs, ys []float32) error {
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
		rec := []string{
			row.ID,
			strconv.FormatFloat(float64(xs[i]), 'f', 6, 32),
			strconv.FormatFloat(float64(ys[i]), 'f', 6, 32),
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// parseHidden splits a comma-separated string of integers into a []int.
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
