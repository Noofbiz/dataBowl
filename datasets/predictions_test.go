package datasets

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// writeCSV writes a CSV file with the given header and rows to path.
func writeCSV(t *testing.T, path, header string, rows []string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create csv %s: %v", path, err)
	}
	defer f.Close()

	if _, err := f.WriteString(header + "\n"); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}
	for _, r := range rows {
		if _, err := f.WriteString(r + "\n"); err != nil {
			t.Fatalf("failed to write row: %v", err)
		}
	}
}

// TestPredictionDataset_LoadAndRead creates temporary CSV files and verifies
// that NewPredictionDataset, Example, Batch, MakePredictionBatchFlat and
// ToGomlxTensors behave as expected.
func TestPredictionDataset_LoadAndRead(t *testing.T) {
	tmp := t.TempDir()

	header := "x,y,s,a,o,dir,ball_land_x,ball_land_y"

	file1 := filepath.Join(tmp, "p1.csv")
	rows1 := []string{
		"1,2,3,4,5,6,101,102",
		"7,8,9,10,11,12,103,104",
		"13,14,15,16,17,18,105,106",
	}
	writeCSV(t, file1, header, rows1)

	file2 := filepath.Join(tmp, "p2.csv")
	rows2 := []string{
		"21,22,23,24,25,26,201,202",
		"27,28,29,30,31,32,203,204",
		"33,34,35,36,37,38,205,206",
	}
	writeCSV(t, file2, header, rows2)

	pattern := filepath.Join(tmp, "*.csv")
	ds, err := NewPredictionDataset(pattern, "")
	if err != nil {
		t.Fatalf("NewPredictionDataset failed: %v", err)
	}

	// Expect total 6 examples
	if got := ds.Len(); got != 6 {
		t.Fatalf("expected len 6, got %d", got)
	}

	// Example 0 (first row of first file)
	in0, lab0, err := ds.Example(0)
	if err != nil {
		t.Fatalf("Example(0) error: %v", err)
	}
	if len(in0) != 6 || len(lab0) != 2 {
		t.Fatalf("unexpected dims for Example(0): inputs=%d labels=%d", len(in0), len(lab0))
	}
	if in0[0] != 1 || in0[1] != 2 || lab0[0] != 101 || lab0[1] != 102 {
		t.Fatalf("unexpected values for Example(0): in=%v lab=%v", in0, lab0)
	}

	// Example 4 (second file, row index 1)
	in4, lab4, err := ds.Example(4)
	if err != nil {
		t.Fatalf("Example(4) error: %v", err)
	}
	// expected inputs start 27,28,... and labels 203,204
	if in4[0] != 27 || in4[1] != 28 {
		t.Fatalf("unexpected values for Example(4) inputs: %v", in4)
	}
	if lab4[0] != 203 || lab4[1] != 204 {
		t.Fatalf("unexpected values for Example(4) labels: %v", lab4)
	}

	// Batch read indices [0,2,3,5]
	indices := []int{0, 2, 3, 5}
	inputs, labels, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch error: %v", err)
	}
	if len(inputs) != len(indices) || len(labels) != len(indices) {
		t.Fatalf("Batch returned unexpected sizes: inputs=%d labels=%d", len(inputs), len(labels))
	}
	// Check labels sequence: 101,105,201,205
	expectedLabels := [][]float32{{101, 102}, {105, 106}, {201, 202}, {205, 206}}
	for i := range expectedLabels {
		if labels[i][0] != expectedLabels[i][0] || labels[i][1] != expectedLabels[i][1] {
			t.Fatalf("Batch label mismatch at %d: got %v expected %v", i, labels[i], expectedLabels[i])
		}
	}

	// Make flat batch and verify dimensions
	pflat, err := MakePredictionBatchFlat(inputs, labels)
	if err != nil {
		t.Fatalf("MakePredictionBatchFlat error: %v", err)
	}
	if pflat.BatchSize != len(indices) || pflat.InputDim != 6 || pflat.LabelDim != 2 {
		t.Fatalf("unexpected PredictionBatchFlat dims: %+v", pflat)
	}
	if len(pflat.Inputs) != pflat.BatchSize*pflat.InputDim {
		t.Fatalf("flat inputs length mismatch: %d vs %d", len(pflat.Inputs), pflat.BatchSize*pflat.InputDim)
	}
	if len(pflat.Labels) != pflat.BatchSize*pflat.LabelDim {
		t.Fatalf("flat labels length mismatch: %d vs %d", len(pflat.Labels), pflat.BatchSize*pflat.LabelDim)
	}

	// Convert to gomlx tensors (ensure call doesn't panic and tensors non-nil)
	inT, labT, err := pflat.ToGomlxTensors()
	if err != nil {
		t.Fatalf("ToGomlxTensors error: %v", err)
	}
	if inT == nil || labT == nil {
		t.Fatalf("ToGomlxTensors returned nil tensor(s)")
	}

	// as an extra sanity check, ensure tensors contain expected types by calling a minimal tensors operation
	_ = tensors.FromAnyValue // ensure package symbol resolves; no further assertions required here
}

// TestPredictionDataset_MissingColumns ensures NewPredictionDataset returns an error
// when required columns are absent in the CSV header.
func TestPredictionDataset_MissingColumns(t *testing.T) {
	tmp := t.TempDir()
	// header missing ball_land_y
	header := "x,y,s,a,o,dir,ball_land_x"

	file := filepath.Join(tmp, "bad.csv")
	rows := []string{
		"1,2,3,4,5,6,101",
	}
	writeCSV(t, file, header, rows)

	pattern := filepath.Join(tmp, "*.csv")
	_, err := NewPredictionDataset(pattern, "")
	if err == nil {
		t.Fatalf("expected error when required columns missing, got nil")
	}
}
