package datasets

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// writeCSVFile writes a CSV file (simple, comma-delimited) at path with the provided header and rows.
// Each row should already be a comma-separated string (easier for test construction).
func writeCSVFile(t *testing.T, path string, header string, rows []string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create csv file %s: %v", path, err)
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

// TestAnalyticsDataset_VariableLengths verifies that AnalyticsDataset correctly groups
// rows by play id when plays have varying lengths and that Batch and Example return
// correctly-shaped sequences. It also verifies MakeAnalyticsBatchFlat fails for
// inconsistent sequence lengths.
func TestAnalyticsDataset_VariableLengths(t *testing.T) {
	tmp := t.TempDir()

	// Header includes play_id and two sequence columns x,y.
	header := "play_id,x,y"

	// Create rows for three plays with lengths 3,2,1 respectively.
	// Play IDs: pA (3 rows), pB (2 rows), pC (1 row)
	rows := []string{
		"pA,1.0,1.1",
		"pA,1.2,1.3",
		"pA,1.4,1.5",
		"pB,2.0,2.1",
		"pB,2.2,2.3",
		"pC,3.0,3.1",
	}

	p := filepath.Join(tmp, "analytics.csv")
	writeCSVFile(t, p, header, rows)

	pattern := filepath.Join(tmp, "*.csv")
	seqCols := []string{"x", "y"}

	ds, err := NewAnalyticsDataset(pattern, "play_id", seqCols)
	if err != nil {
		t.Fatalf("NewAnalyticsDataset failed: %v", err)
	}

	// Expect 3 unique plays
	if got := ds.Len(); got != 3 {
		t.Fatalf("expected 3 plays, got %d", got)
	}

	// Read all plays via Batch
	idx := make([]int, ds.Len())
	for i := range idx {
		idx[i] = i
	}
	buffers, shapes, err := ds.Batch(idx)
	if err != nil {
		t.Fatalf("Batch failed: %v", err)
	}
	if len(buffers) != 3 || len(shapes) != 3 {
		t.Fatalf("Batch returned unexpected sizes: buffers=%d shapes=%d", len(buffers), len(shapes))
	}

	// Shapes should correspond to the expected set {3,2,1} (time) and channels=2
	expectedCounts := map[int]bool{3: true, 2: true, 1: true}
	foundCounts := make(map[int]bool)
	totalElements := 0
	for i, sh := range shapes {
		if len(sh) != 2 {
			t.Fatalf("shape[%d] unexpected dims: %v", i, sh)
		}
		timeSteps := sh[0]
		channels := sh[1]
		if channels != 2 {
			t.Fatalf("expected 2 channels, got %d", channels)
		}
		if !expectedCounts[timeSteps] {
			t.Fatalf("unexpected timeSteps %d found in shape[%d]", timeSteps, i)
		}
		foundCounts[timeSteps] = true
		totalElements += timeSteps * channels
		// buffer size should equal timeSteps * channels
		if len(buffers[i]) != timeSteps*channels {
			t.Fatalf("buffer[%d] length mismatch: got %d expected %d", i, len(buffers[i]), timeSteps*channels)
		}
	}

	// ensure we observed all expected counts
	for k := range expectedCounts {
		if !foundCounts[k] {
			t.Fatalf("expected timeSteps %d not observed among shapes", k)
		}
	}

	// MakeAnalyticsBatchFlat should fail because sequences have different lengths
	if _, err := MakeAnalyticsBatchFlat(buffers, shapes); err == nil {
		t.Fatalf("expected MakeAnalyticsBatchFlat to error for inconsistent shapes, but it succeeded")
	}

	// Total elements in buffers should equal number of rows * channels (here 6 rows * 2 = 12)
	if totalElements != 6*2 {
		t.Fatalf("expected totalElements 12, got %d", totalElements)
	}
}

// TestAnalyticsDataset_FixedLengthBatch verifies MakeAnalyticsBatchFlat and ToGomlxTensor
// succeed when all sequences have the same time length and channel count.
func TestAnalyticsDataset_FixedLengthBatch(t *testing.T) {
	tmp := t.TempDir()

	// header with play id and three channels
	header := "play_id,x,y,s"

	// Create two plays each with 4 timesteps
	rows := []string{
		"A,1,1,0.1",
		"A,2,2,0.2",
		"A,3,3,0.3",
		"A,4,4,0.4",
		"B,5,5,0.5",
		"B,6,6,0.6",
		"B,7,7,0.7",
		"B,8,8,0.8",
	}

	p := filepath.Join(tmp, "fixed.csv")
	writeCSVFile(t, p, header, rows)

	pattern := filepath.Join(tmp, "*.csv")
	seqCols := []string{"x", "y", "s"}

	ds, err := NewAnalyticsDataset(pattern, "play_id", seqCols)
	if err != nil {
		t.Fatalf("NewAnalyticsDataset failed: %v", err)
	}

	if ds.Len() != 2 {
		t.Fatalf("expected 2 plays, got %d", ds.Len())
	}

	indices := []int{0, 1}
	buffers, shapes, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch failed: %v", err)
	}

	// shapes should both be [4,3]
	for i, sh := range shapes {
		if !reflect.DeepEqual(sh, []int{4, 3}) {
			t.Fatalf("shape[%d] unexpected: got %v expected %v", i, sh, []int{4, 3})
		}
	}

	flat, err := MakeAnalyticsBatchFlat(buffers, shapes)
	if err != nil {
		t.Fatalf("MakeAnalyticsBatchFlat failed: %v", err)
	}
	if flat.Batch != 2 || flat.Time != 4 || flat.Channels != 3 {
		t.Fatalf("AnalyticsBatchFlat dims mismatch: %+v", flat)
	}
	if len(flat.Buf) != 2*4*3 {
		t.Fatalf("flat buffer length mismatch: got %d expected %d", len(flat.Buf), 2*4*3)
	}

	// ToGomlxTensor should return a non-nil tensor (we don't deeply inspect tensor internals here)
	if _, err := flat.ToGomlxTensor(); err != nil {
		t.Fatalf("ToGomlxTensor failed: %v", err)
	}
}

// TestAnalyticsDataset_ShuffleDeterministic ensures Shuffle with the same seed
// deterministically produces the same ordering.
func TestAnalyticsDataset_ShuffleDeterministic(t *testing.T) {
	tmp := t.TempDir()

	header := "play_id,x,y"
	rows := []string{
		"P1,1,1",
		"P1,2,2",
		"P2,3,3",
		"P2,4,4",
		"P3,5,5",
		"P3,6,6",
	}

	p := filepath.Join(tmp, "shuffle.csv")
	writeCSVFile(t, p, header, rows)

	pattern := filepath.Join(tmp, "*.csv")
	seqCols := []string{"x", "y"}

	ds, err := NewAnalyticsDataset(pattern, "play_id", seqCols)
	if err != nil {
		t.Fatalf("NewAnalyticsDataset failed: %v", err)
	}

	// Retrieve sequences in original order
	n := ds.Len()
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	beforeBufs, _, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch failed: %v", err)
	}

	// Save original ordering so we can restore it before repeated shuffles.
	orig := append([]string(nil), ds.allPlayIDs...)

	// Shuffle with a fixed seed and capture order
	ds.Shuffle(42)
	afterBufs1, _, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch after shuffle failed: %v", err)
	}

	// Restore original order then shuffle again with same seed to verify determinism.
	ds.allPlayIDs = append([]string(nil), orig...)
	ds.Shuffle(42)
	afterBufs2, _, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch after second shuffle failed: %v", err)
	}

	// Ensure deterministic: afterBufs1 == afterBufs2
	if !reflect.DeepEqual(afterBufs1, afterBufs2) {
		t.Fatalf("shuffle with same seed produced different orderings")
	}

	// It's possible (though unlikely) that the original order equals the shuffled order.
	// We simply ensure that batch lengths remain consistent and buffers are valid.
	if len(beforeBufs) != len(afterBufs1) {
		t.Fatalf("batch length changed after shuffle: before=%d after=%d", len(beforeBufs), len(afterBufs1))
	}
}

// TestAnalyticsDataset_ExampleOutOfRange ensures Example returns an error for invalid indices.
func TestAnalyticsDataset_ExampleOutOfRange(t *testing.T) {
	tmp := t.TempDir()

	header := "play_id,x,y"
	rows := []string{
		"A,1,1",
		"A,2,2",
	}

	p := filepath.Join(tmp, "oob.csv")
	writeCSVFile(t, p, header, rows)

	pattern := filepath.Join(tmp, "*.csv")
	seqCols := []string{"x", "y"}

	ds, err := NewAnalyticsDataset(pattern, "play_id", seqCols)
	if err != nil {
		t.Fatalf("NewAnalyticsDataset failed: %v", err)
	}

	// valid index 0 should work
	if _, _, err := ds.Example(0); err != nil {
		t.Fatalf("Example(0) failed unexpectedly: %v", err)
	}

	// invalid index should error
	if _, _, err := ds.Example(10); err == nil {
		t.Fatalf("expected Example(10) to error, but it succeeded")
	}
}
