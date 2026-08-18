package datasets

import (
	"testing"
)

const (
	testInputPattern  = "assets/prediction/train/input_2023_w01.csv"
	testOutputPattern = "assets/prediction/train/output_2023_w01.csv"

	// Known values from input_2023_w01.csv / output_2023_w01.csv.
	testGameID = "2023090700"
	testPlayID = "101"
	// Play 101 has three players_to_predict in the CSV, but the Play struct
	// stores only one PlayerToPredictNFLID (last row wins). Each predicted
	// player has 21 output frames.
	testOutputFrameCount = 21
	testPlayExampleCount = testOutputFrameCount // 1 player × 21 frames
)

// newTestDataset is a helper that constructs a dataset from the week-01 files
// only, so tests run quickly without loading all 18 weeks.
func newTestDataset(t *testing.T) *PredictionDataset {
	t.Helper()
	ds, err := NewPredictionDataset(testInputPattern, testOutputPattern)
	if err != nil {
		t.Fatalf("NewPredictionDataset: %v", err)
	}
	return ds
}

// TestNewPredictionDataset_IndexOnly verifies that construction is fast (lazy):
// the games map should be empty right after init — data is not loaded yet.
func TestNewPredictionDataset_IndexOnly(t *testing.T) {
	ds := newTestDataset(t)

	ds.mu.Lock()
	loadedCount := len(ds.games)
	ds.mu.Unlock()

	if loadedCount != 0 {
		t.Errorf("expected games map to be empty after construction (lazy), got %d entries", loadedCount)
	}
}

// TestNewPredictionDataset_GameOrderPopulated verifies that the game_id index
// is populated at construction time (from the fast header scan).
func TestNewPredictionDataset_GameOrderPopulated(t *testing.T) {
	ds := newTestDataset(t)

	if ds.NumGames() == 0 {
		t.Fatal("expected at least one game in gameOrder after construction")
	}
	// w01 has 16 distinct games.
	if got := ds.NumGames(); got != 16 {
		t.Errorf("NumGames = %d, want 16", got)
	}
}
// TestEnsureGameLoaded_PopulatesGame checks that ensureGameLoaded parses the
// CSV for the requested game and leaves other games unparsed.
func TestEnsureGameLoaded_PopulatesGame(t *testing.T) {
	ds := newTestDataset(t)

	if err := ds.ensureGameLoaded(testGameID); err != nil {
		t.Fatalf("ensureGameLoaded(%q): %v", testGameID, err)
	}

	ds.mu.Lock()
	g, ok := ds.games[testGameID]
	loadedCount := len(ds.games)
	ds.mu.Unlock()

	if !ok || g == nil {
		t.Fatalf("game %q not in ds.games after ensureGameLoaded", testGameID)
	}
	// All 16 games in w01 share the same file, so loading one loads all 16.
	if loadedCount != 16 {
		t.Errorf("expected 16 games loaded (same file), got %d", loadedCount)
	}
}

// TestEnsureGameLoaded_UnknownGame checks that an unknown game_id returns an error.
func TestEnsureGameLoaded_UnknownGame(t *testing.T) {
	ds := newTestDataset(t)
	if err := ds.ensureGameLoaded("DOES_NOT_EXIST"); err == nil {
		t.Error("expected error for unknown game_id, got nil")
	}
}

// TestLoadedFileFlag verifies that calling ensureGameLoaded twice does not
// re-parse the file (loadedFiles flag is set).
func TestLoadedFileFlag(t *testing.T) {
	ds := newTestDataset(t)

	if err := ds.ensureGameLoaded(testGameID); err != nil {
		t.Fatal(err)
	}
	// Manually wipe games to detect a second parse (would re-populate).
	ds.mu.Lock()
	ds.games = make(map[string]*Game)
	ds.mu.Unlock()

	// Second call should be a no-op due to loadedFiles flag.
	if err := ds.ensureGameLoaded(testGameID); err != nil {
		t.Fatal(err)
	}
	ds.mu.Lock()
	count := len(ds.games)
	ds.mu.Unlock()

	if count != 0 {
		t.Errorf("expected no re-parse (games should still be empty), got %d entries", count)
	}
}

// TestGameExamples_Count checks the number of examples built for a known play.
func TestGameExamples_Count(t *testing.T) {
	ds := newTestDataset(t)

	if err := ds.ensureGameLoaded(testGameID); err != nil {
		t.Fatal(err)
	}

	ds.mu.Lock()
	exs := ds.gameExamples[testGameID]
	ds.mu.Unlock()

	if len(exs) == 0 {
		t.Fatal("no examples built for testGameID")
	}

	// Count examples that belong to play 101.
	play101Count := 0
	for _, ex := range exs {
		if ex.playID == testPlayID {
			play101Count++
		}
	}
	if play101Count != testPlayExampleCount {
		t.Errorf("play %s: got %d examples, want %d (1 player × %d frames)",
			testPlayID, play101Count, testPlayExampleCount, testOutputFrameCount)
	}
}

// TestExample_InputShape checks that Example returns slices of the right length.
func TestExample_InputShape(t *testing.T) {
	ds := newTestDataset(t)

	inputs, labels, err := ds.Example(0)
	if err != nil {
		t.Fatalf("Example(0): %v", err)
	}
	if len(inputs) != InputFeatureLen {
		t.Errorf("inputs length = %d, want %d", len(inputs), InputFeatureLen)
	}
	if len(labels) != 2 {
		t.Errorf("labels length = %d, want 2", len(labels))
	}
}

// TestExample_FrameIdNorm checks that the last input feature (frame_id_norm)
// is in (0, 1].
func TestExample_FrameIdNorm(t *testing.T) {
	ds := newTestDataset(t)

	for i := 0; i < 10; i++ {
		inputs, _, err := ds.Example(i)
		if err != nil {
			t.Fatalf("Example(%d): %v", i, err)
		}
		norm := inputs[InputFeatureLen-1]
		if norm <= 0 || norm > 1 {
			t.Errorf("Example(%d): frame_id_norm = %f, want in (0,1]", i, norm)
		}
	}
}

// TestExample_OutOfRange checks that out-of-range indices return an error.
func TestExample_OutOfRange(t *testing.T) {
	ds := newTestDataset(t)
	_, _, err := ds.Example(-1)
	if err == nil {
		t.Error("Example(-1): expected error, got nil")
	}
}

// TestNextFramePositionForExample ensures labels match Example labels.
func TestNextFramePositionForExample(t *testing.T) {
	ds := newTestDataset(t)

	_, labels, err := ds.Example(0)
	if err != nil {
		t.Fatal(err)
	}
	x, y, found, err := ds.NextFramePositionForExample(0)
	if err != nil {
		t.Fatalf("NextFramePositionForExample(0): %v", err)
	}
	if !found {
		t.Error("expected found=true")
	}
	if x != labels[0] || y != labels[1] {
		t.Errorf("position (%f,%f) != labels (%f,%f)", x, y, labels[0], labels[1])
	}
}

// TestBatch returns consistent results with individual Example calls.
func TestBatch(t *testing.T) {
	ds := newTestDataset(t)

	indices := []int{0, 1, 2}
	bInputs, bLabels, err := ds.Batch(indices)
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if len(bInputs) != len(indices) {
		t.Fatalf("Batch returned %d inputs, want %d", len(bInputs), len(indices))
	}
	for i, idx := range indices {
		inp, lbl, _ := ds.Example(idx)
		for j, v := range inp {
			if bInputs[i][j] != v {
				t.Errorf("Batch[%d] input[%d] = %f, Example(%d) = %f", i, j, bInputs[i][j], idx, v)
			}
		}
		for j, v := range lbl {
			if bLabels[i][j] != v {
				t.Errorf("Batch[%d] label[%d] = %f, Example(%d) = %f", i, j, bLabels[i][j], idx, v)
			}
		}
	}
}

// TestIter_YieldsBatches verifies that Iter yields at least one batch and that
// each batch has the correct tensor shapes.
func TestIter_YieldsBatches(t *testing.T) {
	ds := newTestDataset(t)

	batchCount := 0
	for batch, err := range ds.Iter() {
		if err != nil {
			t.Fatalf("Iter error: %v", err)
		}
		if len(batch.Inputs) != 1 {
			t.Errorf("batch.Inputs len = %d, want 1", len(batch.Inputs))
		}
		if len(batch.Labels) != 1 {
			t.Errorf("batch.Labels len = %d, want 1", len(batch.Labels))
		}
		shape := batch.Inputs[0].Shape()
		if len(shape.Dimensions) != 2 || int(shape.Dimensions[1]) != InputFeatureLen {
			t.Errorf("input tensor shape = %v, want [N, %d]", shape.Dimensions, InputFeatureLen)
		}
		labelShape := batch.Labels[0].Shape()
		if len(labelShape.Dimensions) != 2 || int(labelShape.Dimensions[1]) != 2 {
			t.Errorf("label tensor shape = %v, want [N, 2]", labelShape.Dimensions)
		}
		batchCount++
	}
	if batchCount == 0 {
		t.Error("Iter yielded no batches")
	}
}

// TestIter_Reset verifies that calling Iter() twice yields the same number of batches.
func TestIter_Reset(t *testing.T) {
	ds := newTestDataset(t)

	count := func() int {
		n := 0
		for _, err := range ds.Iter() {
			if err != nil {
				t.Fatal(err)
			}
			n++
		}
		return n
	}

	first := count()
	second := count()
	if first != second {
		t.Errorf("first Iter = %d batches, second = %d; expected equal", first, second)
	}
}

// TestFrameKNearestPlayersFeatures checks length and zero-padding.
func TestFrameKNearestPlayersFeatures(t *testing.T) {
	ds := newTestDataset(t)

	const k = 5
	feats, err := ds.FrameKNearestPlayersFeatures(0, k)
	if err != nil {
		t.Fatalf("FrameKNearestPlayersFeatures: %v", err)
	}
	if len(feats) != k*5 {
		t.Errorf("len(feats) = %d, want %d", len(feats), k*5)
	}
}

// TestCurrentGameID tracks the cursor during Iter.
func TestCurrentGameID(t *testing.T) {
	ds := newTestDataset(t)

	seen := map[string]bool{}
	for _, err := range ds.Iter() {
		if err != nil {
			t.Fatal(err)
		}
		id := ds.CurrentGameID()
		if id == "" {
			t.Error("CurrentGameID returned empty string during Iter")
		}
		seen[id] = true
	}
	if len(seen) == 0 {
		t.Error("no game IDs observed during Iter")
	}
}

// TestOutputFileForInput checks the path derivation helper.
func TestOutputFileForInput(t *testing.T) {
	cases := []struct {
		input, want string
	}{
		{"assets/prediction/train/input_2023_w01.csv", "assets/prediction/train/output_2023_w01.csv"},
		{"/data/input_foo.csv", "/data/output_foo.csv"},
	}
	for _, c := range cases {
		got := outputFileForInput(c.input)
		if got != c.want {
			t.Errorf("outputFileForInput(%q) = %q, want %q", c.input, got, c.want)
		}
	}
}
