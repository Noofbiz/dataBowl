package datasets

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

// Test that TTL expiry forces a re-read of underlying CSV data.
func TestFramePlayersForExample_TTLExpiry(t *testing.T) {
	tmp := t.TempDir()

	inPath := filepath.Join(tmp, "in.csv")
	outPath := filepath.Join(tmp, "out.csv")

	inHeader := "game_id,play_id,nfl_id,frame_id,player_side,player_role,x,y,s,a,o,dir,player_to_predict,num_frames_output"
	outHeader := "game_id,play_id,nfl_id,frame_id,ball_land_x,ball_land_y"

	// initial row: x=1
	rowsIn1 := []string{
		"1,1,10,1,Offense,Other,1,0,0,0,0,0,true,1",
	}
	rowsOut := []string{
		"1,1,10,1,100,50",
	}

	writeCSV(t, inPath, inHeader, rowsIn1)
	writeCSV(t, outPath, outHeader, rowsOut)

	ds, err := NewPredictionDataset(inPath, outPath)
	if err != nil {
		t.Fatalf("NewPredictionDataset failed: %v", err)
	}

	// Set a short TTL so cache expires quickly
	ds.frameIndexTTL = 150 * time.Millisecond

	players1, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample initial failed: %v", err)
	}
	if len(players1) != 1 {
		t.Fatalf("expected 1 player initially, got %d", len(players1))
	}
	if players1[0].X != 1 {
		t.Fatalf("unexpected initial X: got %v want %v", players1[0].X, 1.0)
	}

	// Modify underlying CSV to change x to 42
	rowsIn2 := []string{
		"1,1,10,1,Offense,Other,42,0,0,0,0,0,true,1",
	}
	writeCSV(t, inPath, inHeader, rowsIn2)

	// Immediately calling again should still return cached value (not expired yet)
	playersCached, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample cached call failed: %v", err)
	}
	if playersCached[0].X != 1 {
		t.Fatalf("expected cached X=1 before TTL expiry, got %v", playersCached[0].X)
	}

	// Wait for TTL to expire
	time.Sleep(200 * time.Millisecond)

	// Now call again; cache should be expired and we should see new value
	playersAfter, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample after TTL failed: %v", err)
	}
	if len(playersAfter) != 1 {
		t.Fatalf("expected 1 player after TTL, got %d", len(playersAfter))
	}
	if playersAfter[0].X != 42 {
		t.Fatalf("expected X to reflect updated CSV (42) after TTL expiry, got %v", playersAfter[0].X)
	}
}

// Test that the LRU eviction removes the least-recently-used entry when capacity is exceeded.
func TestFramePlayersForExample_EvictionLRU(t *testing.T) {
	tmp := t.TempDir()

	inPattern := filepath.Join(tmp, "in_*.csv")
	outPattern := filepath.Join(tmp, "out_*.csv")

	inHeader := "game_id,play_id,nfl_id,frame_id,player_side,player_role,x,y,s,a,o,dir,player_to_predict,num_frames_output"
	outHeader := "game_id,play_id,nfl_id,frame_id,ball_land_x,ball_land_y"

	// create 4 distinct pair files, each with one row, so global indices will be 0..3
	for i := 0; i < 4; i++ {
		inPath := filepath.Join(tmp, fmt.Sprintf("in_%c.csv", 'A'+i))
		outPath := filepath.Join(tmp, fmt.Sprintf("out_%c.csv", 'A'+i))
		// Use game_id/play_id/frame_id equal to i+1 to make keys unique
		gid := i + 1
		// x coordinate encodes i to identify reads
		inRow := []string{formatRow(gid, gid, 100+i, gid, "Offense", "Other", float64(i), 0)}
		outRow := []string{formatOutRow(gid, gid, 100+i, gid, 100.0, 50.0)}
		// writeCSV in this package expects (t, path, header, rows)
		writeCSV(t, inPath, inHeader, inRow)
		writeCSV(t, outPath, outHeader, outRow)
	}

	ds, err := NewPredictionDataset(inPattern, outPattern)
	if err != nil {
		t.Fatalf("NewPredictionDataset failed: %v", err)
	}

	// Small capacity so eviction will occur
	ds.frameIndexMaxEntries = 3
	// Make TTL large so expiry does not interfere
	ds.frameIndexTTL = 5 * time.Minute

	// Access indices 0,1,2 -> inserted in that order, LRU back should be 0
	for i := 0; i < 3; i++ {
		if _, err := ds.FramePlayersForExample(i); err != nil {
			t.Fatalf("FramePlayersForExample(%d) failed: %v", i, err)
		}
	}

	// Modify the underlying file for index 0 to change x -> 99, so if it's evicted, we'll observe change
	in0 := filepath.Join(tmp, "in_A.csv")
	out0 := filepath.Join(tmp, "out_A.csv")
	writeCSV(t, in0, inHeader, []string{formatRow(1, 1, 100, 1, "Offense", "Other", 99.0, 0)})
	writeCSV(t, out0, outHeader, []string{formatOutRow(1, 1, 100, 1, 100.0, 50.0)})

	// Now access index 3 to trigger eviction (capacity 3 -> adding 4th entry)
	if _, err := ds.FramePlayersForExample(3); err != nil {
		t.Fatalf("FramePlayersForExample(3) failed: %v", err)
	}

	// Now access index 0. If it was evicted it will be re-read and reflect x=99.
	players0, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample(0) after eviction failed: %v", err)
	}
	if len(players0) != 1 {
		t.Fatalf("expected 1 player for index 0, got %d", len(players0))
	}
	if players0[0].X != 99 {
		t.Fatalf("expected re-read x=99 for index 0 after eviction, got %v", players0[0].X)
	}
}

// formatRow builds a CSV row for input files.
func formatRow(gameID, playID, nfl int, frame int, side, role string, x, y float64) string {
	return format("%d,%d,%d,%d,%s,%s,%.3f,%.3f,0,0,0,0,true,1", gameID, playID, nfl, frame, side, role, x, y)
}

// formatOutRow builds a CSV row for output files.
func formatOutRow(gameID, playID, nfl int, frame int, lx, ly float64) string {
	return format("%d,%d,%d,%d,%.3f,%.3f", gameID, playID, nfl, frame, lx, ly)
}

// tiny wrapper to avoid importing fmt repeatedly in the test body.
func format(formatStr string, args ...interface{}) string {
	return sprintf(formatStr, args...)
}

// Go's fmt.Sprintf is used but redeclare via a private name to avoid extra imports in tests.
func sprintf(formatStr string, args ...interface{}) string {
	return fmt.Sprintf(formatStr, args...)
}
