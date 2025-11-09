package datasets_test

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Noofbiz/dataBowl/datasets"
	"github.com/Noofbiz/dataBowl/monte"
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

func TestFramePlayersForExample_BasicAndCache(t *testing.T) {
	tmp := t.TempDir()

	// Prepare a single pair of input/output CSVs with three players on the same frame.
	inPath := filepath.Join(tmp, "input1.csv")
	outPath := filepath.Join(tmp, "output1.csv")

	inHeader := "game_id,play_id,nfl_id,frame_id,player_side,player_role,x,y,s,a,o,dir,player_to_predict,num_frames_output"
	outHeader := "game_id,play_id,nfl_id,frame_id,ball_land_x,ball_land_y"

	// Three rows in same game/play/frame
	rowsIn := []string{
		"1,100,10,1,Offense,Targeted Receiver,0,0,5,0,0,0,true,1",    // target player (index 0)
		"1,100,11,1,Offense,Other,10,0,4,0,0,0,false,1",              // teammate
		"1,100,12,1,Defense,Defensive Coverage,-8,0,4,0,0,0,false,1", // defender
	}
	// outputs (must have same number of rows)
	rowsOut := []string{
		"1,100,10,1,100,50",
		"1,100,11,1,100,50",
		"1,100,12,1,100,50",
	}

	writeCSV(t, inPath, inHeader, rowsIn)
	writeCSV(t, outPath, outHeader, rowsOut)

	// Create dataset
	patternIn := filepath.Join(tmp, "input*.csv")
	patternOut := filepath.Join(tmp, "output*.csv")
	ds, err := datasets.NewPredictionDataset(patternIn, patternOut)
	if err != nil {
		t.Fatalf("NewPredictionDataset failed: %v", err)
	}

	// There are 3 examples
	if got := ds.Len(); got != 3 {
		t.Fatalf("expected dataset length 3, got %d", got)
	}

	// Fetch frame players for global index 0
	players, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample returned error: %v", err)
	}
	if len(players) != 3 {
		t.Fatalf("expected 3 players in frame, got %d", len(players))
	}

	// Verify presence of roles and sides
	foundTarget := false
	foundDef := false
	foundTeammate := false
	for _, p := range players {
		if strings.EqualFold(p.Role, "Targeted Receiver") && strings.EqualFold(p.Side, "Offense") {
			foundTarget = true
		}
		if strings.Contains(strings.ToLower(p.Role), "defensive") || strings.EqualFold(p.Side, "Defense") {
			foundDef = true
		}
		if strings.EqualFold(p.Role, "Other") && strings.EqualFold(p.Side, "Offense") {
			foundTeammate = true
		}
	}
	if !foundTarget || !foundDef || !foundTeammate {
		t.Fatalf("unexpected players parsed: target=%v teammate=%v def=%v", foundTarget, foundTeammate, foundDef)
	}

	// Call again to exercise the cache path (should return identical results)
	players2, err := ds.FramePlayersForExample(0)
	if err != nil {
		t.Fatalf("FramePlayersForExample (2nd) returned error: %v", err)
	}
	if len(players2) != len(players) {
		t.Fatalf("cache returned different number of players: before=%d after=%d", len(players), len(players2))
	}
}

func TestMonteInfluence_TeamSignAndRoleWeighting(t *testing.T) {
	tmp := t.TempDir()

	inPath := filepath.Join(tmp, "input.csv")
	outPath := filepath.Join(tmp, "output.csv")

	inHeader := "game_id,play_id,nfl_id,frame_id,player_side,player_role,x,y,s,a,o,dir,player_to_predict,num_frames_output"
	outHeader := "game_id,play_id,nfl_id,frame_id,ball_land_x,ball_land_y"

	// We'll create two variants by copying files and toggling a single player's side/role.

	// base rows with placeholder for the second player's side/role that we'll modify per-case
	baseRows := []string{
		// target player at origin (this is index 0 and will be the initial for Simulate)
		"1,200,100,1,Offense,Targeted Receiver,0,0,5,0,0,0,true,1",
		// other player — we'll replace SIDE and ROLE when writing
		"1,200,101,1,%s,%s,10,0,4,0,0,0,false,1",
	}

	// outputs: both rows have the same landing label (base)
	baseOut := []string{
		"1,200,100,1,120,55",
		"1,200,101,1,120,55",
	}

	// helper to write variant and run simulate returning landing X
	runVariant := func(side, role string) float64 {
		// create input rows with specified side/role for the second player
		rows := []string{
			baseRows[0],
			fmtReplace(baseRows[1], "%s", side, role),
		}
		writeCSV(t, inPath, inHeader, rows)
		writeCSV(t, outPath, outHeader, baseOut)

		ds, err := datasets.NewPredictionDataset(inPath, outPath)
		if err != nil {
			t.Fatalf("NewPredictionDataset failed: %v", err)
		}

		// ensure dataset length == 2
		if ds.Len() != 2 {
			t.Fatalf("expected ds.Len() == 2, got %d", ds.Len())
		}

		// Pull initial features from example 0
		inputs, _, err := ds.Example(0)
		if err != nil {
			t.Fatalf("Example(0) failed: %v", err)
		}

		// Create Monte with K=1 so sampling is deterministic
		m, err := monte.NewMonte(ds, 1)
		if err != nil {
			t.Fatalf("NewMonte failed: %v", err)
		}

		// Run a single simulation (one sim, two steps)
		results, err := m.Simulate(0, inputs, 1, 2)
		if err != nil {
			t.Fatalf("Simulate failed: %v", err)
		}
		if len(results) != 1 {
			t.Fatalf("expected 1 result, got %d", len(results))
		}
		return float64(results[0].LandX)
	}

	// Run offensive variant: other player is Offense -> should attract and nudge landing to the right (increase X)
	landOff := runVariant("Offense", "Other")
	// Run defensive variant: other player is Defense -> should repel and nudge landing to the left (decrease X)
	landDef := runVariant("Defense", "Defensive Coverage")

	baseX := 120.0 // base landing X from outputs

	if !(landOff > baseX) {
		t.Fatalf("expected offensive player to nudge landing right: base=%v got=%v", baseX, landOff)
	}
	if !(landDef < baseX) {
		t.Fatalf("expected defensive player to nudge landing left: base=%v got=%v", baseX, landDef)
	}

	// Now verify role weighting: Offense + Other vs Offense + Targeted Receiver produce different magnitudes
	landOffOther := runVariant("Offense", "Other")
	landOffTargeted := runVariant("Offense", "Targeted Receiver")

	// compute absolute nudges
	nudgeOther := math.Abs(landOffOther - baseX)
	nudgeTargeted := math.Abs(landOffTargeted - baseX)

	if !(nudgeTargeted > nudgeOther) {
		t.Fatalf("expected targeted receiver to have larger influence than other: other=%v targeted=%v", nudgeOther, nudgeTargeted)
	}
}

// fmtReplace is a tiny helper to replace two %s in a template string with the provided values.
func fmtReplace(template string, placeholders ...string) string {
	out := template
	for _, v := range placeholders {
		out = strings.Replace(out, "%s", v, 1)
	}
	return out
}
