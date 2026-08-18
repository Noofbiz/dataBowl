package datasets

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// TestRow holds the submission ID and pre-built input features for one
// prediction target from the test set.
type TestRow struct {
	ID     string    // value from the "id" column in the template CSV
	Inputs []float32 // length InputFeatureLen
}

// testPlayerSnap is the last-seen tracking snapshot for a player in a play.
type testPlayerSnap struct {
	x, y, s, a, dir, o float32
	numFrames           int
}

// testPlayMeta holds per-play metadata needed to build input features.
type testPlayMeta struct {
	direction string
	yardline  float32
}

// testGameData holds all parsed data for one game's test input.
type testGameData struct {
	// lastFrameID[playID][nflID] = highest frame_id seen for that player.
	lastFrameID map[string]map[string]int
	// snaps[playID][nflID] = tracking snapshot at lastFrameID.
	snaps map[string]map[string]testPlayerSnap
	// plays[playID] = play-level metadata.
	plays map[string]testPlayMeta
}

func newTestGameData() *testGameData {
	return &testGameData{
		lastFrameID: make(map[string]map[string]int),
		snaps:       make(map[string]map[string]testPlayerSnap),
		plays:       make(map[string]testPlayMeta),
	}
}

// LoadTestData reads test_input.csv and the submission template (test.csv) and
// returns one TestRow per template row, with input features built using the
// same logic as the training dataset:
//
//	[last_x, last_y, last_s, last_a, last_dir, last_o,
//	 play_direction_enc, absolute_yardline_number, frame_id/num_frames_output]
//
// Rows whose game/play/player cannot be resolved are silently skipped.
func LoadTestData(testInputPath, templatePath string) ([]TestRow, error) {
	games := make(map[string]*testGameData)

	if err := parseTestInputCSV(testInputPath, games); err != nil {
		return nil, fmt.Errorf("parse test input: %w", err)
	}

	// Read the template and build TestRows.
	tf, err := os.Open(templatePath)
	if err != nil {
		return nil, fmt.Errorf("open template %s: %w", templatePath, err)
	}
	defer tf.Close()

	r := csv.NewReader(tf)
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read template header: %w", err)
	}

	col := func(name string) int {
		for i, h := range header {
			if strings.EqualFold(strings.TrimSpace(h), name) {
				return i
			}
		}
		return -1
	}
	iID      := col("id")
	iGameID  := col("game_id")
	iPlayID  := col("play_id")
	iNFLID   := col("nfl_id")
	iFrameID := col("frame_id")
	for _, ci := range []int{iID, iGameID, iPlayID, iNFLID, iFrameID} {
		if ci < 0 {
			return nil, fmt.Errorf("template %s is missing a required column", templatePath)
		}
	}

	encDir := func(s string) float32 {
		if strings.EqualFold(s, "right") {
			return 1
		}
		return 0
	}

	var rows []TestRow
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		id          := strings.TrimSpace(rec[iID])
		gameID      := strings.TrimSpace(rec[iGameID])
		playID      := strings.TrimSpace(rec[iPlayID])
		nflID       := strings.TrimSpace(rec[iNFLID])
		outFrameID, _ := strconv.Atoi(strings.TrimSpace(rec[iFrameID]))

		gd, ok := games[gameID]
		if !ok {
			continue
		}
		snapsForPlay, ok := gd.snaps[playID]
		if !ok {
			continue
		}
		snap, ok := snapsForPlay[nflID]
		if !ok {
			continue
		}
		meta, ok := gd.plays[playID]
		if !ok {
			continue
		}
		if snap.numFrames == 0 {
			continue
		}

		rows = append(rows, TestRow{
			ID: id,
			Inputs: []float32{
				snap.x, snap.y, snap.s, snap.a, snap.dir, snap.o,
				encDir(meta.direction),
				meta.yardline,
				float32(outFrameID) / float32(snap.numFrames),
			},
		})
	}
	return rows, nil
}

// parseTestInputCSV reads testInputPath and builds a per-game snapshot map.
// For each (gameID, playID, nflID) triple it keeps only the row with the
// highest frame_id, matching the training feature-building logic.
func parseTestInputCSV(path string, games map[string]*testGameData) error {
	fh, err := os.Open(path)
	if err != nil {
		return err
	}
	defer fh.Close()

	r := csv.NewReader(fh)
	header, err := r.Read()
	if err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	col := func(name string) int {
		for i, h := range header {
			if strings.EqualFold(strings.TrimSpace(h), name) {
				return i
			}
		}
		return -1
	}

	iGameID    := col("game_id")
	iPlayID    := col("play_id")
	iNFLID     := col("nfl_id")
	iFrameID   := col("frame_id")
	iPlayDir   := col("play_direction")
	iYardline  := col("absolute_yardline_number")
	iX         := col("x")
	iY         := col("y")
	iS         := col("s")
	iA         := col("a")
	iDir       := col("dir")
	iO         := col("o")
	iNumFrames := col("num_frames_output")

	for _, ci := range []int{iGameID, iPlayID, iNFLID, iFrameID, iPlayDir, iYardline, iX, iY, iS, iA, iDir, iO, iNumFrames} {
		if ci < 0 {
			return fmt.Errorf("missing required column in %s", path)
		}
	}

	f32 := func(s string) float32 {
		v, _ := strconv.ParseFloat(strings.TrimSpace(s), 32)
		return float32(v)
	}
	atoi := func(s string) int {
		v, _ := strconv.Atoi(strings.TrimSpace(s))
		return v
	}
	str := func(i int, rec []string) string {
		if i < 0 || i >= len(rec) {
			return ""
		}
		return strings.TrimSpace(rec[i])
	}

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if len(rec) <= iNumFrames {
			continue
		}

		gameID  := str(iGameID, rec)
		playID  := str(iPlayID, rec)
		nflID   := str(iNFLID, rec)
		frameID := atoi(rec[iFrameID])

		gd, ok := games[gameID]
		if !ok {
			gd = newTestGameData()
			games[gameID] = gd
		}

		if gd.lastFrameID[playID] == nil {
			gd.lastFrameID[playID] = make(map[string]int)
			gd.snaps[playID] = make(map[string]testPlayerSnap)
		}

		if frameID > gd.lastFrameID[playID][nflID] {
			gd.lastFrameID[playID][nflID] = frameID
			gd.snaps[playID][nflID] = testPlayerSnap{
				x:         f32(rec[iX]),
				y:         f32(rec[iY]),
				s:         f32(rec[iS]),
				a:         f32(rec[iA]),
				dir:       f32(rec[iDir]),
				o:         f32(rec[iO]),
				numFrames: atoi(rec[iNumFrames]),
			}
		}

		if _, exists := gd.plays[playID]; !exists {
			gd.plays[playID] = testPlayMeta{
				direction: str(iPlayDir, rec),
				yardline:  f32(rec[iYardline]),
			}
		}
	}
	return nil
}
