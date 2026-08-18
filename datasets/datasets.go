package datasets

import (
	"encoding/csv"
	"fmt"
	"io"
	"iter"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
)

// InputFeatureLen is the number of features per prediction example:
//
//	[x, y, s, a, dir, o, play_direction_enc, absolute_yardline_number, frame_id_norm]
//
// where frame_id_norm = output_frame_id / num_frames_output ∈ (0, 1].
const InputFeatureLen = 9

// outputKey uniquely identifies one future-frame position in the output CSVs.
type outputKey struct {
	gameID  string
	playID  string
	nflID   string
	frameID int
}

// predExample is one (game, play, nfl_id, output_frame_id) prediction target.
type predExample struct {
	gameID  string
	playID  string
	nflID   string
	frameID int // output frame_id ∈ [1, num_frames_output]

	inputs []float32 // length InputFeatureLen
	labels []float32 // [x, y] from output CSV
}

// PredictionDataset implements train.Dataset for the NFL Big Data Bowl prediction
// challenge.
//
// # Data model
//
// CSVs are parsed lazily, one file pair at a time, into Game/Play/Frame/Player
// structs. At construction only the file paths are indexed and the list of
// game IDs is built by scanning the first column of each input file. Data for
// a game is loaded from disk the first time it is needed and then cached.
//
// # Iter behaviour
//
// Each call to Iter implicitly resets the cursor to 0. One train.Batch is
// yielded per game. The batch tensors are 2-D:
//
//	Inputs[0]  shape [N, InputFeatureLen=9]
//	Labels[0]  shape [N, 2]
//
// where N is the number of prediction targets in that game (one per
// player_to_predict × output frame).
//
// Each input row encodes the player's last pre-throw state plus where in the
// future trajectory the label belongs:
//
//	[last_x, last_y, last_s, last_a, last_dir, last_o,
//	 play_direction_enc, absolute_yardline_number, frame_id/num_frames_output]
//
// Each label row is the ground-truth [x, y] from the output CSV.
type PredictionDataset struct {
	trainInputPattern  string
	trainOutputPattern string
	name               string

	// gameOrder is the sorted slice of game IDs; the cursor indexes into it.
	gameOrder []string

	// currentGameIdx is updated atomically as Iter() advances.
	currentGameIdx atomic.Int64

	// gameToInputFile maps each game_id to the input CSV that contains it.
	gameToInputFile map[string]string

	// loadedFiles tracks which input files have already been parsed.
	loadedFiles map[string]bool

	// games holds parsed input data indexed by game_id (populated lazily).
	games map[string]*Game

	// gameExamples maps game_id → its flattened prediction examples (lazily populated).
	gameExamples map[string][]predExample

	// outputPositions holds ground-truth future positions from the output CSVs
	// (populated lazily alongside their paired input file).
	outputPositions map[outputKey][2]float32

	frameIndexTTL        time.Duration
	frameIndexMaxEntries int

	mu           sync.Mutex
	cacheEnabled bool
}

// NewPredictionDataset prepares a lazily-loading dataset from CSVs matching
// inputPattern and outputPattern (both glob patterns). Empty strings fall back
// to the default asset paths.
//
// Construction only scans the first column of each input file to build a
// game_id → file index; actual CSV rows are parsed on first access.
func NewPredictionDataset(inputPattern, outputPattern string) (*PredictionDataset, error) {
	if inputPattern == "" {
		inputPattern = "datasets/assets/prediction/train/input*.csv"
	}
	if outputPattern == "" {
		outputPattern = "datasets/assets/prediction/train/output*.csv"
	}

	ds := &PredictionDataset{
		trainInputPattern:    inputPattern,
		trainOutputPattern:   outputPattern,
		name:                 "Prediction Dataset",
		frameIndexTTL:        5 * time.Minute,
		frameIndexMaxEntries: 2000,
		games:                make(map[string]*Game),
		gameExamples:         make(map[string][]predExample),
		outputPositions:      make(map[outputKey][2]float32),
		gameToInputFile:      make(map[string]string),
		loadedFiles:          make(map[string]bool),
	}

	inputFiles, err := filepath.Glob(inputPattern)
	if err != nil {
		return nil, fmt.Errorf("glob input pattern %q: %w", inputPattern, err)
	}
	if len(inputFiles) == 0 {
		return nil, fmt.Errorf("no files matched input pattern %q", inputPattern)
	}

	// Index: scan only the game_id column of each input file.
	for _, f := range inputFiles {
		if err := ds.indexInputFile(f); err != nil {
			return nil, fmt.Errorf("index %s: %w", f, err)
		}
	}

	ds.buildGameOrder()
	return ds, nil
}

// indexInputFile scans the game_id column of path (without parsing all fields)
// and records that each game_id found lives in this file.
func (ds *PredictionDataset) indexInputFile(path string) error {
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

	gameIDCol := -1
	for i, h := range header {
		if strings.EqualFold(strings.TrimSpace(h), "game_id") {
			gameIDCol = i
			break
		}
	}
	if gameIDCol < 0 {
		return fmt.Errorf("no game_id column in %s", path)
	}

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if gameIDCol < len(rec) {
			gid := strings.TrimSpace(rec[gameIDCol])
			if gid != "" {
				ds.gameToInputFile[gid] = path
			}
		}
	}
	return nil
}

// outputFileForInput derives the paired output CSV path from an input path.
// e.g. ".../input_2023_w01.csv" → ".../output_2023_w01.csv"
func outputFileForInput(inputPath string) string {
	dir := filepath.Dir(inputPath)
	base := filepath.Base(inputPath)
	outBase := strings.Replace(base, "input", "output", 1)
	return filepath.Join(dir, outBase)
}

// ensureGameLoaded loads the file pair for gameID if not already loaded.
// Callers must NOT hold ds.mu.
func (ds *PredictionDataset) ensureGameLoaded(gameID string) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	inputFile, ok := ds.gameToInputFile[gameID]
	if !ok {
		return fmt.Errorf("unknown game_id %q", gameID)
	}
	if ds.loadedFiles[inputFile] {
		return nil // already loaded
	}

	if err := ds.loadInputCSV(inputFile); err != nil {
		return fmt.Errorf("load input %s: %w", inputFile, err)
	}

	outFile := outputFileForInput(inputFile)
	if err := ds.loadOutputCSV(outFile); err != nil {
		// Missing output file is non-fatal (e.g. test-only input)
		_ = err
	}

	ds.buildExamplesForFile(inputFile)
	ds.loadedFiles[inputFile] = true
	return nil
}

// loadInputCSV parses one input CSV file into Game/Play/Frame/Player structs.
//
// Every row populates:
//   - Game (keyed by game_id) — ID only; metadata fields like HomeTeam are not
//     in the prediction input CSV.
//   - Play (keyed by play_id within Game.Plays) — PlayDirection,
//     AbsoluteYardlineNumber, BallLandX/Y, PlayerToPredictNFLID.
//   - Frame (keyed by frame_id within Play.Frames) — ID only.
//   - Player (keyed by nfl_id within Frame.Players) — all tracking fields.
func (ds *PredictionDataset) loadInputCSV(path string) error {
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

	// Required columns
	iGameID := col("game_id")
	iPlayID := col("play_id")
	iPredict := col("player_to_predict")
	iNFLID := col("nfl_id")
	iFrameID := col("frame_id")
	iPlayDir := col("play_direction")
	iYardline := col("absolute_yardline_number")
	iBallLandX := col("ball_land_x")
	iBallLandY := col("ball_land_y")
	iName := col("player_name")
	iHeight := col("player_height")
	iWeight := col("player_weight")
	iBirth := col("player_birth_date")
	iPosition := col("player_position")
	iSide := col("player_side")
	iRole := col("player_role")
	iX := col("x")
	iY := col("y")
	iS := col("s")
	iA := col("a")
	iDir := col("dir")
	iO := col("o")
	iNumFrames := col("num_frames_output")

	for _, ci := range []int{iGameID, iPlayID, iPredict, iNFLID, iFrameID,
		iPlayDir, iYardline, iBallLandX, iBallLandY,
		iX, iY, iS, iA, iDir, iO, iNumFrames} {
		if ci < 0 {
			return fmt.Errorf("missing required column in %s", path)
		}
	}

	f64 := func(s string) float64 {
		v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
		return v
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

		gameID := str(iGameID, rec)
		playID := str(iPlayID, rec)
		nflID := str(iNFLID, rec)
		frameID := str(iFrameID, rec)
		predict := strings.EqualFold(str(iPredict, rec), "true")

		// --- Game ---
		game, ok := ds.games[gameID]
		if !ok {
			game = &Game{ID: gameID, Plays: make(map[string]Play)}
			ds.games[gameID] = game
		}

		// --- Play ---
		play, ok := game.Plays[playID]
		if !ok {
			play = Play{
				ID:                     playID,
				Frames:                 make(map[string]Frame),
				PlayDirection:          str(iPlayDir, rec),
				AbsoluteYardlineNumber: f64(rec[iYardline]),
				BallLandX:              f64(rec[iBallLandX]),
				BallLandY:              f64(rec[iBallLandY]),
			}
		}
		if predict {
			play.PlayerToPredictNFLID = nflID
			play.PlayerToPredict = true
		}

		// --- Frame ---
		frame, ok := play.Frames[frameID]
		if !ok {
			frame = Frame{ID: frameID, Players: make(map[string]Player)}
		}

		// --- Player ---
		frame.Players[nflID] = Player{
			ID:               nflID,
			DisplayName:      str(iName, rec),
			Height:           str(iHeight, rec),
			Weight:           f64(rec[iWeight]),
			BirthDate:        str(iBirth, rec),
			TypicalRole:      str(iPosition, rec),
			PlayerSide:       str(iSide, rec),
			PlayerRole:       str(iRole, rec),
			PlayRole:         str(iRole, rec),
			PlayerToPredict:  predict,
			X:                f64(rec[iX]),
			Y:                f64(rec[iY]),
			Velocity:         f64(rec[iS]),
			Acceleration:     f64(rec[iA]),
			AngleOfMomentum:  f64(rec[iDir]),
			Orientation:      f64(rec[iO]),
			OutputFrameCount: uint(atoi(rec[iNumFrames])),
		}

		play.Frames[frameID] = frame
		game.Plays[playID] = play
	}
	return nil
}

// loadOutputCSV reads one output CSV into ds.outputPositions.
//
// Expected columns: game_id, play_id, nfl_id, frame_id, x, y
func (ds *PredictionDataset) loadOutputCSV(path string) error {
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

	iGameID := col("game_id")
	iPlayID := col("play_id")
	iNFLID := col("nfl_id")
	iFrameID := col("frame_id")
	iX := col("x")
	iY := col("y")

	for _, ci := range []int{iGameID, iPlayID, iNFLID, iFrameID, iX, iY} {
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

	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if len(rec) <= iY {
			continue
		}
		k := outputKey{
			gameID:  strings.TrimSpace(rec[iGameID]),
			playID:  strings.TrimSpace(rec[iPlayID]),
			nflID:   strings.TrimSpace(rec[iNFLID]),
			frameID: atoi(rec[iFrameID]),
		}
		ds.outputPositions[k] = [2]float32{f32(rec[iX]), f32(rec[iY])}
	}
	return nil
}

// buildGameOrder sorts and stores game IDs for deterministic iteration.
func (ds *PredictionDataset) buildGameOrder() {
	ds.gameOrder = make([]string, 0, len(ds.gameToInputFile))
	for id := range ds.gameToInputFile {
		ds.gameOrder = append(ds.gameOrder, id)
	}
	sort.Strings(ds.gameOrder)
}

// buildExamplesForFile constructs predExample slices for every game that was
// just loaded from inputFile by joining the parsed Game structs against the
// output-CSV positions already stored in ds.outputPositions.
func (ds *PredictionDataset) buildExamplesForFile(inputFile string) {
	encDir := func(s string) float32 {
		if strings.EqualFold(s, "right") {
			return 1
		}
		return 0
	}

	for gameID, file := range ds.gameToInputFile {
		if file != inputFile {
			continue
		}
		if _, already := ds.gameExamples[gameID]; already {
			continue
		}
		game, ok := ds.games[gameID]
		if !ok {
			continue
		}
		var gameExs []predExample

		for playID, play := range game.Plays {
			if !play.PlayerToPredict {
				continue
			}
			nflID := play.PlayerToPredictNFLID

			lastFrameID := -1
			var lastPlayer Player
			for fid, frame := range play.Frames {
				fnum, _ := strconv.Atoi(fid)
				if p, ok := frame.Players[nflID]; ok && fnum > lastFrameID {
					lastFrameID = fnum
					lastPlayer = p
				}
			}
			if lastFrameID < 0 || lastPlayer.OutputFrameCount == 0 {
				continue
			}

			playDirEnc := encDir(play.PlayDirection)
			yardline := float32(play.AbsoluteYardlineNumber)
			nFrames := int(lastPlayer.OutputFrameCount)

			for outFrame := 1; outFrame <= nFrames; outFrame++ {
				k := outputKey{gameID: gameID, playID: playID, nflID: nflID, frameID: outFrame}
				pos, ok := ds.outputPositions[k]
				if !ok {
					continue
				}

				gameExs = append(gameExs, predExample{
					gameID:  gameID,
					playID:  playID,
					nflID:   nflID,
					frameID: outFrame,
					inputs: []float32{
						float32(lastPlayer.X),
						float32(lastPlayer.Y),
						float32(lastPlayer.Velocity),
						float32(lastPlayer.Acceleration),
						float32(lastPlayer.AngleOfMomentum),
						float32(lastPlayer.Orientation),
						playDirEnc,
						yardline,
						float32(outFrame) / float32(nFrames),
					},
					labels: []float32{pos[0], pos[1]},
				})
			}
		}

		ds.gameExamples[gameID] = gameExs
	}
}

// Name implements train.Dataset.
func (ds *PredictionDataset) Name() string { return ds.name }

// NumGames returns the number of distinct games in the dataset.
func (ds *PredictionDataset) NumGames() int { return len(ds.gameOrder) }

// Len returns the total number of games in the dataset.
// It's for an older API, iterate via Iter() for lazy access.
func (ds *PredictionDataset) Len() int {
	return len(ds.gameOrder)
}

// CurrentGameID returns the game_id that Iter() will yield next (or is
// currently yielding). Returns "" before the first call to Iter().
func (ds *PredictionDataset) CurrentGameID() string {
	idx := ds.currentGameIdx.Load()
	if idx < 0 || idx >= int64(ds.Len()) {
		return ""
	}
	return ds.gameOrder[idx]
}

// CurrentGame returns the Game struct for the game Iter() will yield next.
// Returns nil if the cursor is out of range.
func (ds *PredictionDataset) CurrentGame() *Game {
	id := ds.CurrentGameID()
	if id == "" {
		return nil
	}
	_ = ds.ensureGameLoaded(id)
	ds.mu.Lock()
	g := ds.games[id]
	ds.mu.Unlock()
	return g
}

// Iter implements train.Dataset using iter.Seq2.
//
// One train.Batch is yielded per game, loading each file pair on first access.
// The batch tensors are 2-D so the model receives the full game at once:
//
//	Inputs[0]  float32  shape [N, InputFeatureLen]
//	Labels[0]  float32  shape [N, 2]
//
// N is the number of (player_to_predict, output_frame) pairs in the game.
// Each call to Iter resets the cursor to the first game.
func (ds *PredictionDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		ds.currentGameIdx.Store(0)

		for i, gameID := range ds.gameOrder {
			ds.currentGameIdx.Store(int64(i))

			if err := ds.ensureGameLoaded(gameID); err != nil {
				if !yield(train.Batch{}, err) {
					return
				}
				continue
			}

			ds.mu.Lock()
			exs := ds.gameExamples[gameID]
			ds.mu.Unlock()

			if len(exs) == 0 {
				continue
			}

			n := len(exs)
			inputFlat := make([]float32, n*InputFeatureLen)
			labelFlat := make([]float32, n*2)

			for j := range exs {
				copy(inputFlat[j*InputFeatureLen:], exs[j].inputs)
				copy(labelFlat[j*2:], exs[j].labels)
			}

			inputTensor := tensors.FromFlatDataAndDimensions(inputFlat, n, InputFeatureLen)
			labelTensor := tensors.FromFlatDataAndDimensions(labelFlat, n, 2)

			batch := train.Batch{
				Inputs: []*tensors.Tensor{inputTensor},
				Labels: []*tensors.Tensor{labelTensor},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

// globalExample resolves a global example index to a gameID and local index.
func (ds *PredictionDataset) globalExample(i int) (ex predExample, err error) {
	if i < 0 {
		return ex, fmt.Errorf("example index %d out of range", i)
	}
	offset := 0
	for _, gameID := range ds.gameOrder {
		if err2 := ds.ensureGameLoaded(gameID); err2 != nil {
			return ex, err2
		}
		ds.mu.Lock()
		exs := ds.gameExamples[gameID]
		ds.mu.Unlock()
		if i < offset+len(exs) {
			return exs[i-offset], nil
		}
		offset += len(exs)
	}
	return ex, fmt.Errorf("example index %d out of range", i)
}

// Example returns the input features and labels for the i-th global example.
func (ds *PredictionDataset) Example(i int) (inputs []float32, labels []float32, err error) {
	ex, err := ds.globalExample(i)
	if err != nil {
		return nil, nil, err
	}
	return ex.inputs, ex.labels, nil
}

// Batch returns inputs and labels for a slice of global example indices.
func (ds *PredictionDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	inputs := make([][]float32, len(indices))
	labels := make([][]float32, len(indices))
	for bi, idx := range indices {
		inp, lbl, err := ds.Example(idx)
		if err != nil {
			return nil, nil, err
		}
		inputs[bi] = inp
		labels[bi] = lbl
	}
	return inputs, labels, nil
}

// FrameKNearestPlayersFeatures returns flattened features for the K players
// nearest (by Euclidean distance) to example i in its last pre-throw frame,
// excluding the example's own player. Each neighbour contributes 5 floats:
// [x, y, s, a, dir]. Missing neighbours are zero-padded.
func (ds *PredictionDataset) FrameKNearestPlayersFeatures(i, k int) ([]float32, error) {
	ex, err := ds.globalExample(i)
	if err != nil {
		return nil, err
	}

	ds.mu.Lock()
	game, ok := ds.games[ex.gameID]
	ds.mu.Unlock()
	if !ok {
		return make([]float32, k*5), nil
	}
	play, ok := game.Plays[ex.playID]
	if !ok {
		return make([]float32, k*5), nil
	}

	// Find the last frame the player_to_predict appears in.
	lastFrameID := -1
	for fid, frame := range play.Frames {
		fnum, _ := strconv.Atoi(fid)
		if _, has := frame.Players[ex.nflID]; has && fnum > lastFrameID {
			lastFrameID = fnum
		}
	}
	if lastFrameID < 0 {
		return make([]float32, k*5), nil
	}
	frame := play.Frames[strconv.Itoa(lastFrameID)]

	ox := ex.inputs[0]
	oy := ex.inputs[1]

	type candidate struct {
		dist float32
		p    Player
	}
	candidates := make([]candidate, 0, len(frame.Players))
	for nflID, p := range frame.Players {
		if nflID == ex.nflID {
			continue
		}
		dx := float32(p.X) - ox
		dy := float32(p.Y) - oy
		d := float32(math.Sqrt(float64(dx*dx + dy*dy)))
		candidates = append(candidates, candidate{dist: d, p: p})
	}
	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].dist < candidates[b].dist
	})

	out := make([]float32, k*5)
	for j := 0; j < k && j < len(candidates); j++ {
		p := candidates[j].p
		base := j * 5
		out[base+0] = float32(p.X)
		out[base+1] = float32(p.Y)
		out[base+2] = float32(p.Velocity)
		out[base+3] = float32(p.Acceleration)
		out[base+4] = float32(p.AngleOfMomentum)
	}
	return out, nil
}

// NextFramePositionForExample returns the ground-truth (x, y) for example i
// directly from its labels (the output-CSV position for that output frame).
func (ds *PredictionDataset) NextFramePositionForExample(i int) (x, y float32, found bool, err error) {
	ex, err := ds.globalExample(i)
	if err != nil {
		return 0, 0, false, err
	}
	return ex.labels[0], ex.labels[1], true, nil
}

// SetFrameIndexTTL configures the TTL for frame-index cache entries.
func (ds *PredictionDataset) SetFrameIndexTTL(d time.Duration) {
	ds.frameIndexTTL = d
}

// SetFrameIndexMaxEntries configures the maximum number of frame-index LRU entries.
func (ds *PredictionDataset) SetFrameIndexMaxEntries(n int) {
	ds.frameIndexMaxEntries = n
}

// EnableCache is a no-op: loaded data is already held in memory after each file is parsed.
func (ds *PredictionDataset) EnableCache() error {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.cacheEnabled = true
	return nil
}
