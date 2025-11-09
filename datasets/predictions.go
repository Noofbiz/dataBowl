package datasets

import (
	"container/list"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// PredictionDataset represents paired input/output CSV files.
//
// Each input CSV is expected to contain features: x,y,s,a,o,dir
// Each output CSV is expected to contain labels: ball_land_x,ball_land_y
//
// The dataset pairs input/output files by sorting the matched globs and
// matching positions. It supports an optional in-memory cache where each
// cached record is normalized to exactly 8 columns:
// [x,y,s,a,o,dir,ball_land_x,ball_land_y]
type PredictionDataset struct {
	// Patterns used to find CSV files
	InputPattern  string
	OutputPattern string

	// BatchSize for yielding batches (when used as gomlx Dataset)
	BatchSize int

	// Paired paths
	inputPaths  []string
	outputPaths []string

	// Canonical column index mapping for the normalized cached record
	// { "x":0, "y":1, ..., "ball_land_y":7 }
	colIndex map[string]int

	// RNG for potential shuffling
	rand *rand.Rand

	// rowCounts per pair index (number of examples in each input file)
	rowCounts map[int]int

	// cumCounts cumulative prefix sums for mapping global indices
	cumCounts []int

	// total number of examples across all input files
	totalExamples int

	// Protects metadata (rowCounts, cumCounts, totalExamples)
	metaMu sync.RWMutex

	// per-file RW locks for safe concurrent reads
	inputLocks  []sync.RWMutex
	outputLocks []sync.RWMutex

	// Optional cache: map pairIdx -> []normalizedRecords
	cacheEnabled bool
	cacheMu      sync.RWMutex
	cache        map[int][][]string

	// In-memory index for frame players (on-demand LRU cache).
	// Key format: "<game_id>|<play_id>|<frame_id>"
	frameIndexMu         sync.RWMutex
	frameIndex           map[string]*list.Element
	frameIndexList       *list.List // most-recent at Front
	frameIndexMaxEntries int
	frameIndexTTL        time.Duration
}

// cacheEntry represents an entry stored in the frameIndexList LRU list.
type cacheEntry struct {
	key     string
	players []FramePlayer
	created time.Time
}

// FramePlayer represents a single player observed in a particular frame.
// Fields include identifiers and the minimal information Monte needs (position,
// side/team and role). Types are kept simple (strings for IDs) to avoid heavy
// parsing at this stage; numerical fields are parsed where needed.
type FramePlayer struct {
	GameID string  // original CSV game_id as string
	PlayID string  // original CSV play_id as string
	NFLID  string  // nfl_id (string form)
	X      float32 // x coordinate
	Y      float32 // y coordinate
	Side   string  // player_side e.g., "Offense" or "Defense"
	Role   string  // player_role e.g., "Targeted Receiver", "Passer", etc.
}

// FramePlayersForExample returns all players (as FramePlayer) that were present
// in the same game_id / play_id / frame_id as the example identified by the
// provided global index. This allows Monte to examine all players in a frame
// and compute attractive/repulsive forces based on side/role.
//
// The function is defensive: it first maps the global index to a (pairIdx, localIdx)
// using the existing index mapping, then reads the input CSV for that pair to
// discover the three identifying columns (game_id, play_id, frame_id) for the
// target row. It then scans the same CSV and collects all rows that share the
// same triple. Parsing for x/y uses the shared parseFloat32 helper (from util.go).
func (d *PredictionDataset) FramePlayersForExample(globalIdx int) ([]FramePlayer, error) {
	if globalIdx < 0 {
		return nil, fmt.Errorf("global index %d out of range", globalIdx)
	}
	pairIdx, localIdx := d.mapGlobalIndex(globalIdx)
	if pairIdx < 0 || pairIdx >= len(d.inputPaths) {
		return nil, fmt.Errorf("pair index %d out of range", pairIdx)
	}
	inPath := d.inputPaths[pairIdx]

	// Open input CSV and read header to build a name->pos map and find the target identifiers.
	inF, err := os.Open(inPath)
	if err != nil {
		return nil, fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer inF.Close()
	inR := csv.NewReader(inF)
	header, err := inR.Read()
	if err != nil {
		return nil, fmt.Errorf("read input header %s: %w", inPath, err)
	}
	hmap := make(map[string]int, len(header))
	for i, h := range header {
		hmap[strings.TrimSpace(strings.ToLower(h))] = i
	}

	// Seek to the target localIdx row to obtain identifiers for game_id/play_id/frame_id.
	var targetRow []string
	for i := 0; i <= localIdx; i++ {
		targetRow, err = inR.Read()
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("input %s: unexpected EOF while seeking to row %d", inPath, localIdx)
			}
			return nil, fmt.Errorf("read input row %d: %w", localIdx, err)
		}
	}

	getField := func(m map[string]int, rec []string, name string) string {
		if pos, ok := m[name]; ok && pos >= 0 && pos < len(rec) {
			return strings.TrimSpace(rec[pos])
		}
		return ""
	}

	gameID := getField(hmap, targetRow, "game_id")
	playID := getField(hmap, targetRow, "play_id")
	frameID := getField(hmap, targetRow, "frame_id")

	if gameID == "" || playID == "" || frameID == "" {
		return nil, fmt.Errorf("could not determine frame identity for %s row %d (game_id/play_id/frame_id missing)", inPath, localIdx)
	}

	// Build cache key
	key := gameID + "|" + playID + "|" + frameID

	// Check frame index LRU cache (fast path)
	now := time.Now()
	d.frameIndexMu.RLock()
	if elem, ok := d.frameIndex[key]; ok {
		ce := elem.Value.(*cacheEntry)
		// Check TTL based on creation time
		if now.Sub(ce.created) <= d.frameIndexTTL {
			// promote to front (most-recent) and return copy
			d.frameIndexMu.RUnlock()
			d.frameIndexMu.Lock()
			d.frameIndexList.MoveToFront(elem)
			d.frameIndexMu.Unlock()
			out := make([]FramePlayer, len(ce.players))
			copy(out, ce.players)
			return out, nil
		}
	}
	d.frameIndexMu.RUnlock()

	// Cache miss or expired entry: build players list by scanning all input files
	players := make([]FramePlayer, 0, 16)

	for _, pth := range d.inputPaths {
		f, ferr := os.Open(pth)
		if ferr != nil {
			// skip unreadable files
			continue
		}
		r := csv.NewReader(f)
		hdr, herr := r.Read()
		if herr != nil {
			f.Close()
			continue
		}
		mp := make(map[string]int, len(hdr))
		for i, h := range hdr {
			mp[strings.TrimSpace(strings.ToLower(h))] = i
		}
		// quick positions (may be -1 if not present)
		gidPos, _ := mp["game_id"]
		pidPos, _ := mp["play_id"]
		fidPos, _ := mp["frame_id"]
		nflPos, _ := mp["nfl_id"]
		xPos, _ := mp["x"]
		yPos, _ := mp["y"]
		sidePos, _ := mp["player_side"]
		rolePos, _ := mp["player_role"]

		for {
			rec, rerr := r.Read()
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				// read error, stop scanning this file
				break
			}
			// ensure positions exist in this record
			if gidPos < 0 || pidPos < 0 || fidPos < 0 ||
				gidPos >= len(rec) || pidPos >= len(rec) || fidPos >= len(rec) {
				continue
			}
			if strings.TrimSpace(rec[gidPos]) != gameID ||
				strings.TrimSpace(rec[pidPos]) != playID ||
				strings.TrimSpace(rec[fidPos]) != frameID {
				continue
			}
			// matched frame: extract player info
			nflid := ""
			if nflPos >= 0 && nflPos < len(rec) {
				nflid = strings.TrimSpace(rec[nflPos])
			}
			var px, py float32
			if xPos >= 0 && xPos < len(rec) {
				if v, err := parseFloat32(rec[xPos]); err == nil {
					px = v
				}
			}
			if yPos >= 0 && yPos < len(rec) {
				if v, err := parseFloat32(rec[yPos]); err == nil {
					py = v
				}
			}
			side := ""
			if sidePos >= 0 && sidePos < len(rec) {
				side = strings.TrimSpace(rec[sidePos])
			}
			role := ""
			if rolePos >= 0 && rolePos < len(rec) {
				role = strings.TrimSpace(rec[rolePos])
			}
			fp := FramePlayer{
				GameID: gameID,
				PlayID: playID,
				NFLID:  nflid,
				X:      px,
				Y:      py,
				Side:   side,
				Role:   role,
			}
			players = append(players, fp)
		}
		f.Close()
	}

	// Store in LRU cache with eviction if necessary.
	d.frameIndexMu.Lock()
	if elem, ok := d.frameIndex[key]; ok {
		// update existing element and move to front
		d.frameIndexList.MoveToFront(elem)
		ce := elem.Value.(*cacheEntry)
		ce.players = players
		ce.created = now
		d.frameIndexMu.Unlock()
	} else {
		// create new element at front
		ce := &cacheEntry{key: key, players: players, created: now}
		elem := d.frameIndexList.PushFront(ce)
		d.frameIndex[key] = elem
		// Evict least-recently-used if over capacity
		if d.frameIndexMaxEntries > 0 && d.frameIndexList.Len() > d.frameIndexMaxEntries {
			back := d.frameIndexList.Back()
			if back != nil {
				be := back.Value.(*cacheEntry)
				delete(d.frameIndex, be.key)
				d.frameIndexList.Remove(back)
			}
		}
		d.frameIndexMu.Unlock()
	}

	// return a copy to avoid callers mutating internal cache
	out := make([]FramePlayer, len(players))
	copy(out, players)
	return out, nil
}

// ExampleMeta contains identifying metadata for a single global example.
type ExampleMeta struct {
	GameID   string // game identifier
	PlayID   string // play identifier
	FrameID  string // frame identifier (string as present in CSV)
	NFLID    string // nfl_id of the player (may be empty)
	PairIdx  int    // which input/output pair file this example came from
	LocalIdx int    // row index within the pair file
}

// ExampleMeta returns identifying metadata for the example at global index.
// It maps the global index to the pair/local index, reads the input CSV header,
// seeks to the local row and extracts the fields: game_id, play_id, frame_id, nfl_id.
// This helper is useful for upstream code that needs to locate the same player
// across frames or find neighboring rows for the same object.
func (d *PredictionDataset) ExampleMeta(globalIdx int) (*ExampleMeta, error) {
	if globalIdx < 0 {
		return nil, fmt.Errorf("global index %d out of range", globalIdx)
	}
	pairIdx, localIdx := d.mapGlobalIndex(globalIdx)
	if pairIdx < 0 || pairIdx >= len(d.inputPaths) {
		return nil, fmt.Errorf("pair index %d out of range", pairIdx)
	}
	inPath := d.inputPaths[pairIdx]

	f, err := os.Open(inPath)
	if err != nil {
		return nil, fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read input header %s: %w", inPath, err)
	}
	hmap := make(map[string]int, len(header))
	for i, h := range header {
		hmap[strings.TrimSpace(strings.ToLower(h))] = i
	}

	// seek to the target local row
	var rec []string
	for i := 0; i <= localIdx; i++ {
		rec, err = r.Read()
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("input %s: unexpected EOF while seeking to row %d", inPath, localIdx)
			}
			return nil, fmt.Errorf("read input row %d: %w", localIdx, err)
		}
	}

	getField := func(name string) string {
		if pos, ok := hmap[name]; ok && pos >= 0 && pos < len(rec) {
			return strings.TrimSpace(rec[pos])
		}
		return ""
	}

	gameID := getField("game_id")
	playID := getField("play_id")
	frameID := getField("frame_id")
	nflid := getField("nfl_id")

	if gameID == "" || playID == "" || frameID == "" {
		return nil, fmt.Errorf("could not determine frame identity for %s row %d (game_id/play_id/frame_id missing)", inPath, localIdx)
	}

	return &ExampleMeta{
		GameID:   gameID,
		PlayID:   playID,
		FrameID:  frameID,
		NFLID:    nflid,
		PairIdx:  pairIdx,
		LocalIdx: localIdx,
	}, nil
}

// NextFramePositionForExample attempts to find the position (x,y) of the same
// object (identified by nfl_id) in the subsequent frame (frame_id + 1) within
// the same input pair CSV. It returns (x,y,true,nil) if found, (0,0,false,nil)
// if no matching next-frame row exists, or an error if an IO/parse problem occurs.
func (d *PredictionDataset) NextFramePositionForExample(globalIdx int) (float32, float32, bool, error) {
	meta, err := d.ExampleMeta(globalIdx)
	if err != nil {
		return 0, 0, false, err
	}

	// parse numeric frame id (if non-numeric, we cannot reliably compute the next)
	fidStr := strings.TrimSpace(meta.FrameID)
	fid, err := strconv.Atoi(fidStr)
	if err != nil {
		return 0, 0, false, fmt.Errorf("frame_id not numeric for example %d: %s", globalIdx, meta.FrameID)
	}
	nextF := fmt.Sprintf("%d", fid+1)

	// Open the same input CSV and scan for a row with same game_id, play_id,
	// same nfl_id and frame_id == nextF.
	inPath := d.inputPaths[meta.PairIdx]
	f, err := os.Open(inPath)
	if err != nil {
		return 0, 0, false, fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	hdr, err := r.Read()
	if err != nil {
		return 0, 0, false, fmt.Errorf("read header %s: %w", inPath, err)
	}
	mp := make(map[string]int, len(hdr))
	for i, h := range hdr {
		mp[strings.TrimSpace(strings.ToLower(h))] = i
	}

	gidPos, _ := mp["game_id"]
	pidPos, _ := mp["play_id"]
	fidPos, _ := mp["frame_id"]
	nflPos, _ := mp["nfl_id"]
	xPos, _ := mp["x"]
	yPos, _ := mp["y"]

	for {
		rec, rerr := r.Read()
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			// stop scanning on read error
			break
		}
		// ensure necessary positions exist
		if gidPos < 0 || pidPos < 0 || fidPos < 0 || gidPos >= len(rec) || pidPos >= len(rec) || fidPos >= len(rec) {
			continue
		}
		if strings.TrimSpace(rec[gidPos]) != meta.GameID ||
			strings.TrimSpace(rec[pidPos]) != meta.PlayID ||
			strings.TrimSpace(rec[fidPos]) != nextF {
			continue
		}
		// If nfl_id is available, require it to match the meta's nfl id.
		if meta.NFLID != "" {
			if nflPos < 0 || nflPos >= len(rec) || strings.TrimSpace(rec[nflPos]) != meta.NFLID {
				continue
			}
		}
		// parse x,y if present
		var px, py float32
		if xPos >= 0 && xPos < len(rec) {
			if v, perr := parseFloat32(rec[xPos]); perr == nil {
				px = v
			}
		}
		if yPos >= 0 && yPos < len(rec) {
			if v, perr := parseFloat32(rec[yPos]); perr == nil {
				py = v
			}
		}
		return px, py, true, nil
	}

	// not found
	return 0, 0, false, nil
}

// IsPlayerToPredict reports whether the input CSV row corresponding to the
// given global example index has the `player_to_predict` flag set (truthy).
// It returns (true,nil) if the column exists and the cell is "1" or "true"
// (case-insensitive), (false,nil) if the column exists and the cell is empty/0/false,
// and an error if the row cannot be read or the input file cannot be opened.
func (d *PredictionDataset) IsPlayerToPredict(globalIdx int) (bool, error) {
	if globalIdx < 0 {
		return false, fmt.Errorf("global index %d out of range", globalIdx)
	}
	pairIdx, localIdx := d.mapGlobalIndex(globalIdx)
	if pairIdx < 0 || pairIdx >= len(d.inputPaths) {
		return false, fmt.Errorf("pair index %d out of range", pairIdx)
	}
	inPath := d.inputPaths[pairIdx]

	f, err := os.Open(inPath)
	if err != nil {
		return false, fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	hdr, err := r.Read()
	if err != nil {
		return false, fmt.Errorf("read header %s: %w", inPath, err)
	}
	hmap := make(map[string]int, len(hdr))
	for i, h := range hdr {
		hmap[strings.TrimSpace(strings.ToLower(h))] = i
	}

	// Seek to the target row
	var rec []string
	for i := 0; i <= localIdx; i++ {
		rec, err = r.Read()
		if err != nil {
			if err == io.EOF {
				return false, fmt.Errorf("input %s: unexpected EOF while seeking to row %d", inPath, localIdx)
			}
			return false, fmt.Errorf("read input row %d: %w", localIdx, err)
		}
	}

	// Determine column position (support common variants)
	pos, ok := hmap["player_to_predict"]
	if !ok {
		// try alternative forms
		if p, ok2 := hmap["player_to_predict?"]; ok2 {
			pos = p
			ok = true
		} else if p, ok2 := hmap["player_to_predict_flag"]; ok2 {
			pos = p
			ok = true
		}
	}
	if !ok || pos < 0 || pos >= len(rec) {
		// column absent -> treat as not flagged
		return false, nil
	}

	val := strings.TrimSpace(strings.ToLower(rec[pos]))
	if val == "" || val == "0" || val == "false" {
		return false, nil
	}
	if val == "1" || val == "true" {
		return true, nil
	}
	// fallback: non-empty string counts as true
	return true, nil
}

// FrameKNearestPlayersFeatures returns a flattened feature vector for up to k nearest
// other players in the same frame as the example at globalIdx. For each neighbor
// (ordered by increasing distance) the function emits five float32 features:
//
//	[ rel_x, rel_y, dist, side_sign, role_weight ]
//
// where:
// - rel_x, rel_y = neighbor_pos - target_pos
// - dist = euclidean distance to target
// - side_sign = +1.0 for offense, -1.0 for defense
// - role_weight = 2.0 for targeted/receiver roles, 1.1 for passer-like, else 1.0
//
// If fewer than k neighbors are present, remaining slots are zeros. The returned
// slice length is k*5. This helper is intentionally simple and robust: it uses the
// dataset's FramePlayersForExample to obtain frame players and the example inputs
// (preferably cached) to determine the target's current (x,y) position.
func (d *PredictionDataset) FrameKNearestPlayersFeatures(globalIdx, k int) ([]float32, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be > 0")
	}

	// obtain all players in the same frame
	players, err := d.FramePlayersForExample(globalIdx)
	if err != nil {
		return nil, err
	}
	// empty frame -> return zeroed vector
	if len(players) == 0 {
		return make([]float32, k*5), nil
	}

	// Determine target position from the example's inputs when possible
	// (prefer the dataset's Example reader which respects caching).
	var tx, ty float32
	inp, _, err := d.Example(globalIdx)
	if err == nil && len(inp) >= 2 {
		tx = inp[0]
		ty = inp[1]
	} else {
		// fallback: use first player's pos if Example unavailable
		tx = players[0].X
		ty = players[0].Y
	}

	// Build neighbor list with distances
	type nb struct {
		idx  int
		dist float64
	}
	nbs := make([]nb, 0, len(players))
	for i, p := range players {
		dx := float64(p.X - tx)
		dy := float64(p.Y - ty)
		dist := math.Hypot(dx, dy)
		// skip zero-distance self if it exactly matches target position (keeps self out)
		nbs = append(nbs, nb{idx: i, dist: dist})
	}

	// sort by increasing distance
	sort.Slice(nbs, func(i, j int) bool { return nbs[i].dist < nbs[j].dist })

	out := make([]float32, k*5)
	// fill up to k neighbors (closest first)
	for i := 0; i < k && i < len(nbs); i++ {
		p := players[nbs[i].idx]
		relx := p.X - tx
		rely := p.Y - ty
		dist := float32(nbs[i].dist)

		// side sign
		sign := float32(1.0)
		if strings.EqualFold(p.Side, "Defense") || strings.EqualFold(p.Side, "D") {
			sign = -1.0
		}

		// role weight heuristic
		roleMul := float32(1.0)
		lrole := strings.ToLower(strings.TrimSpace(p.Role))
		if strings.Contains(lrole, "target") || strings.Contains(lrole, "receiver") {
			roleMul = 2.0
		} else if strings.Contains(lrole, "pass") {
			roleMul = 1.1
		}

		base := i * 5
		out[base+0] = relx
		out[base+1] = rely
		out[base+2] = dist
		out[base+3] = sign
		out[base+4] = roleMul
	}

	return out, nil
}

// NewPredictionDataset creates a dataset by pairing input and output CSVs.
// If outputPattern is empty, inputPattern will be used for both inputs and outputs
// (backwards compatible mode where a single CSV contains both features and labels).
func NewPredictionDataset(inputPattern, outputPattern string) (*PredictionDataset, error) {
	if outputPattern == "" {
		outputPattern = inputPattern
	}

	inPaths, err := filepath.Glob(inputPattern)
	if err != nil {
		return nil, fmt.Errorf("failed to glob input pattern %s: %w", inputPattern, err)
	}
	if len(inPaths) == 0 {
		return nil, fmt.Errorf("no input CSV files found matching pattern: %s", inputPattern)
	}

	outPaths, err := filepath.Glob(outputPattern)
	if err != nil {
		return nil, fmt.Errorf("failed to glob output pattern %s: %w", outputPattern, err)
	}
	if len(outPaths) == 0 {
		return nil, fmt.Errorf("no output CSV files found matching pattern: %s", outputPattern)
	}

	// Sort deterministically
	sort.Strings(inPaths)
	sort.Strings(outPaths)

	// If counts match, pair by index. If not, attempt basename normalization match.
	pairedIn := make([]string, 0, len(inPaths))
	pairedOut := make([]string, 0, len(inPaths))

	if len(inPaths) == len(outPaths) {
		pairedIn = append(pairedIn, inPaths...)
		pairedOut = append(pairedOut, outPaths...)
	} else {
		// Build normalized map of outputs and try to match inputs by normalized basename
		normalize := func(p string) string {
			b := strings.ToLower(strings.TrimSuffix(filepath.Base(p), filepath.Ext(p)))
			// remove common tokens
			b = strings.ReplaceAll(b, "input", "")
			b = strings.ReplaceAll(b, "output", "")
			b = strings.ReplaceAll(b, "_", "")
			b = strings.ReplaceAll(b, "-", "")
			b = strings.TrimSpace(b)
			return b
		}
		outMap := make(map[string]string, len(outPaths))
		for _, p := range outPaths {
			outMap[normalize(p)] = p
		}
		for _, ip := range inPaths {
			k := normalize(ip)
			if op, ok := outMap[k]; ok {
				pairedIn = append(pairedIn, ip)
				pairedOut = append(pairedOut, op)
			}
		}
		// If pairing failed or partial, fall back to error to avoid silent mismatches.
		if len(pairedIn) == 0 || len(pairedIn) != len(inPaths) || len(pairedIn) != len(outPaths) {
			return nil, fmt.Errorf("could not deterministically pair input and output CSVs (inputs=%d outputs=%d matched=%d). Provide explicit matching patterns that correspond one-to-one",
				len(inPaths), len(outPaths), len(pairedIn))
		}
	}

	// Validate required columns exist in each paired input/output file now so
	// that callers get immediate errors when CSVs are malformed. This mirrors
	// the checks performed later during caching/reads but ensures the constructor
	// fails early for tests and CI.
	reqIn := []string{"x", "y", "s", "a", "o", "dir"}

	for i := range pairedIn {
		inPath := pairedIn[i]
		outPath := pairedOut[i]

		// Open input and read header
		inF, err := os.Open(inPath)
		if err != nil {
			return nil, fmt.Errorf("open input %s: %w", inPath, err)
		}
		inR := csv.NewReader(inF)
		inHeader, err := inR.Read()
		inF.Close()
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("input %s: empty CSV or missing header", inPath)
			}
			return nil, fmt.Errorf("read input header %s: %w", inPath, err)
		}
		inMap := make(map[string]struct{}, len(inHeader))
		for _, h := range inHeader {
			inMap[strings.TrimSpace(strings.ToLower(h))] = struct{}{}
		}
		for _, c := range reqIn {
			if _, ok := inMap[c]; !ok {
				return nil, fmt.Errorf("input %s missing required column %s", inPath, c)
			}
		}

		// Open output and read header
		outF, err := os.Open(outPath)
		if err != nil {
			return nil, fmt.Errorf("open output %s: %w", outPath, err)
		}
		outR := csv.NewReader(outF)
		outHeader, err := outR.Read()
		outF.Close()
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("output %s: empty CSV or missing header", outPath)
			}
			return nil, fmt.Errorf("read output header %s: %w", outPath, err)
		}
		outMap := make(map[string]struct{}, len(outHeader))
		for _, h := range outHeader {
			outMap[strings.TrimSpace(strings.ToLower(h))] = struct{}{}
		}
		// Accept either explicit label columns (ball_land_x/ball_land_y) or legacy x/y names.
		_, hasBallX := outMap["ball_land_x"]
		_, hasBallY := outMap["ball_land_y"]
		_, hasX := outMap["x"]
		_, hasY := outMap["y"]
		if !(hasBallX && hasBallY) && !(hasX && hasY) {
			return nil, fmt.Errorf("output %s missing required columns: need either (ball_land_x,ball_land_y) or (x,y)", outPath)
		}
	}

	ds := &PredictionDataset{
		InputPattern:  inputPattern,
		OutputPattern: outputPattern,
		BatchSize:     32,
		inputPaths:    pairedIn,
		outputPaths:   pairedOut,
		colIndex: map[string]int{
			"x":           0,
			"y":           1,
			"s":           2,
			"a":           3,
			"o":           4,
			"dir":         5,
			"ball_land_x": 6,
			"ball_land_y": 7,
		},
		rand:         rand.New(rand.NewSource(time.Now().UnixNano())),
		rowCounts:    make(map[int]int),
		cache:        make(map[int][][]string),
		cacheEnabled: false,
	}
	ds.inputLocks = make([]sync.RWMutex, len(ds.inputPaths))
	ds.outputLocks = make([]sync.RWMutex, len(ds.outputPaths))

	// Initialize on-demand LRU frame index cache (kept small by default).
	ds.frameIndex = make(map[string]*list.Element)
	ds.frameIndexList = list.New()
	ds.frameIndexMaxEntries = 2000
	ds.frameIndexTTL = 5 * time.Minute

	// build index (counts)
	if err := ds.buildIndex(); err != nil {
		return nil, err
	}

	return ds, nil
}

// SetFrameIndexTTL sets the TTL for the in-memory frame index cache.
func (d *PredictionDataset) SetFrameIndexTTL(ttl time.Duration) {
	d.frameIndexMu.Lock()
	d.frameIndexTTL = ttl
	d.frameIndexMu.Unlock()
}

// SetFrameIndexMaxEntries sets the maximum number of entries for the
// in-memory frame index LRU cache. If the current cache exceeds the new
// capacity, oldest entries will be evicted synchronously.
func (d *PredictionDataset) SetFrameIndexMaxEntries(n int) {
	d.frameIndexMu.Lock()
	d.frameIndexMaxEntries = n
	// Evict least-recently-used entries if we're over capacity.
	if d.frameIndexList != nil && d.frameIndexMaxEntries > 0 {
		for d.frameIndexList.Len() > d.frameIndexMaxEntries {
			back := d.frameIndexList.Back()
			if back == nil {
				break
			}
			be := back.Value.(*cacheEntry)
			delete(d.frameIndex, be.key)
			d.frameIndexList.Remove(back)
		}
	}
	d.frameIndexMu.Unlock()
}

// countCSVRows is provided by datasets/util.go to avoid duplication across files.

// buildIndex counts rows for input files (and validates outputs) and builds cumCounts.
func (d *PredictionDataset) buildIndex() error {
	localCum := make([]int, len(d.inputPaths)+1)
	localCum[0] = 0
	localRowCounts := make(map[int]int, len(d.inputPaths))

	for i, inPath := range d.inputPaths {
		cnt, err := countCSVRows(inPath)
		if err != nil {
			return fmt.Errorf("failed to count rows in input %s: %w", inPath, err)
		}

		localRowCounts[i] = cnt
		localCum[i+1] = localCum[i] + cnt
	}

	d.metaMu.Lock()
	d.cumCounts = localCum
	d.rowCounts = localRowCounts
	d.totalExamples = localCum[len(d.inputPaths)]
	d.metaMu.Unlock()
	return nil
}

// EnableCache loads all paired CSVs into memory and stores normalized records.
// Each normalized record is 8 columns: x,y,s,a,o,dir,ball_land_x,ball_land_y.
func (d *PredictionDataset) EnableCache() error {
	// quick path
	d.cacheMu.RLock()
	if d.cacheEnabled {
		d.cacheMu.RUnlock()
		return nil
	}
	d.cacheMu.RUnlock()

	// prepare cache map
	d.cacheMu.Lock()
	if d.cache == nil {
		d.cache = make(map[int][][]string, len(d.inputPaths))
	}
	d.cacheMu.Unlock()

	numFiles := len(d.inputPaths)
	if numFiles == 0 {
		d.cacheMu.Lock()
		d.cacheEnabled = true
		d.cacheMu.Unlock()
		return nil
	}

	jobs := make(chan int, numFiles)
	results := make(chan struct {
		idx int
		rec [][]string
		err error
		dur time.Duration
	}, numFiles)

	var abort int32
	workerCount := runtime.NumCPU()
	if workerCount > numFiles {
		workerCount = numFiles
	}
	var wg sync.WaitGroup
	wg.Add(workerCount)

	start := time.Now()

	for w := 0; w < workerCount; w++ {
		go func() {
			defer wg.Done()
			for idx := range jobs {
				if atomic.LoadInt32(&abort) != 0 {
					return
				}
				inPath := d.inputPaths[idx]
				outPath := d.outputPaths[idx]
				t0 := time.Now()
				inF, err := os.Open(inPath)
				if err != nil {
					if atomic.CompareAndSwapInt32(&abort, 0, 1) {
						results <- struct {
							idx int
							rec [][]string
							err error
							dur time.Duration
						}{idx: idx, rec: nil, err: fmt.Errorf("open input %s: %w", inPath, err), dur: time.Since(t0)}
					}
					return
				}
				outF, err := os.Open(outPath)
				if err != nil {
					inF.Close()
					if atomic.CompareAndSwapInt32(&abort, 0, 1) {
						results <- struct {
							idx int
							rec [][]string
							err error
							dur time.Duration
						}{idx: idx, rec: nil, err: fmt.Errorf("open output %s: %w", outPath, err), dur: time.Since(t0)}
					}
					return
				}

				inR := csv.NewReader(inF)
				outR := csv.NewReader(outF)

				// read headers (ignore content here, but validate required columns exist)
				inHeader, err := inR.Read()
				if err != nil {
					inF.Close()
					outF.Close()
					if atomic.CompareAndSwapInt32(&abort, 0, 1) {
						results <- struct {
							idx int
							rec [][]string
							err error
							dur time.Duration
						}{idx: idx, rec: nil, err: fmt.Errorf("read input header %s: %w", inPath, err), dur: time.Since(t0)}
					}
					return
				}
				outHeader, err := outR.Read()
				if err != nil {
					inF.Close()
					outF.Close()
					if atomic.CompareAndSwapInt32(&abort, 0, 1) {
						results <- struct {
							idx int
							rec [][]string
							err error
							dur time.Duration
						}{idx: idx, rec: nil, err: fmt.Errorf("read output header %s: %w", outPath, err), dur: time.Since(t0)}
					}
					return
				}

				// build header maps
				inMap := make(map[string]int, len(inHeader))
				for i, h := range inHeader {
					inMap[strings.TrimSpace(strings.ToLower(h))] = i
				}
				outMap := make(map[string]int, len(outHeader))
				for i, h := range outHeader {
					outMap[strings.TrimSpace(strings.ToLower(h))] = i
				}
				// ensure required columns
				reqIn := []string{"x", "y", "s", "a", "o", "dir"}
				for _, r := range reqIn {
					if _, ok := inMap[r]; !ok {
						inF.Close()
						outF.Close()
						if atomic.CompareAndSwapInt32(&abort, 0, 1) {
							results <- struct {
								idx int
								rec [][]string
								err error
								dur time.Duration
							}{idx: idx, rec: nil, err: fmt.Errorf("input %s missing required column %s", inPath, r), dur: time.Since(t0)}
						}
						return
					}
				}
				// outputs: accept either explicit label columns (ball_land_x/ball_land_y)
				// or legacy x/y columns. Fail if neither pair exists.
				_, hasBallX := outMap["ball_land_x"]
				_, hasBallY := outMap["ball_land_y"]
				_, hasX := outMap["x"]
				_, hasY := outMap["y"]
				if !(hasBallX && hasBallY) && !(hasX && hasY) {
					inF.Close()
					outF.Close()
					if atomic.CompareAndSwapInt32(&abort, 0, 1) {
						results <- struct {
							idx int
							rec [][]string
							err error
							dur time.Duration
						}{idx: idx, rec: nil, err: fmt.Errorf("output %s missing required columns (need either ball_land_x/ball_land_y or x/y)", outPath), dur: time.Since(t0)}
					}
					return
				}

				records := make([][]string, 0)
				for {
					inRec, inErr := inR.Read()
					outRec, outErr := outR.Read()
					if inErr == io.EOF || outErr == io.EOF {
						break
					}
					if inErr != nil {
						inF.Close()
						outF.Close()
						if atomic.CompareAndSwapInt32(&abort, 0, 1) {
							results <- struct {
								idx int
								rec [][]string
								err error
								dur time.Duration
							}{idx: idx, rec: nil, err: fmt.Errorf("reading input %s: %w", inPath, inErr), dur: time.Since(t0)}
						}
						return
					}
					if outErr != nil {
						inF.Close()
						outF.Close()
						if atomic.CompareAndSwapInt32(&abort, 0, 1) {
							results <- struct {
								idx int
								rec [][]string
								err error
								dur time.Duration
							}{idx: idx, rec: nil, err: fmt.Errorf("reading output %s: %w", outPath, outErr), dur: time.Since(t0)}
						}
						return
					}
					// normalize into canonical order
					norm := make([]string, 8)
					for i, fname := range []string{"x", "y", "s", "a", "o", "dir"} {
						pos := inMap[fname]
						if pos < 0 || pos >= len(inRec) {
							norm[i] = ""
						} else {
							norm[i] = inRec[pos]
						}
					}
					// labels: prefer ball_land_x/ball_land_y, fall back to x/y
					norm[6] = ""
					norm[7] = ""
					if pos, ok := outMap["ball_land_x"]; ok && pos >= 0 && pos < len(outRec) {
						norm[6] = outRec[pos]
					} else if pos, ok := outMap["x"]; ok && pos >= 0 && pos < len(outRec) {
						norm[6] = outRec[pos]
					}
					if pos, ok := outMap["ball_land_y"]; ok && pos >= 0 && pos < len(outRec) {
						norm[7] = outRec[pos]
					} else if pos, ok := outMap["y"]; ok && pos >= 0 && pos < len(outRec) {
						norm[7] = outRec[pos]
					}
					records = append(records, norm)
				}
				inF.Close()
				outF.Close()

				results <- struct {
					idx int
					rec [][]string
					err error
					dur time.Duration
				}{idx: idx, rec: records, err: nil, dur: time.Since(t0)}
			}
		}()
	}

	// enqueue jobs
	for i := 0; i < numFiles; i++ {
		jobs <- i
	}
	close(jobs)

	// collect results and populate cache
	go func() {
		wg.Wait()
		close(results)
	}()

	var firstErr error
	loaded := 0
	startCollect := time.Now()
	for res := range results {
		if res.err != nil {
			if firstErr == nil {
				firstErr = res.err
			}
			atomic.StoreInt32(&abort, 1)
			fmt.Printf("[EnableCache] error caching pair %d: %v\n", res.idx, res.err)
			continue
		}
		d.cacheMu.Lock()
		d.cache[res.idx] = res.rec
		d.cacheMu.Unlock()
		loaded++
		// progress log occasionally
		if loaded%max(1, numFiles/10) == 0 || loaded == numFiles {
			elapsed := time.Since(startCollect)
			avg := elapsed / time.Duration(loaded)
			remaining := numFiles - loaded
			eta := time.Duration(0)
			if remaining > 0 {
				eta = avg * time.Duration(remaining)
			}
			fmt.Printf("[EnableCache] %d/%d cached elapsed=%s avg=%s eta=%s\n",
				loaded, numFiles, elapsed.Truncate(time.Millisecond), avg.Truncate(time.Millisecond), eta.Truncate(time.Millisecond))
		}
	}

	if firstErr != nil {
		return firstErr
	}

	d.cacheMu.Lock()
	d.cacheEnabled = true
	d.cacheMu.Unlock()
	fmt.Printf("[EnableCache] completed caching %d pairs in %s\n", loaded, time.Since(start).Truncate(time.Millisecond))
	return nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Len returns total examples
func (d *PredictionDataset) Len() int {
	d.metaMu.RLock()
	defer d.metaMu.RUnlock()
	return d.totalExamples
}

// mapGlobalIndex maps a global index to (pairIdx, localIdx) safely using snapshot.
func (d *PredictionDataset) mapGlobalIndex(globalIdx int) (pairIdx, localIdx int) {
	d.metaMu.RLock()
	cum := make([]int, len(d.cumCounts))
	copy(cum, d.cumCounts)
	rowC := make(map[int]int, len(d.rowCounts))
	for k, v := range d.rowCounts {
		rowC[k] = v
	}
	n := len(d.inputPaths)
	d.metaMu.RUnlock()

	if globalIdx < 0 {
		return 0, 0
	}
	if n == 0 {
		return 0, 0
	}

	// prefer cumulative counts
	if len(cum) >= n+1 {
		for i := 0; i < n; i++ {
			if globalIdx < cum[i+1] {
				local := globalIdx - cum[i]
				if cnt, ok := rowC[i]; ok && cnt > 0 && local >= cnt {
					local = cnt - 1
				}
				if local < 0 {
					local = 0
				}
				return i, local
			}
		}
	}

	// fallback linear scan
	total := 0
	for i := 0; i < n; i++ {
		cnt := rowC[i]
		if cnt < 0 {
			cnt = 0
		}
		if globalIdx < total+cnt {
			return i, globalIdx - total
		}
		total += cnt
	}

	last := n - 1
	if last < 0 {
		return 0, 0
	}
	lastCnt := rowC[last]
	if lastCnt <= 0 {
		return last, 0
	}
	return last, lastCnt - 1
}

// Example returns a single example by global index: inputs (6) and labels (2)
func (d *PredictionDataset) Example(idx int) ([]float32, []float32, error) {
	if idx < 0 {
		return nil, nil, fmt.Errorf("index %d out of range", idx)
	}
	pairIdx, local := d.mapGlobalIndex(idx)
	return d.readExample(pairIdx, local)
}

// readExample reads one example for a given pair index and row index
func (d *PredictionDataset) readExample(pairIdx, rowIdx int) ([]float32, []float32, error) {
	// try cache
	d.cacheMu.RLock()
	enabled := d.cacheEnabled
	if enabled {
		if recs, ok := d.cache[pairIdx]; ok {
			d.cacheMu.RUnlock()
			if rowIdx < 0 || rowIdx >= len(recs) {
				return nil, nil, fmt.Errorf("row index %d out of range for cached pair %d", rowIdx, pairIdx)
			}
			rec := recs[rowIdx]
			// rec is normalized: 8 columns
			inputs := make([]float32, 6)
			for i := 0; i < 6; i++ {
				f, err := parseFloat32(rec[i])
				if err != nil {
					return nil, nil, fmt.Errorf("parse input[%d]: %w", i, err)
				}
				inputs[i] = f
			}
			labels := make([]float32, 2)
			lx, err := parseFloat32(rec[6])
			if err != nil {
				return nil, nil, fmt.Errorf("parse ball_land_x: %w", err)
			}
			ly, err := parseFloat32(rec[7])
			if err != nil {
				return nil, nil, fmt.Errorf("parse ball_land_y: %w", err)
			}
			labels[0] = lx
			labels[1] = ly
			return inputs, labels, nil
		}
	}
	d.cacheMu.RUnlock()

	// not cached: read from files directly using per-file header mapping
	if pairIdx < 0 || pairIdx >= len(d.inputPaths) {
		return nil, nil, fmt.Errorf("pair index %d out of range", pairIdx)
	}
	inPath := d.inputPaths[pairIdx]
	outPath := d.outputPaths[pairIdx]

	// Acquire read locks
	d.inputLocks[pairIdx].RLock()
	d.outputLocks[pairIdx].RLock()
	defer d.inputLocks[pairIdx].RUnlock()
	defer d.outputLocks[pairIdx].RUnlock()

	inF, err := os.Open(inPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer inF.Close()
	outF, err := os.Open(outPath)
	if err != nil {
		return nil, nil, fmt.Errorf("open output %s: %w", outPath, err)
	}
	defer outF.Close()

	inR := csv.NewReader(inF)
	outR := csv.NewReader(outF)

	inHeader, err := inR.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("read input header: %w", err)
	}
	outHeader, err := outR.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("read output header: %w", err)
	}
	inMap := make(map[string]int)
	for i, h := range inHeader {
		inMap[strings.TrimSpace(strings.ToLower(h))] = i
	}
	outMap := make(map[string]int)
	for i, h := range outHeader {
		outMap[strings.TrimSpace(strings.ToLower(h))] = i
	}

	// seek to rowIdx by reading rows sequentially
	var inRec, outRec []string
	for i := 0; i <= rowIdx; i++ {
		inRec, err = inR.Read()
		if err != nil {
			return nil, nil, fmt.Errorf("read input row %d: %w", rowIdx, err)
		}
		outRec, err = outR.Read()
		if err != nil {
			return nil, nil, fmt.Errorf("read output row %d: %w", rowIdx, err)
		}
	}

	// parse features
	inputs := make([]float32, 6)
	features := []string{"x", "y", "s", "a", "o", "dir"}
	for i, fName := range features {
		pos, ok := inMap[fName]
		if !ok || pos < 0 || pos >= len(inRec) {
			return nil, nil, fmt.Errorf("input row missing feature %s pos=%d", fName, pos)
		}
		v, err := parseFloat32(inRec[pos])
		if err != nil {
			return nil, nil, fmt.Errorf("parse feature %s: %w", fName, err)
		}
		inputs[i] = v
	}

	// parse labels: accept ball_land_x/ball_land_y or fallback to x/y
	bxPos, ok := outMap["ball_land_x"]
	if !ok {
		bxPos, ok = outMap["x"]
	}
	if !ok || bxPos < 0 || bxPos >= len(outRec) {
		return nil, nil, fmt.Errorf("output row missing ball_land_x/x pos=%d", bxPos)
	}
	byPos, ok := outMap["ball_land_y"]
	if !ok {
		byPos, ok = outMap["y"]
	}
	if !ok || byPos < 0 || byPos >= len(outRec) {
		return nil, nil, fmt.Errorf("output row missing ball_land_y/y pos=%d", byPos)
	}
	bx, err := parseFloat32(outRec[bxPos])
	if err != nil {
		return nil, nil, fmt.Errorf("parse ball_land_x/x: %w", err)
	}
	by, err := parseFloat32(outRec[byPos])
	if err != nil {
		return nil, nil, fmt.Errorf("parse ball_land_y/y: %w", err)
	}
	labels := make([]float32, 2)
	labels[0] = bx
	labels[1] = by
	return inputs, labels, nil
}

// Batch reads multiple examples by global indices
func (d *PredictionDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	inputs := make([][]float32, len(indices))
	labels := make([][]float32, len(indices))

	// group indices by pair
	groups := make(map[int][]struct {
		global int
		pos    int
	})
	for i, gi := range indices {
		pair, _ := d.mapGlobalIndex(gi)
		groups[pair] = append(groups[pair], struct {
			global int
			pos    int
		}{global: gi, pos: i})
	}

	for pairIdx, group := range groups {
		if err := d.readBatchFromFile(pairIdx, group, inputs, labels); err != nil {
			return nil, nil, err
		}
	}
	return inputs, labels, nil
}

// readBatchFromFile reads several examples from a single pair file (input+output)
// indices is a slice of structs with global and target position in batch.
func (d *PredictionDataset) readBatchFromFile(pairIdx int, indices []struct {
	global int
	pos    int
}, inputs, labels [][]float32) error {
	// Fast path: cache
	d.cacheMu.RLock()
	enabled := d.cacheEnabled
	recs, cached := d.cache[pairIdx]
	d.cacheMu.RUnlock()
	// Map local indices to batch positions
	localMap := make(map[int]int)
	for _, it := range indices {
		_, local := d.mapGlobalIndex(it.global)
		localMap[local] = it.pos
	}

	if enabled && cached {
		for localIdx, batchPos := range localMap {
			if localIdx < 0 || localIdx >= len(recs) {
				return fmt.Errorf("cached local index %d out of range for pair %d", localIdx, pairIdx)
			}
			rec := recs[localIdx]
			// parse normalized record
			inputs[batchPos] = make([]float32, 6)
			for i := 0; i < 6; i++ {
				v, err := parseFloat32(rec[i])
				if err != nil {
					return fmt.Errorf("parse cached input[%d]: %w", i, err)
				}
				inputs[batchPos][i] = v
			}
			labels[batchPos] = make([]float32, 2)
			bx, err := parseFloat32(rec[6])
			if err != nil {
				return fmt.Errorf("parse cached ball_land_x: %w", err)
			}
			by, err := parseFloat32(rec[7])
			if err != nil {
				return fmt.Errorf("parse cached ball_land_y: %w", err)
			}
			labels[batchPos][0] = bx
			labels[batchPos][1] = by
		}
		return nil
	}

	// Not cached: read from files with per-file locks
	if pairIdx < 0 || pairIdx >= len(d.inputPaths) {
		return fmt.Errorf("pairIdx %d out of range", pairIdx)
	}
	inPath := d.inputPaths[pairIdx]
	outPath := d.outputPaths[pairIdx]

	// lock both files for reading
	d.inputLocks[pairIdx].RLock()
	d.outputLocks[pairIdx].RLock()
	defer d.inputLocks[pairIdx].RUnlock()
	defer d.outputLocks[pairIdx].RUnlock()

	inF, err := os.Open(inPath)
	if err != nil {
		return fmt.Errorf("open input %s: %w", inPath, err)
	}
	defer inF.Close()
	outF, err := os.Open(outPath)
	if err != nil {
		return fmt.Errorf("open output %s: %w", outPath, err)
	}
	defer outF.Close()

	inR := csv.NewReader(inF)
	outR := csv.NewReader(outF)

	inHeader, err := inR.Read()
	if err != nil {
		return fmt.Errorf("read input header: %w", err)
	}
	outHeader, err := outR.Read()
	if err != nil {
		return fmt.Errorf("read output header: %w", err)
	}
	inMap := make(map[string]int)
	for i, h := range inHeader {
		inMap[strings.TrimSpace(strings.ToLower(h))] = i
	}
	outMap := make(map[string]int)
	for i, h := range outHeader {
		outMap[strings.TrimSpace(strings.ToLower(h))] = i
	}

	// We'll iterate through the file once and extract requested rows.
	targets := make(map[int]int) // rowIdx -> batchPos
	minRow := -1
	maxRow := -1
	for rowIdx := range localMap {
		targets[rowIdx] = localMap[rowIdx]
		if minRow == -1 || rowIdx < minRow {
			minRow = rowIdx
		}
		if maxRow == -1 || rowIdx > maxRow {
			maxRow = rowIdx
		}
	}
	if minRow < 0 {
		return nil
	}

	row := 0
	for {
		inRec, inErr := inR.Read()
		outRec, outErr := outR.Read()
		if inErr == io.EOF || outErr == io.EOF {
			break
		}
		if inErr != nil {
			return fmt.Errorf("read input row: %w", inErr)
		}
		if outErr != nil {
			return fmt.Errorf("read output row: %w", outErr)
		}
		if batchPos, ok := targets[row]; ok {
			// parse features
			inputs[batchPos] = make([]float32, 6)
			for i, fname := range []string{"x", "y", "s", "a", "o", "dir"} {
				pos, ok := inMap[fname]
				if !ok || pos < 0 || pos >= len(inRec) {
					return fmt.Errorf("input missing feature %s at row %d", fname, row)
				}
				v, err := parseFloat32(inRec[pos])
				if err != nil {
					return fmt.Errorf("parse input %s at row %d: %w", fname, row, err)
				}
				inputs[batchPos][i] = v
			}
			// parse labels
			labels[batchPos] = make([]float32, 2)
			if pos, ok := outMap["ball_land_x"]; !ok || pos < 0 || pos >= len(outRec) {
				return fmt.Errorf("output missing ball_land_x at row %d", row)
			} else {
				bx, err := parseFloat32(outRec[pos])
				if err != nil {
					return fmt.Errorf("parse ball_land_x at row %d: %w", row, err)
				}
				labels[batchPos][0] = bx
			}
			if pos, ok := outMap["ball_land_y"]; !ok || pos < 0 || pos >= len(outRec) {
				return fmt.Errorf("output missing ball_land_y at row %d", row)
			} else {
				by, err := parseFloat32(outRec[pos])
				if err != nil {
					return fmt.Errorf("parse ball_land_y at row %d: %w", row, err)
				}
				labels[batchPos][1] = by
			}
		}
		row++
		if row > maxRow {
			break
		}
	}
	return nil
}

// Shuffle seeds the RNG. Implementing full shuffling of a lazy dataset would
// require maintaining a shuffled index mapping; out of scope for this change.
func (d *PredictionDataset) Shuffle(seed int64) {
	d.rand.Seed(seed)
}

// Tensors converts a Batch into gomlx tensors.
func (d *PredictionDataset) Tensors(indices []int) (*tensors.Tensor, *tensors.Tensor, error) {
	in, la, err := d.Batch(indices)
	if err != nil {
		return nil, nil, err
	}
	pb, err := MakePredictionBatchFlat(in, la)
	if err != nil {
		return nil, nil, err
	}
	return pb.ToGomlxTensors()
}

// Name returns a descriptive name
func (d *PredictionDataset) Name() string {
	return "PredictionDataset"
}

// Yield implements a single-batch yield (not a full Dataset epoch iterator).
func (d *PredictionDataset) Yield() (any, []*tensors.Tensor, []*tensors.Tensor, error) {
	indices := make([]int, d.BatchSize)
	inT, labT, err := d.Tensors(indices)
	if err != nil {
		return nil, nil, nil, err
	}
	return nil, []*tensors.Tensor{inT}, []*tensors.Tensor{labT}, nil
}

// Restart is a noop for lazy dataset.
func (d *PredictionDataset) Restart() error {
	return nil
}

// PredictionBatchFlat holds flat buffers.
type PredictionBatchFlat struct {
	Inputs    []float32
	Labels    []float32
	BatchSize int
	InputDim  int
	LabelDim  int
}

// MakePredictionBatchFlat flattens 2D slices into contiguous flat buffers.
func MakePredictionBatchFlat(inputs, labels [][]float32) (*PredictionBatchFlat, error) {
	if len(inputs) != len(labels) {
		return nil, fmt.Errorf("inputs/labels size mismatch: %d != %d", len(inputs), len(labels))
	}
	if len(inputs) == 0 {
		return &PredictionBatchFlat{BatchSize: 0, InputDim: 0, LabelDim: 0}, nil
	}
	batch := len(inputs)
	inputDim := len(inputs[0])
	labelDim := len(labels[0])

	flatIn := make([]float32, batch*inputDim)
	flatLab := make([]float32, batch*labelDim)

	for i := 0; i < batch; i++ {
		if len(inputs[i]) != inputDim {
			return nil, fmt.Errorf("inconsistent input dim at %d", i)
		}
		if len(labels[i]) != labelDim {
			return nil, fmt.Errorf("inconsistent label dim at %d", i)
		}
		copy(flatIn[i*inputDim:(i+1)*inputDim], inputs[i])
		copy(flatLab[i*labelDim:(i+1)*labelDim], labels[i])
	}
	return &PredictionBatchFlat{
		Inputs:    flatIn,
		Labels:    flatLab,
		BatchSize: batch,
		InputDim:  inputDim,
		LabelDim:  labelDim,
	}, nil
}

// ToGomlxTensors converts flat batch into gomlx tensors.
func (b *PredictionBatchFlat) ToGomlxTensors() (*tensors.Tensor, *tensors.Tensor, error) {
	if b.BatchSize == 0 || b.InputDim == 0 || b.LabelDim == 0 {
		emptyInputs := make([][]float32, 0)
		emptyLabels := make([][]float32, 0)
		return tensors.FromAnyValue(emptyInputs), tensors.FromAnyValue(emptyLabels), nil
	}
	inputs := make([][]float32, b.BatchSize)
	labels := make([][]float32, b.BatchSize)
	for i := 0; i < b.BatchSize; i++ {
		inputs[i] = b.Inputs[i*b.InputDim : (i+1)*b.InputDim]
		labels[i] = b.Labels[i*b.LabelDim : (i+1)*b.LabelDim]
	}
	return tensors.FromAnyValue(inputs), tensors.FromAnyValue(labels), nil
}

// parseFloat32 is provided by datasets/util.go to avoid duplication across files.
