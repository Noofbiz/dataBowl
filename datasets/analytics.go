package datasets

import (
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// AnalyticsDataset
// - Stores paths to CSV files with positional/time-series data
// - Loads CSVs on-demand where examples are grouped by a play identifier
// - If the CSVs are not in the file passed, download the data from kaggle.
// - Each example is a time-ordered sequence of positions (x,y) and optionally other channels
// - No labels are provided by default (exploratory / regression use)

// NewAnalyticsDataset creates a new analytics dataset with lazy loading
func NewAnalyticsDataset(pattern string, playIDCol string, seqColNames []string) (*AnalyticsDataset, error) {
	// Find all CSV files matching the pattern
	csvPaths, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to glob pattern %s: %w", pattern, err)
	}
	if len(csvPaths) == 0 {
		return nil, fmt.Errorf("no CSV files found matching pattern: %s", pattern)
	}

	ds := &AnalyticsDataset{
		Pattern:     pattern,
		csvPaths:    csvPaths,
		seqColNames: seqColNames,
		rand:        rand.New(rand.NewSource(time.Now().UnixNano())),
		filePlayIDs: make(map[int][]string),
		playLocations: make(map[string]struct {
			fileIdx int
			rows    []int
		}),
	}

	// Initialize column indices from first file
	if err := ds.initializeColumns(playIDCol); err != nil {
		return nil, err
	}

	// Build index of play IDs across all files
	if err := ds.buildPlayIndex(); err != nil {
		return nil, err
	}

	return ds, nil
}

// initializeColumns determines column indices from the first file
func (a *AnalyticsDataset) initializeColumns(playIDCol string) error {
	file, err := os.Open(a.csvPaths[0])
	if err != nil {
		return fmt.Errorf("failed to open first CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	colIndex := make(map[string]int)
	for i, col := range header {
		normalized := strings.TrimSpace(strings.ToLower(col))
		colIndex[normalized] = i
	}

	// Find play ID column
	playIdx := -1
	if playIDCol != "" {
		if idx, ok := colIndex[strings.ToLower(playIDCol)]; ok {
			playIdx = idx
		}
	}
	if playIdx == -1 {
		// Try common names
		for _, name := range []string{"playid", "play_id", "play"} {
			if idx, ok := colIndex[name]; ok {
				playIdx = idx
				break
			}
		}
	}
	if playIdx == -1 {
		return fmt.Errorf("could not find play ID column")
	}
	a.playIDCol = playIdx

	// Find sequence columns
	a.seqCols = make([]int, len(a.seqColNames))
	for i, col := range a.seqColNames {
		if idx, ok := colIndex[strings.ToLower(col)]; ok {
			a.seqCols[i] = idx
		} else {
			return fmt.Errorf("sequence column %q not found", col)
		}
	}

	return nil
}

// buildPlayIndex scans all files to build an index of play IDs
func (a *AnalyticsDataset) buildPlayIndex() error {
	allPlaysMap := make(map[string]bool)

	for fileIdx, path := range a.csvPaths {
		plays, err := a.scanFileForPlays(fileIdx, path)
		if err != nil {
			return fmt.Errorf("failed to scan %s: %w", path, err)
		}

		a.filePlayIDs[fileIdx] = plays
		for _, play := range plays {
			allPlaysMap[play] = true
		}
	}

	// Convert map to slice
	a.allPlayIDs = make([]string, 0, len(allPlaysMap))
	for play := range allPlaysMap {
		a.allPlayIDs = append(a.allPlayIDs, play)
	}

	return nil
}

// scanFileForPlays scans a file and returns unique play IDs with their locations
func (a *AnalyticsDataset) scanFileForPlays(fileIdx int, path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, err
	}

	playRows := make(map[string][]int)
	rowIdx := 0

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		playID := record[a.playIDCol]
		playRows[playID] = append(playRows[playID], rowIdx)
		rowIdx++
	}

	// Store locations and extract unique plays
	uniquePlays := make([]string, 0, len(playRows))
	for playID, rows := range playRows {
		a.playLocations[playID] = struct {
			fileIdx int
			rows    []int
		}{fileIdx, rows}
		uniquePlays = append(uniquePlays, playID)
	}

	return uniquePlays, nil
}

// Len returns the number of unique plays
func (a *AnalyticsDataset) Len() int {
	return len(a.allPlayIDs)
}

// Example returns a single play's data by index
func (a *AnalyticsDataset) Example(idx int) (sequence []float32, shape []int, err error) {
	if idx < 0 || idx >= len(a.allPlayIDs) {
		return nil, nil, fmt.Errorf("index %d out of range", idx)
	}

	playID := a.allPlayIDs[idx]
	return a.loadPlay(playID)
}

// loadPlay loads data for a specific play ID
func (a *AnalyticsDataset) loadPlay(playID string) ([]float32, []int, error) {
	loc, ok := a.playLocations[playID]
	if !ok {
		return nil, nil, fmt.Errorf("play ID %s not found", playID)
	}

	file, err := os.Open(a.csvPaths[loc.fileIdx])
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, nil, err
	}

	// Read all rows for this play
	timeSteps := len(loc.rows)
	channels := len(a.seqCols)
	sequence := make([]float32, timeSteps*channels)

	currentRow := 0
	seqIdx := 0
	for record, err := reader.Read(); err != io.EOF; record, err = reader.Read() {
		if err != nil {
			return nil, nil, err
		}

		// Check if this row is one we need
		for _, targetRow := range loc.rows {
			if currentRow == targetRow {
				// Extract sequence values
				for _, colIdx := range a.seqCols {
					if colIdx >= len(record) {
						continue
					}
					val, err := parseFloat32(record[colIdx])
					if err != nil {
						return nil, nil, fmt.Errorf("failed to parse value: %w", err)
					}
					sequence[seqIdx] = val
					seqIdx++
				}
				break
			}
		}
		currentRow++
	}

	shape := []int{timeSteps, channels}
	return sequence, shape, nil
}

// Batch loads multiple plays
func (a *AnalyticsDataset) Batch(indices []int) ([][]float32, [][]int, error) {
	sequences := make([][]float32, len(indices))
	shapes := make([][]int, len(indices))

	for i, idx := range indices {
		seq, shape, err := a.Example(idx)
		if err != nil {
			return nil, nil, err
		}
		sequences[i] = seq
		shapes[i] = shape
	}

	return sequences, shapes, nil
}

// Shuffle shuffles the order of plays
func (a *AnalyticsDataset) Shuffle(seed int64) {
	a.rand.Seed(seed)
	a.rand.Shuffle(len(a.allPlayIDs), func(i, j int) {
		a.allPlayIDs[i], a.allPlayIDs[j] = a.allPlayIDs[j], a.allPlayIDs[i]
	})
}

// AnalyticsBatchFlat stores analytics batch data in flat format
type AnalyticsBatchFlat struct {
	Buf      []float32
	Batch    int
	Time     int
	Channels int
}

// MakeAnalyticsBatchFlat creates a flat batch from analytics data
func MakeAnalyticsBatchFlat(buffers [][]float32, shapes [][]int) (*AnalyticsBatchFlat, error) {
	if len(buffers) == 0 {
		return &AnalyticsBatchFlat{}, nil
	}

	// Check that all sequences have the same shape
	timeSteps := shapes[0][0]
	channels := shapes[0][1]

	for i := 1; i < len(shapes); i++ {
		if shapes[i][0] != timeSteps || shapes[i][1] != channels {
			return nil, fmt.Errorf("inconsistent shapes: sequence 0 has shape %v, sequence %d has shape %v",
				shapes[0], i, shapes[i])
		}
	}

	batchSize := len(buffers)
	totalSize := batchSize * timeSteps * channels
	flat := make([]float32, totalSize)

	// Copy data into flat buffer
	for i, buf := range buffers {
		if len(buf) != timeSteps*channels {
			return nil, fmt.Errorf("buffer %d has wrong size: expected %d, got %d",
				i, timeSteps*channels, len(buf))
		}
		copy(flat[i*timeSteps*channels:], buf)
	}

	return &AnalyticsBatchFlat{
		Buf:      flat,
		Batch:    batchSize,
		Time:     timeSteps,
		Channels: channels,
	}, nil
}

// ToGomlxTensor converts AnalyticsBatchFlat to a gomlx tensor
func (b *AnalyticsBatchFlat) ToGomlxTensor() (*tensors.Tensor, error) {
	if b.Batch == 0 || b.Time == 0 || b.Channels == 0 {
		empty := make([][][]float32, 0)
		return tensors.FromAnyValue(empty), nil
	}
	// Reshape flat buffer into 3D slice
	data := make([][][]float32, b.Batch)
	idx := 0
	for i := 0; i < b.Batch; i++ {
		data[i] = make([][]float32, b.Time)
		for j := 0; j < b.Time; j++ {
			data[i][j] = b.Buf[idx : idx+b.Channels]
			idx += b.Channels
		}
	}
	t := tensors.FromAnyValue(data)
	return t, nil
}
