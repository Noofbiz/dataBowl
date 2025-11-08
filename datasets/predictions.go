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

// PredictionDataset provides a gomlx train.Dataset interface that lazily
// loads CSV files matching a given pattern for the prediction portion of the
// competition.
// Each CSV file is expected to have columns:
// "x", "y", "s", "a", "o", "dir", "ball_land_x", "ball_land_y".
type PredictionDataset struct {
	// Pattern used to find CSV files (e.g., "assets/kaggle/*.csv")
	Pattern string

	// BatchSize for yielding batches
	BatchSize int

	// List of CSV file paths matching the pattern
	csvPaths []string

	// Column indices for features and labels (discovered from first file)
	colIndex map[string]int

	// Random generator for shuffling
	rand *rand.Rand

	// Cache for counting rows in each file (index -> row count)
	rowCounts map[int]int

	// Cumulative counts for fast index mapping
	cumCounts []int

	// Total number of examples across all files
	totalExamples int
}

// NewPredictionDataset creates a new prediction dataset that lazily loads
// CSV files matching the given pattern.
func NewPredictionDataset(pattern string) (*PredictionDataset, error) {
	// Find all CSV files matching the pattern
	csvPaths, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to glob pattern %s: %w", pattern, err)
	}
	if len(csvPaths) == 0 {
		return nil, fmt.Errorf("no CSV files found matching pattern: %s", pattern)
	}

	ds := &PredictionDataset{
		Pattern:   pattern,
		BatchSize: 32,
		csvPaths:  csvPaths,
		rand:      rand.New(rand.NewSource(time.Now().UnixNano())),
		rowCounts: make(map[int]int),
	}

	// Read the first file to determine column structure
	if err := ds.initializeColumns(); err != nil {
		return nil, err
	}

	// Count rows in all files to build the index
	if err := ds.buildIndex(); err != nil {
		return nil, err
	}

	return ds, nil
}

// initializeColumns reads the first CSV to determine column indices
func (d *PredictionDataset) initializeColumns() error {
	file, err := os.Open(d.csvPaths[0])
	if err != nil {
		return fmt.Errorf("failed to open first CSV %s: %w", d.csvPaths[0], err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	d.colIndex = make(map[string]int)
	for i, col := range header {
		d.colIndex[strings.TrimSpace(strings.ToLower(col))] = i
	}

	// Verify required columns exist
	required := []string{"x", "y", "s", "a", "o", "dir", "ball_land_x", "ball_land_y"}
	for _, col := range required {
		if _, ok := d.colIndex[col]; !ok {
			return fmt.Errorf("required column %q not found in CSV", col)
		}
	}

	return nil
}

// buildIndex counts rows in all files and builds cumulative counts
func (d *PredictionDataset) buildIndex() error {
	d.cumCounts = make([]int, len(d.csvPaths)+1)
	d.cumCounts[0] = 0

	for i, path := range d.csvPaths {
		count, err := countCSVRows(path)
		if err != nil {
			return fmt.Errorf("failed to count rows in %s: %w", path, err)
		}
		d.rowCounts[i] = count
		d.cumCounts[i+1] = d.cumCounts[i] + count
	}

	d.totalExamples = d.cumCounts[len(d.csvPaths)]
	return nil
}

// Len returns the total number of examples across all CSV files
func (d *PredictionDataset) Len() int {
	return d.totalExamples
}

// Example reads a single example by global index
func (d *PredictionDataset) Example(idx int) (inputs []float32, labels []float32, err error) {
	if idx < 0 || idx >= d.totalExamples {
		return nil, nil, fmt.Errorf("index %d out of range [0, %d)", idx, d.totalExamples)
	}

	// Find which file contains this index
	fileIdx, localIdx := d.mapGlobalIndex(idx)

	// Read the specific row from the file
	return d.readExample(fileIdx, localIdx)
}

// mapGlobalIndex maps a global index to (file index, row index within file)
func (d *PredictionDataset) mapGlobalIndex(globalIdx int) (fileIdx, localIdx int) {
	// Binary search for the file containing this index
	for i := range len(d.csvPaths) {
		if globalIdx < d.cumCounts[i+1] {
			return i, globalIdx - d.cumCounts[i]
		}
	}
	// Should never reach here if globalIdx is valid
	return len(d.csvPaths) - 1, d.rowCounts[len(d.csvPaths)-1] - 1
}

// readExample reads a specific example from a file
func (d *PredictionDataset) readExample(fileIdx, rowIdx int) ([]float32, []float32, error) {
	file, err := os.Open(d.csvPaths[fileIdx])
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Skip to the desired row
	for range rowIdx {
		if _, err := reader.Read(); err != nil {
			return nil, nil, fmt.Errorf("failed to skip to row %d: %w", rowIdx, err)
		}
	}

	// Read the target row
	record, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read row %d: %w", rowIdx, err)
	}

	// Extract features
	inputs := make([]float32, 6)
	features := []string{"x", "y", "s", "a", "o", "dir"}
	for i, feat := range features {
		val, err := parseFloat32(record[d.colIndex[feat]])
		if err != nil {
			return nil, nil, fmt.Errorf("failed to parse %s: %w", feat, err)
		}
		inputs[i] = val
	}

	// Extract labels
	labels := make([]float32, 2)
	ballX, err := parseFloat32(record[d.colIndex["ball_land_x"]])
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse ball_land_x: %w", err)
	}
	ballY, err := parseFloat32(record[d.colIndex["ball_land_y"]])
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse ball_land_y: %w", err)
	}
	labels[0] = ballX
	labels[1] = ballY

	return inputs, labels, nil
}

// Batch reads multiple examples by their indices
func (d *PredictionDataset) Batch(indices []int) ([][]float32, [][]float32, error) {
	inputs := make([][]float32, len(indices))
	labels := make([][]float32, len(indices))

	// Group indices by file for more efficient reading
	fileGroups := make(map[int][]struct{ globalIdx, batchPos int })
	for batchPos, idx := range indices {
		fileIdx, _ := d.mapGlobalIndex(idx)
		fileGroups[fileIdx] = append(fileGroups[fileIdx], struct{ globalIdx, batchPos int }{idx, batchPos})
	}

	// Process each file's indices together
	for fileIdx, group := range fileGroups {
		if err := d.readBatchFromFile(fileIdx, group, inputs, labels); err != nil {
			return nil, nil, err
		}
	}

	return inputs, labels, nil
}

// readBatchFromFile reads multiple examples from a single file
func (d *PredictionDataset) readBatchFromFile(fileIdx int, indices []struct{ globalIdx, batchPos int },
	inputs, labels [][]float32) error {

	file, err := os.Open(d.csvPaths[fileIdx])
	if err != nil {
		return fmt.Errorf("failed to open CSV: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	// Create a map of local indices to batch positions
	localMap := make(map[int]int)
	for _, item := range indices {
		_, localIdx := d.mapGlobalIndex(item.globalIdx)
		localMap[localIdx] = item.batchPos
	}

	// Read through the file
	rowIdx := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read row: %w", err)
		}

		if batchPos, ok := localMap[rowIdx]; ok {
			// Extract features
			inputs[batchPos] = make([]float32, 6)
			features := []string{"x", "y", "s", "a", "o", "dir"}
			for i, feat := range features {
				val, err := parseFloat32(record[d.colIndex[feat]])
				if err != nil {
					return fmt.Errorf("failed to parse %s: %w", feat, err)
				}
				inputs[batchPos][i] = val
			}

			// Extract labels
			labels[batchPos] = make([]float32, 2)
			ballX, err := parseFloat32(record[d.colIndex["ball_land_x"]])
			if err != nil {
				return fmt.Errorf("failed to parse ball_land_x: %w", err)
			}
			ballY, err := parseFloat32(record[d.colIndex["ball_land_y"]])
			if err != nil {
				return fmt.Errorf("failed to parse ball_land_y: %w", err)
			}
			labels[batchPos][0] = ballX
			labels[batchPos][1] = ballY
		}

		rowIdx++
	}

	return nil
}

// Shuffle shuffles the order of examples
func (d *PredictionDataset) Shuffle(seed int64) {
	d.rand.Seed(seed)
	// For lazy loading, we'll need to implement shuffling at the index level
	// This would require maintaining a shuffled index mapping
	// For now, this is a placeholder - actual implementation would depend on requirements
}

// Tensors reads a batch of examples and returns them as gomlx tensors
func (d *PredictionDataset) Tensors(indices []int) (inputs *tensors.Tensor, labels *tensors.Tensor, err error) {
	inData, labData, err := d.Batch(indices)
	if err != nil {
		return nil, nil, err
	}

	pbatch, err := MakePredictionBatchFlat(inData, labData)
	if err != nil {
		return nil, nil, err
	}

	return pbatch.ToGomlxTensors()
}

// Name returns the name of the dataset
func (d *PredictionDataset) Name() string {
	return "PredictionDataset"
}

// Yield returns the next batch of data for the gomlx Dataset interface. Batch
// is determined by the BatchSize field.
func (d *PredictionDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	indices := make([]int, d.BatchSize)
	in, la, err := d.Tensors(indices)
	if err != nil {
		return nil, nil, nil, err
	}
	inputs = []*tensors.Tensor{in}
	labels = []*tensors.Tensor{la}
	return nil, inputs, labels, nil
}

// Restart resets the dataset for a new epoch
func (d *PredictionDataset) Restart() error {
	// For lazy loading, no internal state to reset
	return nil
}

// PredictionBatchFlat stores a batch in flat contiguous buffers
type PredictionBatchFlat struct {
	Inputs    []float32
	Labels    []float32
	BatchSize int
	InputDim  int
	LabelDim  int
}

// MakePredictionBatchFlat flattens a batch into contiguous buffers
func MakePredictionBatchFlat(inputs, labels [][]float32) (*PredictionBatchFlat, error) {
	if len(inputs) != len(labels) {
		return nil, fmt.Errorf("inputs and labels batch sizes don't match: %d != %d", len(inputs), len(labels))
	}
	if len(inputs) == 0 {
		return &PredictionBatchFlat{BatchSize: 0, InputDim: 0, LabelDim: 0}, nil
	}

	batchSize := len(inputs)
	inputDim := len(inputs[0])
	labelDim := len(labels[0])

	flatInputs := make([]float32, batchSize*inputDim)
	flatLabels := make([]float32, batchSize*labelDim)

	for i := range batchSize {
		if len(inputs[i]) != inputDim {
			return nil, fmt.Errorf("inconsistent input dimensions at example %d: expected %d, got %d",
				i, inputDim, len(inputs[i]))
		}
		if len(labels[i]) != labelDim {
			return nil, fmt.Errorf("inconsistent label dimensions at example %d: expected %d, got %d",
				i, labelDim, len(labels[i]))
		}
		copy(flatInputs[i*inputDim:], inputs[i])
		copy(flatLabels[i*labelDim:], labels[i])
	}

	return &PredictionBatchFlat{
		Inputs:    flatInputs,
		Labels:    flatLabels,
		BatchSize: batchSize,
		InputDim:  inputDim,
		LabelDim:  labelDim,
	}, nil
}

// ToGomlxTensors converts PredictionBatchFlat to gomlx tensors
func (b *PredictionBatchFlat) ToGomlxTensors() (*tensors.Tensor, *tensors.Tensor, error) {
	// handle empty batch gracefully
	if b.BatchSize == 0 || b.InputDim == 0 || b.LabelDim == 0 {
		emptyInputs := make([][]float32, 0)
		emptyLabels := make([][]float32, 0)
		inT := tensors.FromAnyValue(emptyInputs)
		labT := tensors.FromAnyValue(emptyLabels)
		return inT, labT, nil
	}
	// Reshape flat data into 2D slices
	inputs := make([][]float32, b.BatchSize)
	labels := make([][]float32, b.BatchSize)
	for i := range b.BatchSize {
		inputs[i] = b.Inputs[i*b.InputDim : (i+1)*b.InputDim]
		labels[i] = b.Labels[i*b.LabelDim : (i+1)*b.LabelDim]
	}
	inT := tensors.FromAnyValue(inputs)
	labT := tensors.FromAnyValue(labels)
	return inT, labT, nil
}
