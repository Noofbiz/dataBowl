package main

// Example command that demonstrates loading the prediction and analytics
// datasets with the convenience auto-discovery helpers and converting small
// batches into gomlx tensors using the helpers provided in the package.
//
// The datasets now use lazy loading - they store file paths and only read
// the actual CSV data when needed, minimizing memory usage.
//
// Usage:
//   go run ./example
//
// Note: this example expects CSVs to exist under the repository's
// assets/kaggle/... paths (the AutoNew* helpers try several common locations).
// If no CSV is found the example will print an error and exit.

import (
	"fmt"
	"log"

	"github.com/Noofbiz/dataBowl/datasets"
)

func main() {
	// Prediction dataset: auto-discover CSV pattern and create lazy-loading dataset
	predPattern := "../assets/kaggle/prediction/train/*.csv"
	predDS, err := datasets.NewPredictionDataset(predPattern, "")
	if err != nil {
		log.Fatalf("failed to auto-load prediction dataset: %v", err)
	}
	fmt.Printf("Using prediction CSV pattern: %s\n", predPattern)
	fmt.Printf("Total prediction examples available: %d\n", predDS.Len())

	// Prepare a small batch (first N examples)
	n := min(8, predDS.Len())
	if n > 0 {
		indices := make([]int, n)
		for i := range n {
			indices[i] = i
		}

		fmt.Printf("Loading batch of %d prediction examples...\n", n)
		inputs, labels, err := predDS.Batch(indices)
		if err != nil {
			log.Fatalf("failed to build prediction batch: %v", err)
		}

		// Convert to flat contiguous buffers and then to gomlx tensors
		predFlat, err := datasets.MakePredictionBatchFlat(inputs, labels)
		if err != nil {
			log.Fatalf("failed to make prediction batch flat: %v", err)
		}

		inT, laT, err := predFlat.ToGomlxTensors()
		if err != nil {
			log.Fatalf("failed to convert prediction batch to gomlx tensors: %v", err)
		}

		// We don't depend on any particular tensor API here; just show we have tensors.
		fmt.Printf("Created prediction tensors: input=%T label=%T\n", inT, laT)
		fmt.Printf("  Input shape: [%d, %d]\n", predFlat.BatchSize, predFlat.InputDim)
		fmt.Printf("  Label shape: [%d, %d]\n", predFlat.BatchSize, predFlat.LabelDim)

		// Show first example's data
		if len(inputs) > 0 {
			fmt.Printf("  First example input: %v\n", inputs[0])
			fmt.Printf("  First example label: %v\n", labels[0])
		}
	}

	fmt.Println()

	// Analytics dataset: auto-discover CSV pattern and create lazy-loading dataset
	// Use default seq columns (x, y) or specify custom ones
	seqColumns := []string{"x", "y", "s", "a", "o", "dir"}
	analyticsPattern := "../assets/kaggle/anal" + "ytics/train/*.csv"
	analyticsDS, err := datasets.NewAnalyticsDataset(analyticsPattern, "play_id", seqColumns)
	if err != nil {
		// Analytics dataset is optional for some workflows
		fmt.Printf("Note: Could not load analytics dataset: %v\n", err)
		fmt.Println("Continuing without analytics data...")
	} else {
		fmt.Printf("Using analytics CSV pattern: %s\n", analyticsPattern)
		fmt.Printf("Total analytics plays available: %d\n", analyticsDS.Len())

		// Prepare a small analytics batch (first M play sequences)
		m := min(4, analyticsDS.Len())

		if m > 0 {
			aidx := make([]int, m)
			for i := range m {
				aidx[i] = i
			}

			fmt.Printf("Loading batch of %d analytics sequences...\n", m)
			buffers, shapes, err := analyticsDS.Batch(aidx)
			if err != nil {
				log.Fatalf("failed to build analytics batch: %v", err)
			}

			// Show information about loaded sequences
			fmt.Println("Loaded sequence shapes:")
			for i, shape := range shapes {
				fmt.Printf("  Sequence %d: [%d timesteps, %d channels]\n", i, shape[0], shape[1])
			}

			// Try to pack into a single contiguous tensor if all sequences share same shape
			analyticsFlat, err := datasets.MakeAnalyticsBatchFlat(buffers, shapes)
			if err != nil {
				fmt.Printf("Note: Could not pack analytics batch (sequences have different lengths): %v\n", err)
				fmt.Println("In practice, you would need to pad/truncate sequences to a fixed length.")
			} else {
				analyticsTensor, err := analyticsFlat.ToGomlxTensor()
				if err != nil {
					log.Fatalf("failed to convert analytics batch to gomlx tensor: %v", err)
				}
				fmt.Printf("Created analytics tensor: %T\n", analyticsTensor)
				fmt.Printf("  Shape: [%d batch, %d timesteps, %d channels]\n",
					analyticsFlat.Batch, analyticsFlat.Time, analyticsFlat.Channels)
			}
		}
	}

	fmt.Println("\nExample completed successfully!")
	fmt.Println("Note: Data was loaded lazily - CSV files were only read when needed for the batch.")
}
