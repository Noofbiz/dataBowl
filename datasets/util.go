package datasets

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func parseFloat32(s string) (float32, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty string")
	}
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return 0, err
	}
	return float32(v), nil
}

// countCSVRows counts the number of data rows in a CSV file (excluding header)
func countCSVRows(path string) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return 0, err
	}

	count := 0
	for {
		_, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return 0, err
		}
		count++
	}

	return count, nil
}

// Auto-discovery helpers

func autoFindCSV(patterns []string) (string, error) {
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err == nil && len(matches) > 0 {
			return pattern, nil
		}
	}
	return "", fmt.Errorf("no CSV files found in common locations")
}

// FindCSVInAssets finds CSV files in a specified directory
func FindCSVInAssets(dir string) (string, error) {
	pattern := filepath.Join(dir, "*.csv")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no CSV files found in %s", dir)
	}
	return matches[0], nil
}
