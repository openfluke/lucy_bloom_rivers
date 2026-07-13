package examples

import (
	"bufio"
	"fmt"
	"os"

	seedtraining "github.com/openfluke/loom/lucy/examples/seed_training"
	lucytesting "github.com/openfluke/loom/lucy/testing"
)

// RunSeedTrainingMenu runs seed training on real wine data (menu [22]).
func RunSeedTrainingMenu(reader *bufio.Reader) {
	fmt.Println("\n[22] Seed training — real UCI wine data · layer_seed hill-climb · seeds-only reload")
	_ = reader
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedtraining.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}

// RunSeedTrainingAuto runs seed training without menu (LOOM_SEED_TRAINING=1).
func RunSeedTrainingAuto() {
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedtraining.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}
