package examples

import (
	"bufio"
	"fmt"
	"os"

	seedproof "github.com/openfluke/loom/lucy/examples/seed_proof"
	lucytesting "github.com/openfluke/loom/lucy/testing"
)

// RunSeedProofMenu runs the 3-layer seed-only proof (menu [20]).
func RunSeedProofMenu(reader *bufio.Reader) {
	fmt.Println("\n[20] Seed proof — train layer_seed · weights↔seed · reload trained net")
	_ = reader
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedproof.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}

// RunSeedProofAuto runs seed proof without menu (LOOM_SEED_PROOF=1).
func RunSeedProofAuto() {
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedproof.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}
