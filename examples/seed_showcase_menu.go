package examples

import (
	"bufio"
	"fmt"
	"os"

	seedshowcase "github.com/openfluke/loom/lucy/examples/seed_showcase"
	lucytesting "github.com/openfluke/loom/lucy/testing"
)

// RunSeedShowcaseMenu runs train→infinite manifest→reload for all layer types (menu [21]).
func RunSeedShowcaseMenu(reader *bufio.Reader) {
	fmt.Println("\n[21] Seed showcase — Train · weights→infinite manifest · reload all layers")
	_ = reader
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedshowcase.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}

// RunSeedShowcaseAuto runs seed showcase without menu (LOOM_SEED_SHOWCASE=1).
func RunSeedShowcaseAuto() {
	if err := os.MkdirAll(lucytesting.DefaultOutputDir, 0o755); err != nil {
		fmt.Printf("  FAIL mkdir %s: %v\n", lucytesting.DefaultOutputDir, err)
		os.Exit(1)
	}
	if !seedshowcase.RunAll(lucytesting.DefaultOutputDir) {
		os.Exit(1)
	}
}
