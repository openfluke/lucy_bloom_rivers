package examples

import (
	"bufio"
	"fmt"
	"os"

	seedpoc "github.com/openfluke/loom/lucy/examples/seed_poc"
)

// RunSeedPOCMenu runs topology-only seed POC (menu [18]).
func RunSeedPOCMenu(reader *bufio.Reader) {
	fmt.Println("\n[18] Seed topology POC — recipe seeds from network shape (no weights)")
	_ = reader
	if !seedpoc.RunAll() {
		os.Exit(1)
	}
}

// RunSeedPOCAuto runs topology POC without menu (LOOM_SEED_POC=1).
func RunSeedPOCAuto() {
	if !seedpoc.RunAll() {
		os.Exit(1)
	}
}
