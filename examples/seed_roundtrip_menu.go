package examples

import (
	"bufio"
	"fmt"
	"os"

	seedroundtrip "github.com/openfluke/loom/lucy/examples/seed_roundtrip"
)

// RunSeedRoundTripMenu runs seed round-trip tests (menu [19]).
func RunSeedRoundTripMenu(reader *bufio.Reader) {
	fmt.Println("\n[19] Seed round trip — dense first · weights↔seeds · all layers later")
	_ = reader
	if !seedroundtrip.RunAll() {
		os.Exit(1)
	}
}

// RunSeedRoundTripAuto runs round trip without menu (LOOM_SEED_ROUNDTRIP=1).
func RunSeedRoundTripAuto() {
	if !seedroundtrip.RunAll() {
		os.Exit(1)
	}
}
