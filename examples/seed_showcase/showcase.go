// Package showcase trains layer_seed values for every loom layer family (seed mutation,
// not backprop). Weights are always He-init from seed — no weight bytes in the file.
package seedshowcase

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

const (
	showcaseName   = "seed-showcase-all-layers"
	showcaseFormat = "chaosglue-seed-showcase-v5"
	seedFile       = "showcase.seeds.json"
)

// ShowcaseSection is one layer-family demo (train layer_seed → save → reload).
type ShowcaseSection struct {
	Kind           string          `json:"kind"`
	Name           string          `json:"name"`
	TopologySeed   uint64          `json:"topology_seed"`
	Payload        json.RawMessage `json:"payload"`
	WeightFPs      []uint64        `json:"weight_fps"`
	TrainedOutputs []float32       `json:"trained_outputs,omitempty"`
}

// SeedShowcaseFile stores trained seeds-only manifests for every supported layer type.
type SeedShowcaseFile struct {
	Format   string            `json:"format"`
	Name     string            `json:"name"`
	Training string            `json:"training"`
	Sections []ShowcaseSection `json:"sections"`
}

// RunAll trains layer seeds on first run; reruns reload from showcase.seeds.json only.
func RunAll(dir string) bool {
	if dir == "" {
		dir = "."
	}
	path := filepath.Join(dir, seedFile)

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  seed_showcase — all layers · train layer_seed · reload      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	if _, err := os.Stat(path); err == nil {
		return runReload(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		fmt.Printf("  FAIL stat %s: %v\n", path, err)
		return false
	}
	return runTrainAndSave(path)
}

func runReload(path string) bool {
	fmt.Printf("\n── RELOAD: %s → layer_seed → He-init weights → forward ──\n", seedFile)
	fmt.Println("   delete file to train again")

	file, err := loadShowcase(path)
	if err != nil {
		fmt.Printf("  FAIL load: %v\n", err)
		return false
	}
	printShowcaseSummary(file, path)

	ok := true
	for i, sec := range file.Sections {
		fmt.Printf("\n── [%d] %s (%s) ──\n", i+1, sec.Name, sec.Kind)
		got, err := rebuildSection(sec)
		if err != nil {
			fmt.Printf("  FAIL rebuild: %v\n", err)
			ok = false
			continue
		}
		fmt.Printf("    forward %s layer_seeds=%d weight_fp=%v\n",
			fmtFloats(got), sectionLayerSeedCount(sec), sec.WeightFPs)
		if err := verifySectionSeedReload(sec); err != nil {
			fmt.Printf("  ✗ %v\n", err)
			ok = false
			continue
		}
		fmt.Println("  ✓ weights from layer_seed only (no override blobs)")
	}
	if ok {
		fmt.Println("\n✓ All sections restored from seeds-only manifests")
	}
	return ok
}

func runTrainAndSave(path string) bool {
	fmt.Printf("\n── TRAIN: mutate layer_seed per layer family (weights always He-init) ──\n")

	sections, err := buildAllSections()
	if err != nil {
		fmt.Printf("  FAIL build sections: %v\n", err)
		return false
	}

	file := SeedShowcaseFile{
		Format:   showcaseFormat,
		Name:     showcaseName,
		Training: "layer-seed-mutation",
		Sections: sections,
	}

	for i, sec := range sections {
		fmt.Printf("\n── [%d] %s layer_seeds=%d output=%s weight_fp=%v ──\n",
			i+1, sec.Kind, sectionLayerSeedCount(sec), fmtFloats(sec.TrainedOutputs), sec.WeightFPs)
		if err := verifySectionSeedReload(sec); err != nil {
			fmt.Printf("  FAIL: %v\n", err)
			return false
		}
	}

	if err := saveShowcase(path, file); err != nil {
		fmt.Printf("  FAIL save: %v\n", err)
		return false
	}
	info, _ := os.Stat(path)
	fmt.Printf("\n── Saved %d sections → %s (%d bytes, seeds only) ──\n", len(sections), path, info.Size())
	fmt.Printf("\n✓ First run done. Run again to reload from %s only.\n", seedFile)
	return true
}

func saveShowcase(path string, f SeedShowcaseFile) error {
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadShowcase(path string) (SeedShowcaseFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SeedShowcaseFile{}, err
	}
	if len(data) > 8*1024*1024 {
		return SeedShowcaseFile{}, fmt.Errorf("refusing bloated seed file (%d bytes)", len(data))
	}
	var f SeedShowcaseFile
	if err := json.Unmarshal(data, &f); err != nil {
		return SeedShowcaseFile{}, err
	}
	switch f.Format {
	case showcaseFormat:
	case "chaosglue-seed-showcase-v4", "chaosglue-seed-showcase-v2", "chaosglue-seed-showcase-v1":
		return SeedShowcaseFile{}, fmt.Errorf("format %q stores compressed weight overrides — delete %s and retrain for v5 seeds-only", f.Format, seedFile)
	default:
		return SeedShowcaseFile{}, fmt.Errorf("unknown format %q — delete %s and retrain", f.Format, seedFile)
	}
	return f, nil
}

func printShowcaseSummary(f SeedShowcaseFile, path string) {
	info, _ := os.Stat(path)
	size := int64(0)
	if info != nil {
		size = info.Size()
	}
	fmt.Printf("\n  file: %s (%d bytes)\n", path, size)
	fmt.Printf("  training=%q format=%s\n", f.Training, f.Format)
	for i, sec := range f.Sections {
		fmt.Printf("    [%d] %s topology_seed=0x%x layer_seeds=%d\n",
			i+1, sec.Kind, sec.TopologySeed, sectionLayerSeedCount(sec))
	}
}

func sliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}