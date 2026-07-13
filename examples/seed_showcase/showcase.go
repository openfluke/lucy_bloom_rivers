// Package seedshowcase trains loom layers with backprop, encodes weights into infinite
// layer seed manifests, and reloads from seeds to match outputs.
package seedshowcase

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

const (
	showcaseName   = "seed-showcase-all-layers"
	showcaseFormat = "chaosglue-seed-showcase-v4"
	seedFile       = "showcase.seeds.json"
)

// ShowcaseSection is one layer-family demo (train → infinite manifest → reload).
type ShowcaseSection struct {
	Kind           string          `json:"kind"`
	Name           string          `json:"name"`
	TopologySeed   uint64          `json:"topology_seed"`
	Payload        json.RawMessage `json:"payload"`
	WeightFPs      []uint64        `json:"weight_fps"`
	TrainedOutputs []float32       `json:"trained_outputs,omitempty"`
}

// SeedShowcaseFile stores trained infinite manifests for every supported layer type.
type SeedShowcaseFile struct {
	Format   string            `json:"format"`
	Name     string            `json:"name"`
	Training string            `json:"training"`
	Sections []ShowcaseSection `json:"sections"`

	// v2 dense-only fields (reload compat)
	TopologySeed   uint64                            `json:"topology_seed,omitempty"`
	Sizes          []int                             `json:"sizes,omitempty"`
	FinalLoss      float64                           `json:"final_loss,omitempty"`
	Layers         []poly.InfiniteDenseLayerManifest `json:"layers,omitempty"`
	InitOutputs    [][]float32                       `json:"init_outputs,omitempty"`
	TrainedOutputs [][]float32                       `json:"trained_outputs_legacy,omitempty"`
}

// RunAll trains on first run and saves seeds; reruns reload from showcase.seeds.json only.
func RunAll(dir string) bool {
	if dir == "" {
		dir = "."
	}
	path := filepath.Join(dir, seedFile)

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  seed_showcase — all layers · Train · weights→seed · reload  ║")
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
	fmt.Printf("\n── RELOAD: %s → seeds+overrides → weights → forward ──\n", seedFile)
	fmt.Println("   delete file to train again")

	file, err := loadShowcase(path)
	if err != nil {
		fmt.Printf("  FAIL load: %v\n", err)
		return false
	}
	printShowcaseSummary(file, path)

	if file.Format == "chaosglue-seed-showcase-v2" {
		return reloadV2Dense(file)
	}

	ok := true
	for i, sec := range file.Sections {
		fmt.Printf("\n── [%d] %s (%s) ──\n", i+1, sec.Name, sec.Kind)
		got, err := rebuildSection(sec)
		if err != nil {
			fmt.Printf("  FAIL rebuild: %v\n", err)
			ok = false
			continue
		}
		fmt.Printf("    forward %s overrides=%d weight_fp=%v\n",
			fmtFloats(got), sectionOverrideSummary(sec), sec.WeightFPs)
		if err := verifySectionWeightReload(sec); err != nil {
			fmt.Printf("  ✗ %v\n", err)
			ok = false
			continue
		}
		fmt.Println("  ✓ weights restored from seed manifests")
	}
	if ok {
		fmt.Println("\n✓ All layer sections restored trained weights from seed manifests")
	}
	return ok
}

func reloadV2Dense(file SeedShowcaseFile) bool {
	net, err := buildNetFromShowcaseV2(file)
	if err != nil {
		fmt.Printf("  FAIL seed→weights: %v\n", err)
		return false
	}
	inputs := demoInputs(10, file.Sizes[0])
	outs := forwardAll(net, inputs)
	if len(file.TrainedOutputs) > 0 && !outputsEqual(outs, file.TrainedOutputs) {
		fmt.Println("\n✗ FAIL: v2 dense outputs differ")
		return false
	}
	fmt.Println("\n✓ v2 dense MLP reload OK")
	return true
}

func runTrainAndSave(path string) bool {
	fmt.Printf("\n── TRAIN: loom poly.Train per layer family (CPU backprop) ──\n")

	sections, err := buildAllSections()
	if err != nil {
		fmt.Printf("  FAIL build sections: %v\n", err)
		return false
	}

	file := SeedShowcaseFile{
		Format:   showcaseFormat,
		Name:     showcaseName,
		Training: "loom-poly-Train-backprop",
		Sections: sections,
	}

	for i, sec := range sections {
		fmt.Printf("\n── [%d] %s trained overrides=%d output=%s weight_fp=%v ──\n",
			i+1, sec.Kind, sectionOverrideSummary(sec), fmtFloats(sec.TrainedOutputs), sec.WeightFPs)
		if err := verifySectionWeightReload(sec); err != nil {
			fmt.Printf("  FAIL: %v\n", err)
			return false
		}
		got, err := rebuildSection(sec)
		if err != nil {
			fmt.Printf("  FAIL inline forward: %v\n", err)
			return false
		}
		_ = got
	}

	if err := saveShowcase(path, file); err != nil {
		fmt.Printf("  FAIL save: %v\n", err)
		return false
	}
	info, _ := os.Stat(path)
	fmt.Printf("\n── Saved %d sections → %s (%d bytes) ──\n", len(sections), path, info.Size())
	fmt.Printf("\n✓ First run done. Run again to reload from %s only.\n", seedFile)
	return true
}

func buildNetFromShowcaseV2(f SeedShowcaseFile) (*poly.VolumetricNetwork, error) {
	if len(f.Layers) == 0 {
		return nil, fmt.Errorf("showcase: no layers")
	}
	net := poly.NewVolumetricNetwork(1, 1, 1, len(f.Layers))
	net.InitSeed = f.TopologySeed
	for i := range f.Layers {
		lm := f.Layers[i]
		bl, err := poly.BuildDenseLayerFromInfiniteManifest(&lm)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		l := net.GetLayer(0, 0, 0, i)
		*l = *bl
	}
	return net, nil
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
	if len(data) > 32*1024*1024 {
		return SeedShowcaseFile{}, fmt.Errorf("refusing bloated seed file (%d bytes)", len(data))
	}
	var f SeedShowcaseFile
	if err := json.Unmarshal(data, &f); err != nil {
		return SeedShowcaseFile{}, err
	}
	switch f.Format {
	case showcaseFormat, "chaosglue-seed-showcase-v2", "chaosglue-seed-showcase-v1":
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
	if f.Format == "chaosglue-seed-showcase-v2" {
		fmt.Printf("  v2 dense MLP topology_seed=0x%x sizes=%v\n", f.TopologySeed, f.Sizes)
		return
	}
	for i, sec := range f.Sections {
		fmt.Printf("    [%d] %s topology_seed=0x%x overrides=%d\n",
			i+1, sec.Kind, sec.TopologySeed, sectionOverrideSummary(sec))
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
