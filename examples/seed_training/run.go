// Package seedtraining — hill-climb layer_seed on real UCI wine data; seeds-only save/reload.
package seedtraining

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

const (
	runName     = "wine-quality-regression"
	runFormat   = "chaosglue-seed-training-v1"
	seedFile    = "seed_training.seeds.json"
	featureDesc = "alcohol, volatile acidity, citric acid, pH (normalized)"
)

// SeedTrainingFile stores trained layer_seed values only — weights expand via He-init.
type SeedTrainingFile struct {
	Format       string              `json:"format"`
	Name         string              `json:"name"`
	Dataset      string              `json:"dataset"`
	Features     string              `json:"features"`
	TrainSamples int                 `json:"train_samples"`
	ValSamples   int                 `json:"val_samples"`
	TopologySeed uint64              `json:"topology_seed"`
	Sizes        []int               `json:"sizes"`
	Layers       []SeedTrainingLayer `json:"layers"`
	TrainMSE     float32             `json:"train_mse"`
	ValMSE       float32             `json:"val_mse"`
	TrainMAE     float32             `json:"train_mae"`
	ValMAE       float32             `json:"val_mae"`
}

// SeedTrainingLayer — one layer_seed per dense layer.
type SeedTrainingLayer struct {
	Index     int    `json:"index"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
}

// RunAll trains layer seeds on wine data (first run) or reloads from saved seeds.
func RunAll(seedDir string) bool {
	if seedDir == "" {
		seedDir = "."
	}
	path := filepath.Join(seedDir, seedFile)

	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Seed training — real data · layer_seed hill-climb       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════╝")

	if _, err := os.Stat(path); err == nil {
		return runReloadOnly(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		fmt.Printf("  FAIL stat %s: %v\n", path, err)
		return false
	}
	return runFirstTime(path)
}

func runReloadOnly(path string) bool {
	fmt.Printf("\n── Rerun: %s — trained layer_seed → He-init weights ──\n", seedFile)
	fmt.Println("   delete file to train again on wine data")

	file, err := loadFile(path)
	if err != nil {
		fmt.Printf("  FAIL load: %v\n", err)
		return false
	}
	printFileSummary(file, path)

	trainX, trainY, valX, valY, err := loadSplits()
	if err != nil {
		fmt.Printf("  FAIL dataset: %v\n", err)
		return false
	}

	net, err := buildNetFromFile(file)
	if err != nil {
		fmt.Printf("  FAIL seeds→weights: %v\n", err)
		return false
	}

	trainMSE := evalMSE(net, trainX, trainY)
	valMSE := evalMSE(net, valX, valY)
	trainMAE := evalMAE(net, trainX, trainY)
	valMAE := evalMAE(net, valX, valY)

	fmt.Printf("\n  metrics (recomputed): train MSE=%.5f MAE=%.5f | val MSE=%.5f MAE=%.5f\n",
		trainMSE, trainMAE, valMSE, valMAE)
	if math.Abs(float64(trainMSE-file.TrainMSE)) > 1e-4 || math.Abs(float64(valMSE-file.ValMSE)) > 1e-4 {
		fmt.Println("\n✗ FAIL: metrics differ from saved baseline")
		return false
	}

	printDeviationHeatmap(net, valX, valY, "validation (reloaded trained seeds)")
	printDeviationHeatmap(net, trainX, trainY, "train (reloaded trained seeds)")
	printPredictionTable(net, valX, valY, 8)
	fmt.Println("\n✓ Reload complete — weights from layer_seed He-init, not a weight file.")
	return true
}

func runFirstTime(path string) bool {
	fmt.Printf("\n── First run: no %s ──\n", seedFile)

	all, err := loadWineDataset()
	if err != nil {
		fmt.Printf("  FAIL wine data: %v\n", err)
		return false
	}
	trainS, valS := splitDataset(all, 0.8)
	trainX, trainY := samplesToBatches(trainS)
	valX, valY := samplesToBatches(valS)

	fmt.Printf("\n  REAL DATA: UCI Wine Quality (red) — %d rows total\n", len(all))
	fmt.Printf("  features: %s\n", featureDesc)
	fmt.Printf("  target: quality score / 10 (regression)\n")
	fmt.Printf("  split: %d train · %d validation\n", len(trainS), len(valS))

	sizes := []int{4, 16, 8, 1}
	dtypes := []string{"float32", "float32", "float32"}
	topo := poly.DenseTopologySeed(runName, sizes)
	fmt.Printf("\n3-layer dense sizes=%v topology_seed=0x%x\n", sizes, topo)

	manifest, err := poly.BuildDenseManifest(topo, sizes, dtypes)
	if err != nil {
		fmt.Printf("  FAIL BuildDenseManifest: %v\n", err)
		return false
	}
	net, err := poly.BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL build: %v\n", err)
		return false
	}

	seeds := make([]uint64, len(manifest.Layers))
	for i, l := range manifest.Layers {
		seeds[i] = l.LayerSeed
	}

	fmt.Println("\n── Init layer seeds (topology recipe) ──")
	for i, l := range manifest.Layers {
		fmt.Printf("    layer %d %dx%d init_seed=0x%x\n", i, l.In, l.Out, l.LayerSeed)
	}

	initSeeds := append([]uint64(nil), seeds...)

	lossBefore := evalMSE(net, trainX, trainY)
	valBefore := evalMSE(net, valX, valY)
	fmt.Printf("\n── BEFORE seed training (He-init from init seeds) ──\n")
	fmt.Printf("  train MSE=%.5f  val MSE=%.5f\n", lossBefore, valBefore)
	printDeviationHeatmap(net, valX, valY, "validation BEFORE")
	printDeviationHeatmap(net, trainX, trainY, "train BEFORE")

	fmt.Println("\n── Train: mutate layer_seed only (weights always He-init from seed) ──")
	fmt.Println("  not poly.Train backprop · not brute-force weight inversion")
	trainLayerSeeds(net, seeds, sizes, trainX, trainY)

	lossAfter := evalMSE(net, trainX, trainY)
	valAfter := evalMSE(net, valX, valY)
	trainMAE := evalMAE(net, trainX, trainY)
	valMAE := evalMAE(net, valX, valY)
	fmt.Printf("  train MSE=%.5f → %.5f  val MSE=%.5f → %.5f\n", lossBefore, lossAfter, valBefore, valAfter)
	fmt.Printf("  train MAE=%.5f  val MAE=%.5f\n", trainMAE, valMAE)

	printDeviationHeatmap(net, valX, valY, "validation AFTER")
	printDeviationHeatmap(net, trainX, trainY, "train AFTER")

	initNet, err := rebuildNet(topo, sizes, dtypes, initSeeds)
	if err != nil {
		fmt.Printf("  FAIL rebuild init net: %v\n", err)
		return false
	}
	printBeforeAfterComparison(initNet, net, valX, valY, "validation")
	printBeforeAfterComparison(initNet, net, trainX, trainY, "train")

	if lossAfter >= lossBefore {
		fmt.Println("\n✗ FAIL: seed training did not reduce train loss")
		return false
	}

	fmt.Println("\n── Trained seeds (changed from init where hill-climb found improvement) ──")
	for i, seed := range seeds {
		initTopo := poly.DenseLayerWeightSeed(topo, i)
		changed := ""
		if seed != initTopo {
			changed = " *"
		}
		if !layerWeightsMatchSeed(net.GetLayer(0, 0, 0, i), sizes[i], seed) {
			fmt.Printf("  FAIL layer %d weights do not match seed 0x%x\n", i, seed)
			return false
		}
		fmt.Printf("    layer %d trained_seed=0x%x%s\n", i, seed, changed)
	}

	printPredictionTable(net, valX, valY, 10)

	layers := make([]SeedTrainingLayer, len(seeds))
	for i, l := range manifest.Layers {
		layers[i] = SeedTrainingLayer{
			Index:     i,
			In:        l.In,
			Out:       l.Out,
			LayerSeed: seeds[i],
			DType:     l.DType,
		}
	}
	file := SeedTrainingFile{
		Format:       runFormat,
		Name:         runName,
		Dataset:      "UCI Wine Quality (red)",
		Features:     featureDesc,
		TrainSamples: len(trainS),
		ValSamples:   len(valS),
		TopologySeed: topo,
		Sizes:        sizes,
		Layers:       layers,
		TrainMSE:     lossAfter,
		ValMSE:       valAfter,
		TrainMAE:     trainMAE,
		ValMAE:       valMAE,
	}
	if err := saveFile(path, file); err != nil {
		fmt.Printf("  FAIL save: %v\n", err)
		return false
	}
	fmt.Printf("\n✓ Saved seeds-only %s (%d bytes)\n", path, fileSize(path))
	fmt.Println("  rerun to reload trained net from layer_seed → He-init weights")
	return true
}

func loadSplits() (trainX, trainY, valX, valY [][]float32, err error) {
	all, err := loadWineDataset()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	trainS, valS := splitDataset(all, 0.8)
	trainX, trainY = samplesToBatches(trainS)
	valX, valY = samplesToBatches(valS)
	return trainX, trainY, valX, valY, nil
}

func buildNetFromFile(f SeedTrainingFile) (*poly.VolumetricNetwork, error) {
	dtypes := make([]string, len(f.Layers))
	for i, l := range f.Layers {
		dtypes[i] = l.DType
	}
	manifest, err := poly.BuildDenseManifest(f.TopologySeed, f.Sizes, dtypes)
	if err != nil {
		return nil, err
	}
	for i, l := range f.Layers {
		manifest.Layers[i].LayerSeed = l.LayerSeed
	}
	return poly.BuildDenseVolumetricFromManifest(manifest)
}

func printFileSummary(f SeedTrainingFile, path string) {
	fmt.Printf("\n  file: %s (%d bytes)\n", path, fileSize(path))
	fmt.Printf("  dataset: %s (%d train / %d val)\n", f.Dataset, f.TrainSamples, f.ValSamples)
	fmt.Printf("  topology_seed=0x%x sizes=%v\n", f.TopologySeed, f.Sizes)
	for _, l := range f.Layers {
		fmt.Printf("    layer %d %dx%d %s trained_layer_seed=0x%x\n",
			l.Index, l.In, l.Out, l.DType, l.LayerSeed)
	}
	fmt.Printf("  saved metrics: train MSE=%.5f val MSE=%.5f\n", f.TrainMSE, f.ValMSE)
}

func printPredictionTable(net *poly.VolumetricNetwork, inputs, targets [][]float32, maxRows int) {
	fmt.Println("\n  sample predictions (quality/10):")
	if maxRows > len(inputs) {
		maxRows = len(inputs)
	}
	for i := 0; i < maxRows; i++ {
		t := poly.NewTensorFromSlice(inputs[i], 1, len(inputs[i]))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		pred := float32(0)
		if out != nil && len(out.Data) > 0 {
			pred = out.Data[0]
		}
		actual := targets[i][0]
		fmt.Printf("    [%02d] actual=%.2f pred=%.2f err=%+.3f\n", i+1, actual, pred, pred-actual)
	}
	if len(inputs) > maxRows {
		fmt.Printf("    … +%d more validation rows\n", len(inputs)-maxRows)
	}
}

func saveFile(path string, f SeedTrainingFile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadFile(path string) (SeedTrainingFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SeedTrainingFile{}, err
	}
	if len(data) > 64*1024 {
		return SeedTrainingFile{}, fmt.Errorf("refusing bloated seed file (%d bytes)", len(data))
	}
	var f SeedTrainingFile
	if err := json.Unmarshal(data, &f); err != nil {
		return SeedTrainingFile{}, err
	}
	if f.Format != runFormat {
		return SeedTrainingFile{}, fmt.Errorf("unknown format %q — delete %s and rerun", f.Format, seedFile)
	}
	return f, nil
}

func fileSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
