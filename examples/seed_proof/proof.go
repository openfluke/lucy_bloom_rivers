// Package seedproof — train layer_seed values; weights always He-init from seed; save/reload trained seeds.
package seedproof

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
	proofName   = "seed-proof-3layer"
	proofFormat = "chaosglue-seed-proof-v4"
	seedFile    = "proof.seeds"
)

// SeedProofFile: topology + one layer_seed per layer (trained). No weight bytes.
type SeedProofFile struct {
	Format         string           `json:"format"`
	Name           string           `json:"name"`
	TopologySeed   uint64           `json:"topology_seed"`
	Sizes          []int            `json:"sizes"`
	Layers         []SeedProofLayer `json:"layers"`
	InitOutputs    [][]float32      `json:"init_outputs"`
	TrainedOutputs [][]float32      `json:"trained_outputs"`
}

// SeedProofLayer — layer_seed expands to all weights via He-init (InitWeightStoreHeSeeded).
type SeedProofLayer struct {
	Index     int    `json:"index"`
	In        int    `json:"in"`
	Out       int    `json:"out"`
	LayerSeed uint64 `json:"layer_seed"`
	DType     string `json:"dtype"`
}

// RunAll: first run trains layer seeds; saves trained seeds. Rerun reloads trained net from seeds.
func RunAll(seedDir string) bool {
	if seedDir == "" {
		seedDir = "."
	}
	path := filepath.Join(seedDir, seedFile)

	fmt.Println("╔══════════════════════════════════════════════════════════╗")
	fmt.Println("║  Seed proof — train layer_seed · weights↔seed · reload   ║")
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
	fmt.Printf("\n── Rerun: %s — trained layer_seed → weights (no train) ──\n", seedFile)
	fmt.Println("   delete file to build + train again")

	proof, err := loadProof(path)
	if err != nil {
		fmt.Printf("  FAIL load: %v\n", err)
		return false
	}
	printProofSummary(proof, path)

	net, err := buildNetFromLayerSeeds(proof)
	if err != nil {
		fmt.Printf("  FAIL seeds→weights: %v\n", err)
		return false
	}

	inputs := demoInputs(10, proof.Sizes[0])
	fmt.Println("\n── TRAINED model rebuilt from saved layer seeds ──")
	printForwardBlock(net, inputs)

	outs := forwardAll(net, inputs)
	if len(proof.TrainedOutputs) > 0 {
		if !outputsEqual(outs, proof.TrainedOutputs) {
			fmt.Println("\n✗ FAIL: outputs differ from trained baseline in file")
			return false
		}
		fmt.Println("\n✓ Trained model restored from layer seeds (bit-exact vs saved trained outputs)")
	}
	fmt.Println("\n✓ Reload complete — weights from layer_seed He-init, not a weight file.")
	return true
}

func runFirstTime(path string) bool {
	fmt.Printf("\n── First run: no %s ──\n", seedFile)

	sizes := []int{4, 8, 4, 2}
	dtypes := []string{"float32", "float32", "float32"}
	topo := poly.DenseTopologySeed(proofName, sizes)
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

	inputs := demoInputs(10, sizes[0])
	targets := trainTargets(10, sizes[len(sizes)-1])

	fmt.Println("\n── Topology layer seeds (init recipe from topology_seed) ──")
	for i, l := range manifest.Layers {
		fmt.Printf("    layer %d %dx%d init_seed=0x%x\n", i, l.In, l.Out, l.LayerSeed)
	}

	fmt.Println("\n── BEFORE train (init seeds → He-init weights) ──")
	printForwardBlock(net, inputs)
	outsInit := forwardAll(net, inputs)
	fmt.Println("\n  10 final outputs (init):")
	printOutputs(outsInit, "    ")

	if _, err := poly.ManifestFromDenseNetwork(net, topo, sizes, dtypes); err != nil {
		fmt.Printf("\n  (pre-train extract check failed unexpectedly: %v)\n", err)
	} else {
		fmt.Println("\n  weights→seeds (init): ManifestFromDenseNetwork OK")
	}

	fmt.Println("\n── Train: optimize layer_seed (weights always He-init from seed) ──")
	lossBefore := evalMSE(net, inputs, targets)
	fmt.Printf("  loss before %.6f\n", lossBefore)
	trainLayerSeeds(net, seeds, sizes, inputs, targets)
	lossAfter := evalMSE(net, inputs, targets)
	fmt.Printf("  loss after  %.6f\n", lossAfter)
	if lossAfter >= lossBefore {
		fmt.Println("\n✗ FAIL: seed training did not reduce loss")
		return false
	}

	fmt.Println("\n── AFTER train (trained layer seeds → He-init weights) ──")
	printForwardBlock(net, inputs)
	outsTrained := forwardAll(net, inputs)
	fmt.Println("\n  10 final outputs (trained):")
	printOutputs(outsTrained, "    ")
	if outputsEqual(outsInit, outsTrained) {
		fmt.Println("\n✗ FAIL: trained outputs identical to init")
		return false
	}

	fmt.Println("\n── weights→seeds (trained weights must match their layer_seed) ──")
	for i, seed := range seeds {
		l := net.GetLayer(0, 0, 0, i)
		if !layerWeightsMatchSeed(l, sizes[i], seed) {
			fmt.Printf("  FAIL layer %d weights do not match seed 0x%x\n", i, seed)
			return false
		}
		initTopo := poly.DenseLayerWeightSeed(topo, i)
		changed := ""
		if seed != initTopo {
			changed = " (changed from init)"
		}
		fmt.Printf("    layer %d trained_seed=0x%x%s\n", i, seed, changed)
	}

	proof := proofFromSeeds(topo, sizes, seeds, dtypes, outsInit, outsTrained)
	if err := saveProof(path, proof); err != nil {
		fmt.Printf("  FAIL save: %v\n", err)
		return false
	}
	info, _ := os.Stat(path)
	fmt.Printf("\n── Saved trained layer seeds only → %s (%d bytes) ──\n", path, info.Size())
	printProofJSON(path)

	fmt.Println("\n── Reload check: file → seeds→weights → trained outputs ──")
	reloaded, err := buildNetFromLayerSeeds(proof)
	if err != nil {
		fmt.Printf("  FAIL rebuild: %v\n", err)
		return false
	}
	outsReload := forwardAll(reloaded, inputs)
	if !outputsEqual(outsTrained, outsReload) {
		fmt.Println("\n✗ FAIL: reload did not match trained outputs")
		return false
	}
	fmt.Println("✓ Saved trained seeds reproduce trained outputs")

	fmt.Printf("\n✓ First run done. Run [20] again — reload trained model from seeds only.\n")
	return true
}

func proofFromSeeds(topo uint64, sizes []int, seeds []uint64, dtypes []string, initOuts, trainedOuts [][]float32) SeedProofFile {
	p := SeedProofFile{
		Format: proofFormat, Name: proofName,
		TopologySeed:   topo,
		Sizes:          append([]int(nil), sizes...),
		InitOutputs:    cloneOutputs(initOuts),
		TrainedOutputs: cloneOutputs(trainedOuts),
	}
	for i := range seeds {
		p.Layers = append(p.Layers, SeedProofLayer{
			Index: i, In: sizes[i], Out: sizes[i+1],
			LayerSeed: seeds[i], DType: dtypes[i],
		})
	}
	return p
}

func buildNetFromLayerSeeds(p SeedProofFile) (*poly.VolumetricNetwork, error) {
	m := &poly.DenseWeightManifest{
		TopologySeed: p.TopologySeed,
		Sizes:        append([]int(nil), p.Sizes...),
	}
	for _, l := range p.Layers {
		m.Layers = append(m.Layers, poly.DenseLayerManifest{
			Index: l.Index, In: l.In, Out: l.Out,
			LayerSeed: l.LayerSeed, DType: l.DType,
		})
	}
	return poly.BuildDenseVolumetricFromManifest(m)
}

// layerWeightsMatchSeed reports whether layer weights equal He-init from seed.
func layerWeightsMatchSeed(l *poly.VolumetricLayer, in int, seed uint64) bool {
	if l == nil || l.WeightStore == nil {
		return false
	}
	tmp := poly.NewWeightStore(len(l.WeightStore.Master))
	poly.InitWeightStoreHeSeeded(tmp, in, seed)
	master := l.WeightStore.Master
	for i := range master {
		if master[i] != tmp.Master[i] {
			return false
		}
	}
	return true
}

func trainLayerSeeds(net *poly.VolumetricNetwork, seeds []uint64, sizes []int, inputs, targets [][]float32) {
	const epochs = 40
	const mutPerLayer = 200
	rng := poly.NewSeedRNG(poly.SeedFrom("seed-proof-train", seeds[0]))

	bestLoss := evalMSE(net, inputs, targets)
	for epoch := 0; epoch < epochs; epoch++ {
		for li := range seeds {
			in := sizes[li]
			for m := 0; m < mutPerLayer; m++ {
				trial := mutateSeed(seeds[li], rng.Uint64())
				applyLayerSeed(net, li, trial, in)
				loss := evalMSE(net, inputs, targets)
				if loss < bestLoss {
					bestLoss = loss
					seeds[li] = trial
				}
			}
			applyLayerSeed(net, li, seeds[li], in)
		}
	}
}

func applyLayerSeed(net *poly.VolumetricNetwork, layerIdx int, seed uint64, in int) {
	l := net.GetLayer(0, 0, 0, layerIdx)
	poly.InitWeightStoreHeSeeded(l.WeightStore, in, seed)
}

func mutateSeed(seed, noise uint64) uint64 {
	switch noise % 3 {
	case 0:
		bit := (noise / 3) % 64
		return seed ^ (1 << bit)
	case 1:
		return seed + (noise>>6) + 1
	default:
		return seed ^ noise
	}
}

func evalMSE(net *poly.VolumetricNetwork, inputs, targets [][]float32) float32 {
	var sum float32
	for i, in := range inputs {
		t := poly.NewTensorFromSlice(in, 1, len(in))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		if out == nil {
			continue
		}
		for j := range out.Data {
			d := out.Data[j] - targets[i][j]
			sum += d * d
		}
	}
	return sum / float32(len(inputs))
}

func printForwardBlock(net *poly.VolumetricNetwork, inputs [][]float32) {
	fmt.Println("\n  Chained forwards (all 10 samples):")
	for i, in := range inputs {
		fmt.Printf("    sample [%02d]\n", i+1)
		printChain(forwardChain(net, in), "      ")
	}
}

type chainStep struct {
	Layer int
	Pre   []float32
	Post  []float32
}

func forwardChain(net *poly.VolumetricNetwork, input []float32) []chainStep {
	cur := poly.NewTensorFromSlice(input, 1, len(input))
	var steps []chainStep
	for i := 0; i < net.LayersPerCell; i++ {
		layer := net.GetLayer(0, 0, 0, i)
		pre, post := poly.DenseForwardPolymorphic(layer, cur)
		steps = append(steps, chainStep{
			Layer: i,
			Pre:   append([]float32(nil), pre.Data...),
			Post:  append([]float32(nil), post.Data...),
		})
		cur = post
	}
	return steps
}

func printChain(steps []chainStep, indent string) {
	if indent == "" {
		indent = "    "
	}
	for _, s := range steps {
		fmt.Printf("%slayer %d  pre %s → post %s\n", indent, s.Layer, fmtFloats(s.Pre), fmtFloats(s.Post))
	}
}

func printProofSummary(p SeedProofFile, path string) {
	info, _ := os.Stat(path)
	size := int64(0)
	if info != nil {
		size = info.Size()
	}
	fmt.Printf("\n  file: %s (%d bytes)\n", path, size)
	fmt.Printf("  topology_seed=0x%x sizes=%v\n", p.TopologySeed, p.Sizes)
	for _, l := range p.Layers {
		fmt.Printf("    layer %d %dx%d %s trained_layer_seed=0x%x\n",
			l.Index, l.In, l.Out, l.DType, l.LayerSeed)
	}
}

func saveProof(path string, p SeedProofFile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func loadProof(path string) (SeedProofFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SeedProofFile{}, err
	}
	if len(data) > 128*1024 {
		return SeedProofFile{}, fmt.Errorf("refusing bloated seed file (%d bytes)", len(data))
	}
	var p SeedProofFile
	if err := json.Unmarshal(data, &p); err != nil {
		return SeedProofFile{}, err
	}
	if p.Format != proofFormat {
		return SeedProofFile{}, fmt.Errorf("unknown format %q — delete %s and rerun first-time", p.Format, seedFile)
	}
	return p, nil
}

func cloneOutputs(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i, row := range in {
		out[i] = append([]float32(nil), row...)
	}
	return out
}

func demoInputs(n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = 0.01 * float32((i+j)%11)
		}
		out[i] = v
	}
	return out
}

func trainTargets(n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(math.Sin(float64(i+1)*0.3 + float64(j)*0.7))
		}
		out[i] = v
	}
	return out
}

func forwardAll(net *poly.VolumetricNetwork, inputs [][]float32) [][]float32 {
	outs := make([][]float32, len(inputs))
	for i, in := range inputs {
		t := poly.NewTensorFromSlice(in, 1, len(in))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		if out == nil {
			panic("forward nil")
		}
		outs[i] = append([]float32(nil), out.Data...)
	}
	return outs
}

func printOutputs(outs [][]float32, indent string) {
	if indent == "" {
		indent = "  "
	}
	for i, o := range outs {
		fmt.Printf("%s[%02d] %s\n", indent, i+1, fmtFloats(o))
	}
}

func fmtFloats(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	const maxShow = 6
	s := "["
	for i, x := range v {
		if i >= maxShow {
			s += fmt.Sprintf(", …+%d", len(v)-maxShow)
			break
		}
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%.5f", x)
	}
	return s + "]"
}

func outputsEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func printProofJSON(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	fmt.Println(string(data))
}
