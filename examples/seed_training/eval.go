package seedtraining

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

func evalInputs(inputs [][]float32) []*poly.Tensor[float32] {
	out := make([]*poly.Tensor[float32], len(inputs))
	for i, in := range inputs {
		out[i] = poly.NewTensorFromSlice(in, 1, len(in))
	}
	return out
}

func evalExpected(targets [][]float32) []float64 {
	out := make([]float64, len(targets))
	for i, t := range targets {
		out[i] = float64(t[0])
	}
	return out
}

func evaluateNet(net *poly.VolumetricNetwork, inputs, targets [][]float32) (*poly.DeviationMetrics, error) {
	return poly.EvaluateNetworkPolymorphic(net, evalInputs(inputs), evalExpected(targets))
}

func printDeviationHeatmap(net *poly.VolumetricNetwork, inputs, targets [][]float32, title string) {
	m, err := evaluateNet(net, inputs, targets)
	if err != nil {
		fmt.Printf("  FAIL evaluation %s: %v\n", title, err)
		return
	}
	fmt.Printf("\n── %s (poly.DeviationMetrics 0–100 quality + error buckets) ──\n", title)
	m.PrintSummary()
	printSampleScoreStrip(m)
}

func printBeforeAfterComparison(initNet, trainedNet *poly.VolumetricNetwork, inputs, targets [][]float32, split string) {
	tensors := evalInputs(inputs)
	expected := evalExpected(targets)
	results, err := poly.MultiNetworkEvaluation(map[string]*poly.VolumetricNetwork{
		"init seeds":     initNet,
		"trained seeds":  trainedNet,
	}, tensors, expected)
	if err != nil {
		fmt.Printf("  FAIL before/after comparison: %v\n", err)
		return
	}
	fmt.Printf("\n── %s — init vs trained (0–100 quality score) ──\n", split)
	poly.PrintMultiNetworkSummary(results)
}

// printSampleScoreStrip — one block per sample, height ∝ quality (0–100).
func printSampleScoreStrip(m *poly.DeviationMetrics) {
	if m == nil || len(m.Results) == 0 {
		return
	}
	fmt.Printf("\n  per-sample quality (0–100): ")
	for _, r := range m.Results {
		q := math.Max(0, 100-r.Deviation)
		ch := sampleQualityChar(q)
		fmt.Printf("%c", ch)
	}
	fmt.Printf("  (%d samples)\n", len(m.Results))
	fmt.Println("  legend: ░ poor · ▒ fair · ▓ good · █ excellent")
}

func sampleQualityChar(score float64) rune {
	switch {
	case score >= 90:
		return '█'
	case score >= 70:
		return '▓'
	case score >= 50:
		return '▒'
	default:
		return '░'
	}
}

func rebuildNet(topo uint64, sizes []int, dtypes []string, seeds []uint64) (*poly.VolumetricNetwork, error) {
	manifest, err := poly.BuildDenseManifest(topo, sizes, dtypes)
	if err != nil {
		return nil, err
	}
	for i, s := range seeds {
		manifest.Layers[i].LayerSeed = s
	}
	return poly.BuildDenseVolumetricFromManifest(manifest)
}
