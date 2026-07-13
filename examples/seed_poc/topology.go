// Package seedpoc demonstrates topology-only seeds (no weight blobs).
package seedpoc

import (
	"fmt"

	"github.com/openfluke/loom/poly"
)

const tag = "lucy-bloom-rivers"

// RunAll runs the seed topology POC.
func RunAll() bool {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Seed topology POC — recipe seeds from shape only            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	ok := true
	ok = runDenseTopology() && ok
	ok = runEntityTransformerTopology() && ok
	ok = runEntityWireSeed() && ok
	return ok
}

func runDenseTopology() bool {
	fmt.Println("\n── dense MLP topology ──")
	sizes := []int{4, 8, 4, 2}
	topo := poly.DenseTopologySeed(tag, sizes)
	fmt.Printf("  sizes=%v → topology_seed=0x%x\n", sizes, topo)
	for i := 0; i < len(sizes)-1; i++ {
		ls := poly.DenseLayerWeightSeed(topo, i)
		fmt.Printf("  layer %d (%d→%d) layer_seed=0x%x\n", i, sizes[i], sizes[i+1], ls)
	}
	again := poly.DenseTopologySeed(tag, sizes)
	if again != topo {
		fmt.Printf("  FAIL topology seed unstable: 0x%x vs 0x%x\n", topo, again)
		return false
	}
	fmt.Println("  OK same topology → same seeds")
	return true
}

func runEntityTransformerTopology() bool {
	fmt.Println("\n── entity transformer topology (Llama-style block) ──")
	dims := poly.HFDecoderDims{
		NumLayers: 1, HiddenSize: 16, NumHeads: 4, NumKVHeads: 4, HeadDim: 4,
		QueryDim: 16, KVDim: 16, IntermediateSize: 32,
		RMSNormEps: 1e-5, RoPEFreqBase: 10000, Activation: poly.ActivationSilu,
	}
	modelID := "demo/tiny-lm"
	recipe := poly.EntityTransformerRecipeSeed(modelID, poly.HFArchLlamaStyleDecoder, dims, poly.DTypeFloat32)
	fmt.Printf("  model=%s hidden=%d layers=%d vocab=64\n", modelID, dims.HiddenSize, dims.NumLayers)
	fmt.Printf("  recipe_seed=0x%x\n", recipe)
	again := poly.EntityTransformerRecipeSeed(modelID, poly.HFArchLlamaStyleDecoder, dims, poly.DTypeFloat32)
	if again != recipe {
		fmt.Println("  FAIL recipe seed unstable")
		return false
	}
	fmt.Println("  OK same dims → same recipe seed")
	return true
}

func runEntityWireSeed() bool {
	fmt.Println("\n── .entity wire stores init_seed in header ──")
	seed := poly.SeedFrom(tag, "wire", "tiny")
	dims := poly.HFDecoderDims{
		NumLayers: 1, HiddenSize: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4,
		QueryDim: 8, KVDim: 8, IntermediateSize: 16, Activation: poly.ActivationSilu,
	}
	et := poly.BuildSeededEntityTransformer(seed, dims, 32, poly.DTypeFloat32, true, true)
	if et == nil {
		fmt.Println("  FAIL build")
		return false
	}
	wire, err := poly.SerializeEntityTransformer(et)
	if err != nil {
		fmt.Printf("  FAIL serialize: %v\n", err)
		return false
	}
	got, err := poly.ExtractEntitySeed(wire)
	if err != nil {
		fmt.Printf("  FAIL extract: %v\n", err)
		return false
	}
	match := got == seed
	fmt.Printf("  init_seed=0x%x extract=0x%x match=%v\n", seed, got, match)
	return match
}
