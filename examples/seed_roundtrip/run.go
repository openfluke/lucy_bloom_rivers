// Package seedroundtrip tests seed → weights → output → weights→seed for each layer/dtype.
package seedroundtrip

import (
	"fmt"
	"hash/fnv"
	"math"

	"github.com/openfluke/loom/poly"
)

const tag = "lucy-roundtrip"

// RunAll runs round-trip tests; dense first, then other layer types as they land.
func RunAll() bool {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  Seed round trip — same seeds → same weights & outputs       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	ok := true
	ok = runDense() && ok
	ok = runSwiGLU() && ok
	ok = runMHA() && ok
	ok = runPendingLayers() && ok
	return ok
}

func runDense() bool {
	fmt.Println("\n══ Dense — multi-layer · multi-dtype ══")
	sizes := []int{4, 8, 4, 2}
	dtypes := []string{"float32", "int8", "int32"}
	topo := poly.DenseTopologySeed(tag, sizes)

	manifest, err := poly.BuildDenseManifest(topo, sizes, dtypes)
	if err != nil {
		fmt.Printf("  FAIL build manifest: %v\n", err)
		return false
	}
	fmt.Printf("  topology_seed=0x%x sizes=%v\n", topo, sizes)
	for _, layer := range manifest.Layers {
		fmt.Printf("    layer %d %dx%d %s seed=0x%x weight_fp=0x%x\n",
			layer.Index, layer.In, layer.Out, layer.DType, layer.LayerSeed, layer.WeightFP)
	}
	fmt.Printf("  network_fp=0x%x forward_fp=0x%x\n", manifest.NetworkFP, manifest.ForwardFP)

	// Same seeds → same weights (rebuild check)
	rebuilt, err := poly.RebuildDenseManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL seeds→weights rebuild: %v\n", err)
		return false
	}
	seedWeightsOK := rebuilt.NetworkFP == manifest.NetworkFP && rebuilt.ForwardFP == manifest.ForwardFP
	fmt.Printf("  seeds→weights→same output: %v\n", seedWeightsOK)

	// Build volumetric net and compare forward hash
	netA, err := poly.BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL volumetric build A: %v\n", err)
		return false
	}
	netB, err := poly.BuildDenseVolumetricFromManifest(rebuilt)
	if err != nil {
		fmt.Printf("  FAIL volumetric build B: %v\n", err)
		return false
	}
	hashA := forwardHash(netA, sizes[0])
	hashB := forwardHash(netB, sizes[0])
	forwardOK := hashA == hashB
	fmt.Printf("  forward hash A=0x%x B=0x%x same=%v\n", hashA, hashB, forwardOK)

	// Weights → seeds (extract manifest from built weights)
	extracted, err := poly.ManifestFromDenseNetwork(netA, topo, sizes, dtypes)
	if err != nil {
		fmt.Printf("  FAIL weights→seeds: %v\n", err)
		return false
	}
	weightsToSeedOK := extracted.NetworkFP == manifest.NetworkFP
	fmt.Printf("  weights→seeds extract: network_fp match=%v forward_fp=0x%x\n", weightsToSeedOK, extracted.ForwardFP)
	for i := range manifest.Layers {
		match := extracted.Layers[i].LayerSeed == manifest.Layers[i].LayerSeed
		fmt.Printf("    layer %d recovered_seed=0x%x match=%v\n", i, extracted.Layers[i].LayerSeed, match)
		if !match {
			weightsToSeedOK = false
		}
	}

	// JSON manifest round trip (seeds only, tiny file)
	jsonBytes, err := poly.MarshalDenseManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL marshal: %v\n", err)
		return false
	}
	parsed, err := poly.ParseDenseManifest(jsonBytes)
	if err != nil {
		fmt.Printf("  FAIL parse: %v\n", err)
		return false
	}
	_, err = poly.RebuildDenseManifest(parsed)
	jsonOK := err == nil
	fmt.Printf("  JSON manifest (%d bytes) seeds-only round trip: %v\n", len(jsonBytes), jsonOK)

	ok := seedWeightsOK && forwardOK && weightsToSeedOK && jsonOK && extracted.ForwardFP == hashA
	if ok {
		fmt.Println("  Dense round trip OK")
	} else {
		fmt.Println("  Dense round trip FAIL")
	}
	return ok
}

func runSwiGLU() bool {
	fmt.Println("\n══ SwiGLU — multi-block · multi-dtype ══")
	specs := []poly.SwiGLUSpec{
		{Hidden: 8, Intermediate: 16},
		{Hidden: 8, Intermediate: 12},
		{Hidden: 8, Intermediate: 20},
	}
	dtypes := []string{"float32", "int8", "int32"}
	topo := poly.SwiGLUTopologySeed(tag, specs)

	manifest, err := poly.BuildSwiGLUManifest(topo, specs, dtypes)
	if err != nil {
		fmt.Printf("  FAIL build manifest: %v\n", err)
		return false
	}
	fmt.Printf("  topology_seed=0x%x specs=%v\n", topo, specs)
	for _, layer := range manifest.Layers {
		fmt.Printf("    layer %d hidden=%d inter=%d %s seed=0x%x weight_fp=0x%x\n",
			layer.Index, layer.Hidden, layer.Intermediate, layer.DType, layer.LayerSeed, layer.WeightFP)
	}
	fmt.Printf("  network_fp=0x%x forward_fp=0x%x\n", manifest.NetworkFP, manifest.ForwardFP)

	rebuilt, err := poly.RebuildSwiGLUManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL seeds→weights rebuild: %v\n", err)
		return false
	}
	seedWeightsOK := rebuilt.NetworkFP == manifest.NetworkFP && rebuilt.ForwardFP == manifest.ForwardFP
	fmt.Printf("  seeds→weights→same output: %v\n", seedWeightsOK)

	netA, err := poly.BuildSwiGLUVolumetricFromManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL volumetric build A: %v\n", err)
		return false
	}
	netB, err := poly.BuildSwiGLUVolumetricFromManifest(rebuilt)
	if err != nil {
		fmt.Printf("  FAIL volumetric build B: %v\n", err)
		return false
	}
	inputDim := specs[0].Hidden
	hashA := forwardHash(netA, inputDim)
	hashB := forwardHash(netB, inputDim)
	forwardOK := hashA == hashB
	fmt.Printf("  forward hash A=0x%x B=0x%x same=%v\n", hashA, hashB, forwardOK)

	extracted, err := poly.ManifestFromSwiGLUNetwork(netA, topo, specs, dtypes)
	if err != nil {
		fmt.Printf("  FAIL weights→seeds: %v\n", err)
		return false
	}
	weightsToSeedOK := extracted.NetworkFP == manifest.NetworkFP
	fmt.Printf("  weights→seeds extract: network_fp match=%v forward_fp=0x%x\n", weightsToSeedOK, extracted.ForwardFP)
	for i := range manifest.Layers {
		match := extracted.Layers[i].LayerSeed == manifest.Layers[i].LayerSeed
		fmt.Printf("    layer %d recovered_seed=0x%x match=%v\n", i, extracted.Layers[i].LayerSeed, match)
		if !match {
			weightsToSeedOK = false
		}
	}

	jsonBytes, err := poly.MarshalSwiGLUManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL marshal: %v\n", err)
		return false
	}
	parsed, err := poly.ParseSwiGLUManifest(jsonBytes)
	if err != nil {
		fmt.Printf("  FAIL parse: %v\n", err)
		return false
	}
	_, err = poly.RebuildSwiGLUManifest(parsed)
	jsonOK := err == nil
	fmt.Printf("  JSON manifest (%d bytes) seeds-only round trip: %v\n", len(jsonBytes), jsonOK)

	ok := seedWeightsOK && forwardOK && weightsToSeedOK && jsonOK && extracted.ForwardFP == hashA
	if ok {
		fmt.Println("  SwiGLU round trip OK")
	} else {
		fmt.Println("  SwiGLU round trip FAIL")
	}
	return ok
}

func runMHA() bool {
	fmt.Println("\n══ MHA — multi-block · multi-dtype ══")
	specs := []poly.MHASpec{
		{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8},
		{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8},
		{DModel: 8, NumHeads: 4, NumKVHeads: 2, HeadDim: 2, QueryDim: 8},
	}
	dtypes := []string{"float32", "int8", "int32"}
	topo := poly.MHATopologySeed(tag, specs)

	manifest, err := poly.BuildMHAManifest(topo, specs, dtypes)
	if err != nil {
		fmt.Printf("  FAIL build manifest: %v\n", err)
		return false
	}
	fmt.Printf("  topology_seed=0x%x specs=%v\n", topo, specs)
	for _, layer := range manifest.Layers {
		fmt.Printf("    layer %d d=%d heads=%d kv=%d %s seed=0x%x weight_fp=0x%x\n",
			layer.Index, layer.DModel, layer.NumHeads, layer.NumKVHeads, layer.DType, layer.LayerSeed, layer.WeightFP)
	}
	fmt.Printf("  network_fp=0x%x forward_fp=0x%x\n", manifest.NetworkFP, manifest.ForwardFP)

	rebuilt, err := poly.RebuildMHAManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL seeds→weights rebuild: %v\n", err)
		return false
	}
	seedWeightsOK := rebuilt.NetworkFP == manifest.NetworkFP && rebuilt.ForwardFP == manifest.ForwardFP
	fmt.Printf("  seeds→weights→same output: %v\n", seedWeightsOK)

	netA, err := poly.BuildMHAVolumetricFromManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL volumetric build A: %v\n", err)
		return false
	}
	netB, err := poly.BuildMHAVolumetricFromManifest(rebuilt)
	if err != nil {
		fmt.Printf("  FAIL volumetric build B: %v\n", err)
		return false
	}
	inputDim := specs[0].DModel
	hashA := forwardHash(netA, inputDim)
	hashB := forwardHash(netB, inputDim)
	forwardOK := hashA == hashB
	fmt.Printf("  forward hash A=0x%x B=0x%x same=%v\n", hashA, hashB, forwardOK)

	extracted, err := poly.ManifestFromMHANetwork(netA, topo, specs, dtypes)
	if err != nil {
		fmt.Printf("  FAIL weights→seeds: %v\n", err)
		return false
	}
	weightsToSeedOK := extracted.NetworkFP == manifest.NetworkFP
	fmt.Printf("  weights→seeds extract: network_fp match=%v forward_fp=0x%x\n", weightsToSeedOK, extracted.ForwardFP)
	for i := range manifest.Layers {
		match := extracted.Layers[i].LayerSeed == manifest.Layers[i].LayerSeed
		fmt.Printf("    layer %d recovered_seed=0x%x match=%v\n", i, extracted.Layers[i].LayerSeed, match)
		if !match {
			weightsToSeedOK = false
		}
	}

	jsonBytes, err := poly.MarshalMHAManifest(manifest)
	if err != nil {
		fmt.Printf("  FAIL marshal: %v\n", err)
		return false
	}
	parsed, err := poly.ParseMHAManifest(jsonBytes)
	if err != nil {
		fmt.Printf("  FAIL parse: %v\n", err)
		return false
	}
	_, err = poly.RebuildMHAManifest(parsed)
	jsonOK := err == nil
	fmt.Printf("  JSON manifest (%d bytes) seeds-only round trip: %v\n", len(jsonBytes), jsonOK)

	ok := seedWeightsOK && forwardOK && weightsToSeedOK && jsonOK && extracted.ForwardFP == hashA
	if ok {
		fmt.Println("  MHA round trip OK")
	} else {
		fmt.Println("  MHA round trip FAIL")
	}
	return ok
}

func runPendingLayers() bool {
	fmt.Println("\n══ Other layers / dtypes (coming next) ══")
	pending := []string{"RMSNorm", "RNN", "LSTM", "CNN", "Embedding", "21 dtypes"}
	for _, name := range pending {
		fmt.Printf("  [ ] %s round trip\n", name)
	}
	fmt.Println("  (dense is the template — plug each layer into seedroundtrip)")
	return true
}

func forwardHash(net *poly.VolumetricNetwork, inputDim int) uint64 {
	in := poly.NewTensorFromSlice(demoInput(inputDim), 1, inputDim)
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0
	}
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range out.Data {
		bits := math.Float32bits(v)
		buf[0] = byte(bits)
		buf[1] = byte(bits >> 8)
		buf[2] = byte(bits >> 16)
		buf[3] = byte(bits >> 24)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func demoInput(n int) []float32 {
	in := make([]float32, n)
	for i := range in {
		in[i] = 0.01 * float32(i%11)
	}
	return in
}
