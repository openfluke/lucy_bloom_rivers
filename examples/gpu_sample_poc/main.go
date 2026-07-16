// gpu_sample_poc compares SmolLM2-135M WebGPU decode with:
//   A) baseline — map full logits to CPU, SampleTopK on host each step
//   B) GPUSampleGreedy — ArgMax on GPU, map back only a 4-byte token id
//
// Run from this directory (needs lucy_entities + HF tokenizer snapshot):
//
//	go run .
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

const (
	repoID      = "HuggingFaceTB/SmolLM2-135M-Instruct"
	benchTokens = 64
	fixedPrompt = "Explain in one short paragraph why the sky appears blue during the day."
)

func main() {
	entity := entityPath(repoID)
	if _, err := os.Stat(entity); err != nil {
		fmt.Printf("❌ missing entity: %s\n", entity)
		os.Exit(1)
	}
	snap, err := resolveSnapshot(repoID)
	if err != nil {
		fmt.Printf("❌ tokenizer snapshot: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" GPU sample PoC — SmolLM2-135M")
	fmt.Println(" A) baseline: logits GPU→CPU every decode step")
	fmt.Println(" B) greedy:   ArgMax on GPU, map 4-byte token id only")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf("entity: %s\n", entity)

	tr, tk, ef, eos, err := loadGPU(entity, snap)
	if err != nil {
		fmt.Printf("❌ load: %v\n", err)
		os.Exit(1)
	}
	defer ef.Close()

	encode := func(text string) []uint32 { return tk.Encode(text, true) }
	decode := func(tokens []uint32) string { return tk.Decode(tokens, true) }
	sys := "You are a helpful assistant. Be concise, coherent, and avoid repetition."

	baseOpts := poly.GenOptions{
		MaxTokens:     benchTokens,
		MinTokens:     benchTokens,
		Temperature:   0,
		TopK:          1,
		Deterministic: true,
		EOSTokens:     eos,
		Silent:        true,
	}

	// Warmup (untimed)
	fmt.Println("\n⏳ warmup …")
	warm := baseOpts
	warm.MaxTokens = 4
	warm.MinTokens = 4
	_, _ = tr.Generate(encode, decode, nil, sys, fixedPrompt, warm)
	tr.Reset()

	fmt.Println("\n── A) baseline (full logits readback) ──")
	tr.Reset()
	replyA, mA := tr.Generate(encode, decode, nil, sys, fixedPrompt, baseOpts)
	printMetrics("baseline", mA, replyA)

	fmt.Println("\n── B) GPUSampleGreedy + fused pass + chunked decode ──")
	tr.Reset()
	optsB := baseOpts
	optsB.GPUSampleGreedy = true
	t0 := time.Now()
	replyB, mB := tr.Generate(encode, decode, nil, sys, fixedPrompt, optsB)
	_ = t0
	printMetrics("gpu_chunk", mB, replyB)

	fmt.Println("\n── delta ──")
	if mA.DecodeTokPerSec > 0 && mB.DecodeTokPerSec > 0 {
		fmt.Printf("decode speedup B/A:     %.2f×\n", mB.DecodeTokPerSec/mA.DecodeTokPerSec)
	}
	if mA.PrefillTokPerSec > 0 && mB.PrefillTokPerSec > 0 {
		fmt.Printf("prefill (TTFT) B/A:     %.2f×\n", mB.PrefillTokPerSec/mA.PrefillTokPerSec)
	}
	fmt.Printf("readback A: ~%d bytes/step (logits) | B: chunked hist MapAsync\n", tr.VocabSize*4)
	switch {
	case mB.DecodeTokPerSec >= 90:
		fmt.Println("\nVerdict: chunked fused GPU decode is in the usable / Ollama-class band.")
	case mB.DecodeTokPerSec >= 50:
		fmt.Println("\nVerdict: clear win vs legacy ~25 tok/s Loom GPU — keep fast path (decode GEMV + Q4 LM + chunk).")
	default:
		fmt.Println("\nVerdict: still short of target — check chunk path / adapter.")
	}
}

func printMetrics(label string, m poly.GenMetrics, reply string) {
	fmt.Printf("[%s]\n", label)
	fmt.Printf("  prefill: %d tok @ %.1f tok/s  (%v)\n", m.PrefillTokens, m.PrefillTokPerSec, m.PrefillTime.Round(time.Millisecond))
	fmt.Printf("  decode:  %d tok @ %.1f tok/s  (%v)\n", m.GeneratedTokens, m.DecodeTokPerSec, m.DecodeTime.Round(time.Millisecond))
	fmt.Printf("  total:   %.1f tok/s\n", m.TotalTokPerSec)
	fmt.Printf("  reply:   %q\n", truncate(strings.TrimSpace(reply), 160))
}

func loadGPU(entity, snap string) (*poly.Transformer[float32], *poly.Tokenizer, *poly.EntityFile, []int, error) {
	tk, err := poly.LoadTokenizer(filepath.Join(snap, "tokenizer.json"))
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("tokenizer: %w", err)
	}
	ef, err := poly.OpenEntityFile(entity)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	et, err := ef.LoadEntityTransformerTopology()
	if err != nil {
		ef.Close()
		return nil, nil, nil, nil, err
	}
	poly.PrepareEntityTransformerInference(et)
	template := poly.TemplateForHFModelID(repoID)
	tr := poly.BuildTransformerFromEntity[float32](et, template)
	tr.SetRMSNormEps(et.Dims.RMSNormEps)
	numLayers := et.Dims.NumLayers
	if numLayers <= 0 && et.Network != nil {
		numLayers = len(et.Network.Layers) / 4
	}
	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = 512
	}

	dtype := et.WeightDType
	if dtype == 0 {
		dtype = poly.DTypeFloat32
	}

	if err := tr.Network.InitWGPU(); err != nil {
		ef.Close()
		return nil, nil, nil, nil, fmt.Errorf("InitWGPU: %w", err)
	}
	tr.Network.UseGPU = true

	for li := 0; li < numLayers; li++ {
		base := li * 4
		indices := []int{base, base + 1, base + 2, base + 3}
		if err := ef.LoadNetworkLayerWeights(tr.Network, indices); err != nil {
			ef.Close()
			return nil, nil, nil, nil, fmt.Errorf("load block %d: %w", li, err)
		}
		poly.PrepareEntityTransformerLayerIndices(et, indices)
		for j := 0; j < 4; j++ {
			layer := &tr.Network.Layers[base+j]
			if layer.Type == poly.LayerRMSNorm {
				layer.DType = poly.DTypeFloat32
			} else {
				layer.DType = dtype
			}
			if err := layer.SyncToGPU(); err != nil {
				ef.Close()
				return nil, nil, nil, nil, fmt.Errorf("sync block %d: %w", li, err)
			}
			layer.ReleaseInferenceHostWeights()
		}
		poly.ReleaseInferenceTransientMemory()
	}
	if err := tr.SyncGlobalWeightsToGPUSequential(); err != nil {
		ef.Close()
		return nil, nil, nil, nil, fmt.Errorf("global sync: %w", err)
	}
	_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
	tr.Reset()
	tr.ReleaseInferenceHostWeights()

	eos := poly.LoadEOSTokenIDsFromConfigPath(filepath.Join(snap, "config.json"))
	fmt.Printf("✅ GPU ready: layers=%d hidden=%d vocab=%d dtype=%v\n", numLayers, tr.HiddenSize, tr.VocabSize, dtype)
	return tr, tk, ef, eos, nil
}

func entityPath(modelID string) string {
	name := strings.ReplaceAll(modelID, "/", "--") + ".entity"
	// Prefer lucy_bloom_rivers/lucy_entities when running from examples/gpu_sample_poc.
	candidates := []string{
		filepath.Join("..", "..", "lucy_entities", name),
		filepath.Join("lucy_entities", name),
		filepath.Join("..", "lucy_entities", name),
	}
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}
	abs, _ := filepath.Abs(candidates[0])
	return abs
}

func resolveSnapshot(modelID string) (string, error) {
	hubDir, models, err := poly.HFInventoryMergedModels()
	if err != nil {
		return "", err
	}
	found := false
	for _, m := range models {
		if m == modelID {
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("model %q not in HF cache at %s", modelID, hubDir)
	}
	return poly.HFResolveSnapshotDirPreferManual(hubDir, modelID)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
