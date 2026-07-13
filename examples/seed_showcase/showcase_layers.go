package seedshowcase

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

type denseStackPayload struct {
	Sizes  []int                        `json:"sizes"`
	Layers []poly.InfiniteLayerManifest `json:"layers"`
}

func buildAllSections() ([]ShowcaseSection, error) {
	builders := []func() (ShowcaseSection, error){
		demoDenseStack,
		demoSwiGLU,
		demoMHA,
		demoCNN1,
		demoCNN2,
		demoCNN3,
		demoRNN,
		demoLSTM,
		demoEmbedding,
		demoResidual,
	}
	out := make([]ShowcaseSection, 0, len(builders))
	for _, build := range builders {
		sec, err := build()
		if err != nil {
			return nil, err
		}
		out = append(out, sec)
	}
	return out, nil
}

const trainMode = poly.TrainingModeCPUMC

func trainNet(net *poly.VolumetricNetwork, batches []poly.TrainingBatch[float32]) (float64, error) {
	net.ReleaseFP32MasterWhenIdle = false
	_ = poly.ConfigureNetworkForMode(net, trainMode)
	cfg := &poly.TrainingConfig{
		Epochs:       50,
		LearningRate: 0.08,
		LossType:     "mse",
		Mode:         trainMode,
	}
	result, err := poly.Train[float32](net, batches, cfg)
	if err != nil {
		return 0, err
	}
	net.EnsureTrainingWeights()
	return result.FinalLoss, nil
}

func wireNet(net *poly.VolumetricNetwork) {
	for i := range net.Layers {
		net.Layers[i].Network = net
	}
}

func forwardTensor(net *poly.VolumetricNetwork, in *poly.Tensor[float32]) []float32 {
	net.ReleaseFP32MasterWhenIdle = false
	wireNet(net)
	_ = poly.ConfigureNetworkForMode(net, trainMode)
	net.EnsureTrainingWeights()
	for i := range net.Layers {
		poly.SyncWeightStoreForForward(&net.Layers[i])
	}
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		panic("forward nil")
	}
	return append([]float32(nil), out.Data...)
}

func sinTargets(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i) * 0.41))
	}
	return out
}

func repeatBatches(in, tgt *poly.Tensor[float32], n int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], n)
	for i := range batches {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  in.Clone(),
			Target: tgt.Clone(),
		}
	}
	return batches
}

func demoDenseStack() (ShowcaseSection, error) {
	const name = "dense-mlp"
	sizes := []int{4, 8, 4, 2}
	topo := poly.DenseTopologySeed(showcaseName+"-"+name, sizes)
	manifest, err := poly.BuildDenseManifest(topo, sizes, []string{"float32", "float32", "float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	net, err := poly.BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		return ShowcaseSection{}, err
	}
	inputs := demoInputs(6, sizes[0])
	targets := trainTargets(6, sizes[len(sizes)-1])
	if _, err := trainNet(net, trainingBatches(inputs, targets)); err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(sizes[0]), 1, sizes[0])
	trained := forwardTensor(net, in)

	layers := make([]poly.InfiniteLayerManifest, len(net.Layers))
	for i := 0; i < len(net.Layers); i++ {
		l := net.GetLayer(0, 0, 0, i)
		seed := poly.DenseLayerWeightSeed(topo, i)
		m, err := poly.ManifestFromLayer(l, seed)
		if err != nil {
			return ShowcaseSection{}, err
		}
		layers[i] = *m
	}
	payload, err := json.Marshal(denseStackPayload{Sizes: sizes, Layers: layers})
	if err != nil {
		return ShowcaseSection{}, err
	}
	return ShowcaseSection{
		Kind: name, Name: name, TopologySeed: topo, Payload: payload,
		WeightFPs: weightFPsFromManifests(layers), TrainedOutputs: trained,
	}, nil
}

func demoSwiGLU() (ShowcaseSection, error) {
	const name = "swiglu"
	specs := []poly.SwiGLUSpec{{Hidden: 8, Intermediate: 16}}
	topo := poly.SwiGLUTopologySeed(showcaseName+"-"+name, specs)
	net, err := poly.BuildSwiGLUVolumetricFromManifest(mustSwiGLUManifest(topo, specs))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(8), 1, 8)
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.SwiGLULayerWeightSeed(topo, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoMHA() (ShowcaseSection, error) {
	const name = "mha"
	specs := []poly.MHASpec{{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}}
	topo := poly.MHATopologySeed(showcaseName+"-"+name, specs)
	net, err := poly.BuildMHAVolumetricFromManifest(mustMHAManifest(topo, specs))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(8), 1, 8)
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.MHALayerWeightSeed(topo, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoCNN1() (ShowcaseSection, error) { return demoCNN("cnn1", 1, 8) }
func demoCNN2() (ShowcaseSection, error) { return demoCNN("cnn2", 2, 8) }
func demoCNN3() (ShowcaseSection, error) { return demoCNN("cnn3", 3, 4) }

func demoCNN(name string, dim, spatial int) (ShowcaseSection, error) {
	spec := poly.CNNSpec{Dim: dim, InputChannels: 2, Filters: 4, Spatial: spatial, KernelSize: 3}
	topo := poly.CNNTopologySeed(showcaseName+"-"+name, []poly.CNNSpec{spec})
	net, err := poly.BuildCNNVolumetricFromManifest(mustCNNManifest(topo, spec))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.CNNDemoInput(spec)
	if in == nil {
		return ShowcaseSection{}, fmt.Errorf("%s: demo input nil", name)
	}
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 4)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.CNNLayerWeightSeed(topo, spec, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoRNN() (ShowcaseSection, error) {
	const name = "rnn"
	sizes := []int{4, 6}
	topo := poly.RNNTopologySeed(showcaseName+"-"+name, sizes)
	net, err := poly.BuildRNNVolumetricFromManifest(mustRNNManifest(topo, sizes))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.RNNLayerWeightSeed(topo, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoLSTM() (ShowcaseSection, error) {
	const name = "lstm"
	sizes := []int{4, 6}
	topo := poly.LSTMTopologySeed(showcaseName+"-"+name, sizes)
	net, err := poly.BuildLSTMVolumetricFromManifest(mustLSTMManifest(topo, sizes))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.LSTMLayerWeightSeed(topo, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoEmbedding() (ShowcaseSection, error) {
	const name = "embedding"
	spec := poly.EmbeddingSpec{VocabSize: 16, EmbeddingDim: 8, SeqLen: 4}
	topo := poly.EmbeddingTopologySeed(showcaseName+"-"+name, []poly.EmbeddingSpec{spec})
	net, err := poly.BuildEmbeddingVolumetricFromManifest(mustEmbeddingManifest(topo, spec))
	if err != nil {
		return ShowcaseSection{}, err
	}
	tokens := poly.EmbeddingDemoTokens(spec.VocabSize, spec.SeqLen)
	outInit, _, _ := poly.ForwardPolymorphic(net, tokens)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(tokens, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, tokens)
	l := net.GetLayer(0, 0, 0, 0)
	seed := poly.EmbeddingLayerWeightSeed(topo, 0)
		m, err := poly.ManifestFromLayer(l, seed)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return sectionFromManifest(name, topo, payload, m, trained), nil
}

func demoResidual() (ShowcaseSection, error) {
	const name = "residual"
	spec := poly.ResidualSpec{In: 4, Out: 4}
	topo := poly.ResidualTopologySeed(showcaseName+"-"+name, spec)
	net, err := poly.BuildResidualVolumetricFromManifest(mustResidualManifest(topo, spec))
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	outInit, _, _ := poly.ForwardPolymorphic(net, in)
	tgt := poly.NewTensorFromSlice(sinTargets(len(outInit.Data)), outInit.Shape...)
	if _, err := trainNet(net, repeatBatches(in, tgt, 6)); err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardTensor(net, in)
	m, err := poly.ManifestFromResidualBlock(net, topo, spec)
	if err != nil {
		return ShowcaseSection{}, err
	}
	payload, _ := json.Marshal(m)
	return ShowcaseSection{
		Kind: name, Name: name, TopologySeed: topo, Payload: payload,
		WeightFPs: []uint64{m.Dense.WeightFP}, TrainedOutputs: trained,
	}, nil
}

func rebuildSection(sec ShowcaseSection) ([]float32, error) {
	switch sec.Kind {
	case "dense-mlp":
		var p denseStackPayload
		if err := json.Unmarshal(sec.Payload, &p); err != nil {
			return nil, err
		}
		net := poly.NewVolumetricNetwork(1, 1, 1, len(p.Layers))
		net.InitSeed = sec.TopologySeed
		for i, lm := range p.Layers {
			bl, err := poly.BuildLayerFromManifest(&lm)
			if err != nil {
				return nil, err
			}
			*net.GetLayer(0, 0, 0, i) = *bl
		}
		wireNet(net)
		in := poly.NewTensorFromSlice(seedVec(p.Sizes[0]), 1, p.Sizes[0])
		return forwardTensor(net, in), nil
	case "residual":
		m, err := poly.ParseInfiniteResidual(sec.Payload)
		if err != nil {
			return nil, err
		}
		net, err := poly.BuildResidualFromManifest(m)
		if err != nil {
			return nil, err
		}
		wireNet(net)
		in := poly.NewTensorFromSlice(seedVec(m.Spec.In), 1, m.Spec.In)
		return forwardTensor(net, in), nil
	default:
		m, err := poly.ParseInfiniteLayer(sec.Payload)
		if err != nil {
			return nil, err
		}
		l, err := poly.BuildLayerFromManifest(m)
		if err != nil {
			return nil, err
		}
		net := poly.NewVolumetricNetwork(1, 1, 1, 1)
		*net.GetLayer(0, 0, 0, 0) = *l
		wireNet(net)
		switch m.Kind {
		case "swiglu":
			in := poly.NewTensorFromSlice(seedVec(m.Hidden), 1, m.Hidden)
			return forwardTensor(net, in), nil
		case "mha":
			dim := 8
			if m.MHA != nil {
				dim = m.MHA.DModel
			}
			in := poly.NewTensorFromSlice(seedVec(dim), 1, dim)
			return forwardTensor(net, in), nil
		case "cnn1", "cnn2", "cnn3":
			if m.CNN == nil {
				return nil, fmt.Errorf("cnn spec missing")
			}
			in := poly.CNNDemoInput(*m.CNN)
			if in == nil {
				return nil, fmt.Errorf("cnn demo input nil")
			}
			return forwardTensor(net, in), nil
		case "rnn", "lstm":
			in := poly.NewTensorFromSlice(seedVec(m.In), 1, m.In)
			return forwardTensor(net, in), nil
		case "embedding":
			vocab, seq := 16, 4
			if m.Embedding != nil {
				vocab, seq = m.Embedding.VocabSize, m.Embedding.SeqLen
			}
			tokens := poly.EmbeddingDemoTokens(vocab, seq)
			return forwardTensor(net, tokens), nil
		default:
			return nil, fmt.Errorf("unknown layer kind %q", m.Kind)
		}
	}
}

func seedVec(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.01 * float32((i+1)%9)
	}
	return out
}

func mustSwiGLUManifest(topo uint64, specs []poly.SwiGLUSpec) *poly.SwiGLUWeightManifest {
	m, err := poly.BuildSwiGLUManifest(topo, specs, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustMHAManifest(topo uint64, specs []poly.MHASpec) *poly.MHAWeightManifest {
	m, err := poly.BuildMHAManifest(topo, specs, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustCNNManifest(topo uint64, spec poly.CNNSpec) *poly.CNNWeightManifest {
	m, err := poly.BuildCNNManifest(topo, []poly.CNNSpec{spec}, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustRNNManifest(topo uint64, sizes []int) *poly.RNNWeightManifest {
	m, err := poly.BuildRNNManifest(topo, sizes, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustLSTMManifest(topo uint64, sizes []int) *poly.LSTMWeightManifest {
	m, err := poly.BuildLSTMManifest(topo, sizes, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustEmbeddingManifest(topo uint64, spec poly.EmbeddingSpec) *poly.EmbeddingWeightManifest {
	m, err := poly.BuildEmbeddingManifest(topo, []poly.EmbeddingSpec{spec}, []string{"float32"})
	if err != nil {
		panic(err)
	}
	return m
}

func mustResidualManifest(topo uint64, spec poly.ResidualSpec) *poly.ResidualWeightManifest {
	m, err := poly.BuildResidualManifest(topo, spec, "float32")
	if err != nil {
		panic(err)
	}
	return m
}

func sectionOverrideSummary(sec ShowcaseSection) int {
	if sec.Kind == "dense-mlp" {
		var p denseStackPayload
		if json.Unmarshal(sec.Payload, &p) != nil {
			return -1
		}
		n := 0
		for _, l := range p.Layers {
			n += l.OverrideCount()
		}
		return n
	}
	if sec.Kind == "residual" {
		m, _ := poly.ParseInfiniteResidual(sec.Payload)
		if m == nil {
			return -1
		}
		return m.OverrideCount()
	}
	m, _ := poly.ParseInfiniteLayer(sec.Payload)
	if m == nil {
		return -1
	}
	return m.OverrideCount()
}

func sectionFromManifest(name string, topo uint64, payload []byte, m *poly.InfiniteLayerManifest, trained []float32) ShowcaseSection {
	return ShowcaseSection{
		Kind: name, Name: name, TopologySeed: topo, Payload: payload,
		WeightFPs: []uint64{m.WeightFP}, TrainedOutputs: trained,
	}
}

func weightFPsFromManifests(layers []poly.InfiniteLayerManifest) []uint64 {
	out := make([]uint64, len(layers))
	for i, l := range layers {
		out[i] = l.WeightFP
	}
	return out
}

func storedWeightFPs(sec ShowcaseSection) ([]uint64, error) {
	if len(sec.WeightFPs) > 0 {
		return sec.WeightFPs, nil
	}
	switch sec.Kind {
	case "dense-mlp":
		var p denseStackPayload
		if err := json.Unmarshal(sec.Payload, &p); err != nil {
			return nil, err
		}
		return weightFPsFromManifests(p.Layers), nil
	case "residual":
		m, err := poly.ParseInfiniteResidual(sec.Payload)
		if err != nil {
			return nil, err
		}
		return []uint64{m.Dense.WeightFP}, nil
	default:
		m, err := poly.ParseInfiniteLayer(sec.Payload)
		if err != nil {
			return nil, err
		}
		return []uint64{m.WeightFP}, nil
	}
}

func rebuiltWeightFPs(sec ShowcaseSection) ([]uint64, error) {
	switch sec.Kind {
	case "dense-mlp":
		var p denseStackPayload
		if err := json.Unmarshal(sec.Payload, &p); err != nil {
			return nil, err
		}
		out := make([]uint64, len(p.Layers))
		for i, lm := range p.Layers {
			bl, err := poly.BuildLayerFromManifest(&lm)
			if err != nil {
				return nil, err
			}
			out[i] = poly.WeightStoreFingerprint(bl.WeightStore)
		}
		return out, nil
	case "residual":
		m, err := poly.ParseInfiniteResidual(sec.Payload)
		if err != nil {
			return nil, err
		}
		net, err := poly.BuildResidualFromManifest(m)
		if err != nil {
			return nil, err
		}
		dense := net.GetLayer(0, 0, 0, 0)
		return []uint64{poly.WeightStoreFingerprint(dense.WeightStore)}, nil
	default:
		m, err := poly.ParseInfiniteLayer(sec.Payload)
		if err != nil {
			return nil, err
		}
		bl, err := poly.BuildLayerFromManifest(m)
		if err != nil {
			return nil, err
		}
		return []uint64{poly.WeightStoreFingerprint(bl.WeightStore)}, nil
	}
}

func verifySectionWeightReload(sec ShowcaseSection) error {
	want, err := storedWeightFPs(sec)
	if err != nil {
		return fmt.Errorf("stored weight fp: %w", err)
	}
	got, err := rebuiltWeightFPs(sec)
	if err != nil {
		return fmt.Errorf("rebuild weight fp: %w", err)
	}
	if len(want) != len(got) {
		return fmt.Errorf("weight fp count %d vs %d", len(got), len(want))
	}
	for i := range want {
		if want[i] != got[i] {
			return fmt.Errorf("layer %d weight fp 0x%x vs 0x%x", i, got[i], want[i])
		}
	}
	return nil
}
