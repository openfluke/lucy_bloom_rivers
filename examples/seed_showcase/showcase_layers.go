package seedshowcase

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

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

func sinTargets(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i) * 0.41))
	}
	return out
}

func seedVec(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.01 * float32((i+1)%9)
	}
	return out
}

func sectionFromManifest(kind, name string, topo uint64, payload []byte, weightFPs []uint64, trained []float32) ShowcaseSection {
	return ShowcaseSection{
		Kind: kind, Name: name, TopologySeed: topo,
		Payload: payload, WeightFPs: weightFPs, TrainedOutputs: trained,
	}
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
	inputs := demoInputs(8, sizes[0])
	targets := trainTargets(8, sizes[len(sizes)-1])
	seeds := make([]uint64, len(manifest.Layers))
	for i := range manifest.Layers {
		seeds[i] = manifest.Layers[i].LayerSeed
	}
	lossBefore := evalMSE(net, inputs, targets)
	trainDenseSeeds(net, seeds, sizes, inputs, targets)
	lossAfter := evalMSE(net, inputs, targets)
	if lossAfter >= lossBefore {
		return ShowcaseSection{}, fmt.Errorf("dense-mlp: seed training did not reduce loss")
	}
	for i, s := range seeds {
		manifest.Layers[i].LayerSeed = s
	}
	if err := refreshDenseManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, err = poly.BuildDenseVolumetricFromManifest(manifest)
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(sizes[0]), 1, sizes[0])
	trained := forwardNet(net, in)
	payload, err := json.Marshal(manifest)
	if err != nil {
		return ShowcaseSection{}, err
	}
	return sectionFromManifest(name, name, topo, payload, denseManifestWeightFPs(manifest), trained), nil
}

func demoSwiGLU() (ShowcaseSection, error) {
	const name = "swiglu"
	specs := []poly.SwiGLUSpec{{Hidden: 8, Intermediate: 16}}
	topo := poly.SwiGLUTopologySeed(showcaseName+"-"+name, specs)
	manifest, err := poly.BuildSwiGLUManifest(topo, specs, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(8), 1, 8)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildSwiGLUVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshSwiGLUManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, err := poly.BuildSwiGLUVolumetricFromManifest(manifest)
	if err != nil {
		return ShowcaseSection{}, err
	}
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoMHA() (ShowcaseSection, error) {
	const name = "mha"
	specs := []poly.MHASpec{{DModel: 8, NumHeads: 2, NumKVHeads: 2, HeadDim: 4, QueryDim: 8}}
	topo := poly.MHATopologySeed(showcaseName+"-"+name, specs)
	manifest, err := poly.BuildMHAManifest(topo, specs, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(8), 1, 8)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildMHAVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshMHAManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildMHAVolumetricFromManifest(manifest)
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoCNN1() (ShowcaseSection, error) { return demoCNN("cnn1", 1, 8) }
func demoCNN2() (ShowcaseSection, error) { return demoCNN("cnn2", 2, 8) }
func demoCNN3() (ShowcaseSection, error) { return demoCNN("cnn3", 3, 4) }

func demoCNN(name string, dim, spatial int) (ShowcaseSection, error) {
	spec := poly.CNNSpec{Dim: dim, InputChannels: 2, Filters: 4, Spatial: spatial, KernelSize: 3}
	topo := poly.CNNTopologySeed(showcaseName+"-"+name, []poly.CNNSpec{spec})
	manifest, err := poly.BuildCNNManifest(topo, []poly.CNNSpec{spec}, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.CNNDemoInput(spec)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildCNNVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshCNNManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildCNNVolumetricFromManifest(manifest)
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoRNN() (ShowcaseSection, error) {
	const name = "rnn"
	sizes := []int{4, 6}
	topo := poly.RNNTopologySeed(showcaseName+"-"+name, sizes)
	manifest, err := poly.BuildRNNManifest(topo, sizes, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildRNNVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshRNNManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildRNNVolumetricFromManifest(manifest)
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoLSTM() (ShowcaseSection, error) {
	const name = "lstm"
	sizes := []int{4, 6}
	topo := poly.LSTMTopologySeed(showcaseName+"-"+name, sizes)
	manifest, err := poly.BuildLSTMManifest(topo, sizes, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildLSTMVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshLSTMManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildLSTMVolumetricFromManifest(manifest)
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoEmbedding() (ShowcaseSection, error) {
	const name = "embedding"
	spec := poly.EmbeddingSpec{VocabSize: 16, EmbeddingDim: 8, SeqLen: 4}
	topo := poly.EmbeddingTopologySeed(showcaseName+"-"+name, []poly.EmbeddingSpec{spec})
	manifest, err := poly.BuildEmbeddingManifest(topo, []poly.EmbeddingSpec{spec}, []string{"float32"})
	if err != nil {
		return ShowcaseSection{}, err
	}
	tokens := poly.EmbeddingDemoTokens(spec.VocabSize, spec.SeqLen)
	manifest.Layers[0].LayerSeed = trainOneLayerSeed(topo, manifest.Layers[0].LayerSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.Layers[0].LayerSeed = seed
			return poly.BuildEmbeddingVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, tokens)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, tokens, tgt)
		},
	)
	if err := refreshEmbeddingManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildEmbeddingVolumetricFromManifest(manifest)
	trained := forwardNet(net, tokens)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.Layers[0].WeightFP}, trained), nil
}

func demoResidual() (ShowcaseSection, error) {
	const name = "residual"
	spec := poly.ResidualSpec{In: 4, Out: 4}
	topo := poly.ResidualTopologySeed(showcaseName+"-"+name, spec)
	manifest, err := poly.BuildResidualManifest(topo, spec, "float32")
	if err != nil {
		return ShowcaseSection{}, err
	}
	in := poly.NewTensorFromSlice(seedVec(4), 1, 4)
	manifest.DenseSeed = trainOneLayerSeed(topo, manifest.DenseSeed,
		func(seed uint64) (*poly.VolumetricNetwork, error) {
			manifest.DenseSeed = seed
			return poly.BuildResidualVolumetricFromManifest(manifest)
		},
		func(net *poly.VolumetricNetwork) float32 {
			out, _, _ := poly.ForwardPolymorphic(net, in)
			if out == nil {
				return 1e9
			}
			tgt := poly.NewTensorFromSlice(sinTargets(len(out.Data)), out.Shape...)
			return evalMSETensor(net, in, tgt)
		},
	)
	if err := refreshResidualManifest(manifest); err != nil {
		return ShowcaseSection{}, err
	}
	net, _ := poly.BuildResidualVolumetricFromManifest(manifest)
	trained := forwardNet(net, in)
	payload, _ := json.Marshal(manifest)
	return sectionFromManifest(name, name, topo, payload, []uint64{manifest.DenseWeightFP}, trained), nil
}

func forwardNet(net *poly.VolumetricNetwork, in *poly.Tensor[float32]) []float32 {
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		panic("forward nil")
	}
	return append([]float32(nil), out.Data...)
}

func rebuildSection(sec ShowcaseSection) ([]float32, error) {
	switch sec.Kind {
	case "dense-mlp":
		var m poly.DenseWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildDenseVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.NewTensorFromSlice(seedVec(m.Sizes[0]), 1, m.Sizes[0])
		return forwardNet(net, in), nil
	case "swiglu":
		var m poly.SwiGLUWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildSwiGLUVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.NewTensorFromSlice(seedVec(m.Specs[0].Hidden), 1, m.Specs[0].Hidden)
		return forwardNet(net, in), nil
	case "mha":
		var m poly.MHAWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildMHAVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		dim := m.Specs[0].DModel
		in := poly.NewTensorFromSlice(seedVec(dim), 1, dim)
		return forwardNet(net, in), nil
	case "cnn1", "cnn2", "cnn3":
		var m poly.CNNWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildCNNVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.CNNDemoInput(m.Specs[0])
		if in == nil {
			return nil, fmt.Errorf("cnn demo input nil")
		}
		return forwardNet(net, in), nil
	case "rnn":
		var m poly.RNNWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildRNNVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.NewTensorFromSlice(seedVec(m.Sizes[0]), 1, m.Sizes[0])
		return forwardNet(net, in), nil
	case "lstm":
		var m poly.LSTMWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildLSTMVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.NewTensorFromSlice(seedVec(m.Sizes[0]), 1, m.Sizes[0])
		return forwardNet(net, in), nil
	case "embedding":
		var m poly.EmbeddingWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildEmbeddingVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		spec := m.Specs[0]
		tokens := poly.EmbeddingDemoTokens(spec.VocabSize, spec.SeqLen)
		return forwardNet(net, tokens), nil
	case "residual":
		var m poly.ResidualWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildResidualVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		in := poly.NewTensorFromSlice(seedVec(m.Spec.In), 1, m.Spec.In)
		return forwardNet(net, in), nil
	default:
		return nil, fmt.Errorf("unknown section kind %q", sec.Kind)
	}
}

func verifySectionSeedReload(sec ShowcaseSection) error {
	if err := assertSeedsOnlyPayload(sec.Payload); err != nil {
		return err
	}
	want, err := storedWeightFPs(sec)
	if err != nil {
		return err
	}
	got, err := rebuiltWeightFPs(sec)
	if err != nil {
		return err
	}
	if len(want) != len(got) {
		return fmt.Errorf("weight fp count %d vs %d", len(got), len(want))
	}
	for i := range want {
		if want[i] != got[i] {
			return fmt.Errorf("layer %d weight fp 0x%x vs 0x%x", i, got[i], want[i])
		}
	}
	if len(sec.TrainedOutputs) > 0 {
		out, err := rebuildSection(sec)
		if err != nil {
			return err
		}
		if !sliceEqual(out, sec.TrainedOutputs) {
			return fmt.Errorf("forward output mismatch vs saved trained_outputs")
		}
	}
	return nil
}

func assertSeedsOnlyPayload(payload []byte) error {
	s := string(payload)
	if containsJSONKey(s, "overrides") {
		return fmt.Errorf("payload contains weight overrides — retrain with v5 seeds-only format")
	}
	if containsJSONKey(s, "loom-infinite-layer") {
		return fmt.Errorf("payload contains infinite-layer manifest — retrain with v5 seeds-only format")
	}
	return nil
}

func containsJSONKey(s, key string) bool {
	return len(s) > 0 && (len(key) > 0) &&
		(jsonContains(s, `"`+key+`"`) || jsonContains(s, `"format": "loom-infinite`))
}

func jsonContains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func storedWeightFPs(sec ShowcaseSection) ([]uint64, error) {
	if len(sec.WeightFPs) > 0 {
		return sec.WeightFPs, nil
	}
	return rebuiltWeightFPs(sec)
}

func rebuiltWeightFPs(sec ShowcaseSection) ([]uint64, error) {
	switch sec.Kind {
	case "dense-mlp":
		var m poly.DenseWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildDenseVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		return layerWeightFPs(net), nil
	case "residual":
		var m poly.ResidualWeightManifest
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		net, err := poly.BuildResidualVolumetricFromManifest(&m)
		if err != nil {
			return nil, err
		}
		return []uint64{poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)}, nil
	default:
		var m struct {
			Layers []struct {
				WeightFP uint64 `json:"weight_fp"`
			} `json:"layers"`
			DenseWeightFP uint64 `json:"dense_weight_fp"`
		}
		if err := json.Unmarshal(sec.Payload, &m); err != nil {
			return nil, err
		}
		if m.DenseWeightFP != 0 {
			return []uint64{m.DenseWeightFP}, nil
		}
		out := make([]uint64, len(m.Layers))
		for i, l := range m.Layers {
			out[i] = l.WeightFP
		}
		return out, nil
	}
}

func layerWeightFPs(net *poly.VolumetricNetwork) []uint64 {
	out := make([]uint64, net.LayersPerCell)
	for i := 0; i < net.LayersPerCell; i++ {
		l := net.GetLayer(0, 0, 0, i)
		out[i] = poly.WeightStoreFingerprint(l.WeightStore)
	}
	return out
}

func denseManifestWeightFPs(m *poly.DenseWeightManifest) []uint64 {
	out := make([]uint64, len(m.Layers))
	for i, l := range m.Layers {
		out[i] = l.WeightFP
	}
	return out
}

func sectionLayerSeedCount(sec ShowcaseSection) int {
	switch sec.Kind {
	case "dense-mlp":
		var m poly.DenseWeightManifest
		if json.Unmarshal(sec.Payload, &m) != nil {
			return -1
		}
		return len(m.Layers)
	case "residual":
		return 1
	default:
		var m struct {
			Layers []struct {
				LayerSeed uint64 `json:"layer_seed"`
			} `json:"layers"`
		}
		if json.Unmarshal(sec.Payload, &m) != nil {
			return -1
		}
		return len(m.Layers)
	}
}

func refreshDenseManifest(m *poly.DenseWeightManifest) error {
	net, err := poly.BuildDenseVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	for i := range m.Layers {
		l := net.GetLayer(0, 0, 0, i)
		m.Layers[i].WeightFP = poly.WeightStoreFingerprint(l.WeightStore)
	}
	out, err := poly.ForwardDenseManifest(m)
	if err != nil {
		return err
	}
	m.ForwardFP = denseHashFromFloat64(out)
	m.NetworkFP = xorLayerFPs(m.Layers)
	return nil
}

func refreshSwiGLUManifest(m *poly.SwiGLUWeightManifest) error {
	net, err := poly.BuildSwiGLUVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := swigluForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshMHAManifest(m *poly.MHAWeightManifest) error {
	net, err := poly.BuildMHAVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := mhaForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshCNNManifest(m *poly.CNNWeightManifest) error {
	net, err := poly.BuildCNNVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := cnnForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshRNNManifest(m *poly.RNNWeightManifest) error {
	net, err := poly.BuildRNNVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := rnnForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshLSTMManifest(m *poly.LSTMWeightManifest) error {
	net, err := poly.BuildLSTMVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := lstmForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshEmbeddingManifest(m *poly.EmbeddingWeightManifest) error {
	net, err := poly.BuildEmbeddingVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.Layers[0].WeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := embeddingForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	m.NetworkFP = m.Layers[0].WeightFP
	return nil
}

func refreshResidualManifest(m *poly.ResidualWeightManifest) error {
	net, err := poly.BuildResidualVolumetricFromManifest(m)
	if err != nil {
		return err
	}
	m.DenseWeightFP = poly.WeightStoreFingerprint(net.GetLayer(0, 0, 0, 0).WeightStore)
	fp, err := residualForwardFP(m)
	if err != nil {
		return err
	}
	m.ForwardFP = fp
	return nil
}

// forward FP helpers via rebuild + demo input (manifests are seeds-only).
func swigluForwardFP(m *poly.SwiGLUWeightManifest) (uint64, error) {
	net, err := poly.BuildSwiGLUVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	in := poly.NewTensorFromSlice(seedVec(m.Specs[0].Hidden), 1, m.Specs[0].Hidden)
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("swiglu forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func mhaForwardFP(m *poly.MHAWeightManifest) (uint64, error) {
	net, err := poly.BuildMHAVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	dim := m.Specs[0].DModel
	in := poly.NewTensorFromSlice(seedVec(dim), 1, dim)
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("mha forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func cnnForwardFP(m *poly.CNNWeightManifest) (uint64, error) {
	net, err := poly.BuildCNNVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	in := poly.CNNDemoInput(m.Specs[0])
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("cnn forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func rnnForwardFP(m *poly.RNNWeightManifest) (uint64, error) {
	net, err := poly.BuildRNNVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	in := poly.NewTensorFromSlice(seedVec(m.Sizes[0]), 1, m.Sizes[0])
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("rnn forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func lstmForwardFP(m *poly.LSTMWeightManifest) (uint64, error) {
	net, err := poly.BuildLSTMVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	in := poly.NewTensorFromSlice(seedVec(m.Sizes[0]), 1, m.Sizes[0])
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 0, fmt.Errorf("lstm forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func embeddingForwardFP(m *poly.EmbeddingWeightManifest) (uint64, error) {
	net, err := poly.BuildEmbeddingVolumetricFromManifest(m)
	if err != nil {
		return 0, err
	}
	spec := m.Specs[0]
	tokens := poly.EmbeddingDemoTokens(spec.VocabSize, spec.SeqLen)
	out, _, _ := poly.ForwardPolymorphic(net, tokens)
	if out == nil {
		return 0, fmt.Errorf("embedding forward nil")
	}
	return seedOutputHash(out.Data), nil
}

func residualForwardFP(m *poly.ResidualWeightManifest) (uint64, error) {
	out, err := poly.ForwardResidualManifest(m)
	if err != nil {
		return 0, err
	}
	return seedOutputHash(out), nil
}

func seedOutputHash(data []float32) uint64 {
	h := uint64(14695981039346656037)
	for _, v := range data {
		bits := math.Float32bits(v)
		h ^= uint64(bits)
		h *= 1099511628211
	}
	return h
}

func denseHashFromFloat64(out []float64) uint64 {
	data := make([]float32, len(out))
	for i, v := range out {
		data[i] = float32(v)
	}
	return seedOutputHash(data)
}

func xorLayerFPs(layers []poly.DenseLayerManifest) uint64 {
	var fp uint64
	for _, l := range layers {
		fp ^= l.WeightFP
	}
	return fp
}
