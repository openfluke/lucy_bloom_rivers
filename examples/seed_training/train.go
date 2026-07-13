package seedtraining

import (
	"github.com/openfluke/loom/poly"
)

func trainLayerSeeds(net *poly.VolumetricNetwork, seeds []uint64, sizes []int, inputs, targets [][]float32) {
	const epochs = 60
	const mutPerLayer = 350
	rng := poly.NewSeedRNG(poly.SeedFrom("seed-training-wine", seeds[0]))

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

func evalMAE(net *poly.VolumetricNetwork, inputs, targets [][]float32) float32 {
	var sum float32
	for i, in := range inputs {
		t := poly.NewTensorFromSlice(in, 1, len(in))
		out, _, _ := poly.ForwardPolymorphic(net, t)
		if out == nil {
			continue
		}
		for j := range out.Data {
			d := out.Data[j] - targets[i][j]
			if d < 0 {
				d = -d
			}
			sum += d
		}
	}
	return sum / float32(len(inputs))
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
