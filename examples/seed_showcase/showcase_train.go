package seedshowcase

import (
	"github.com/openfluke/loom/poly"
)

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

func evalMSETensor(net *poly.VolumetricNetwork, in, tgt *poly.Tensor[float32]) float32 {
	out, _, _ := poly.ForwardPolymorphic(net, in)
	if out == nil {
		return 1e9
	}
	var sum float32
	for j := range out.Data {
		d := out.Data[j] - tgt.Data[j]
		sum += d * d
	}
	return sum / float32(len(out.Data))
}

func trainDenseSeeds(net *poly.VolumetricNetwork, seeds []uint64, sizes []int, inputs, targets [][]float32) {
	const epochs = 50
	const mutPerLayer = 250
	rng := poly.NewSeedRNG(poly.SeedFrom("seed-showcase-train", seeds[0]))

	bestLoss := evalMSE(net, inputs, targets)
	for epoch := 0; epoch < epochs; epoch++ {
		for li := range seeds {
			in := sizes[li]
			for m := 0; m < mutPerLayer; m++ {
				trial := mutateSeed(seeds[li], rng.Uint64())
				applyDenseLayerSeed(net, li, trial, in)
				loss := evalMSE(net, inputs, targets)
				if loss < bestLoss {
					bestLoss = loss
					seeds[li] = trial
				}
			}
			applyDenseLayerSeed(net, li, seeds[li], in)
		}
	}
}

func applyDenseLayerSeed(net *poly.VolumetricNetwork, layerIdx int, seed uint64, in int) {
	l := net.GetLayer(0, 0, 0, layerIdx)
	poly.InitWeightStoreHeSeeded(l.WeightStore, in, seed)
}

func trainOneLayerSeed(
	rngKey uint64,
	seed uint64,
	rebuild func(uint64) (*poly.VolumetricNetwork, error),
	lossFn func(*poly.VolumetricNetwork) float32,
) uint64 {
	const epochs = 50
	const mutPerTrial = 300
	rng := poly.NewSeedRNG(poly.SeedFrom("seed-showcase-train", rngKey, seed))

	net, err := rebuild(seed)
	if err != nil {
		return seed
	}
	bestLoss := lossFn(net)
	best := seed
	for epoch := 0; epoch < epochs; epoch++ {
		for m := 0; m < mutPerTrial; m++ {
			trial := mutateSeed(best, rng.Uint64())
			net, err = rebuild(trial)
			if err != nil {
				continue
			}
			loss := lossFn(net)
			if loss < bestLoss {
				bestLoss = loss
				best = trial
			}
		}
	}
	return best
}
