package seedshowcase

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/poly"
)

func trainingBatches(inputs, targets [][]float32) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], len(inputs))
	for i := range inputs {
		batches[i] = poly.TrainingBatch[float32]{
			Input:  poly.NewTensorFromSlice(inputs[i], 1, len(inputs[i])),
			Target: poly.NewTensorFromSlice(targets[i], 1, len(targets[i])),
		}
	}
	return batches
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
		if !sliceEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}
