package qndnn

import "math"

var (
	WithSigmoid = func() *func(*Neuron) *Neuron {
		f := func(n *Neuron) *Neuron {
			n.Functions = NeuronFunctions{
				Activation: Sigmoid,
				Derivative: DerivativeSigmoid,
			}
			return n
		}
		return &f
	}
	WithRelu = func() *func(*Neuron) *Neuron {
		f := func(n *Neuron) *Neuron {
			n.Functions = NeuronFunctions{
				Activation: Relu,
				Derivative: DerivativeRelu,
			}
			return n
		}
		return &f
	}
	WithTanh = func() *func(*Neuron) *Neuron {
		f := func(n *Neuron) *Neuron {
			n.Functions = NeuronFunctions{
				Activation: HyperbolicTangent,
				Derivative: DerivativeHyperbolicTangent,
			}
			return n
		}
		return &f
	}
)

type NeuronFunctions struct {
	Activation func(float64) float64
	Derivative func(float64) float64
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -1.0*x))
}

func DerivativeSigmoid(y float64) float64 {
	x := Sigmoid(y)
	return x * (1 - x)
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func DerivativeRelu(y float64) float64 {
	if Relu(y) > 0 {
		return 1
	} else {
		return 0
	}
}

func HyperbolicTangent(x float64) float64 {
	return (math.Pow(math.E, x) - math.Pow(math.E, -x)) / (math.Pow(math.E, x) + math.Pow(math.E, -x))
}

func DerivativeHyperbolicTangent(y float64) float64 {
	return 1 - math.Pow(HyperbolicTangent(y), 2)
}
