package qndnn

import "math"

var (
	WithSigmoid = func(n *Neuron) *Neuron {
		n.Functions = NeuronFunctions{
			Activation: sigmoid,
			Derivative: derivativeSigmoid,
		}
		return n
	}
	WithReLU = func(n *Neuron) *Neuron {
		n.Functions = NeuronFunctions{
			Activation: relu,
			Derivative: reluDerivative,
		}
		return n
	}
	WithTanh = func(n *Neuron) *Neuron {
		n.Functions = NeuronFunctions{
			Activation: hyperbolicTan,
			Derivative: hyperbolicTanDerivative,
		}
		return n
	}
)

type NeuronFunctions struct {
	Activation func(float64) float64
	Derivative func(float64) float64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -1.0*x))
}

func derivativeSigmoid(y float64) float64 {
	return sigmoid(y) * (1 - sigmoid(y))
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluDerivative(y float64) float64 {
	if relu(y) >= 0 {
		return 1
	} else {
		return 0
	}
}

func hyperbolicTan(x float64) float64 {
	return (math.Pow(math.E, x) - math.Pow(math.E, -1)) / (math.Pow(math.E, x) + math.Pow(math.E, -1))
}

func hyperbolicTanDerivative(y float64) float64 {
	return 1 - math.Pow(hyperbolicTan(y), 2)
}
