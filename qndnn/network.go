package qndnn

import (
	"fmt"
	"math/rand/v2"
	"sync"
)

type Input struct {
	N             *Neuron
	Weight        float64
	PendingChange float64
}

func (i *Input) Result() float64 {
	return i.N.Value() * i.Weight
}

func (i *Input) UpdateWeight() {
	i.Weight -= i.PendingChange
	i.PendingChange = 0
}

type Neuron struct {
	Inputs    []*Input
	Bias      float64
	Functions NeuronFunctions

	Preset *float64 // mostly used for input definition
}

func (n *Neuron) Value() float64 {
	if n.Preset != nil {
		return *n.Preset
	}

	var mu sync.Mutex
	v := 0.0
	var q sync.WaitGroup
	q.Add(len(n.Inputs))
	for _, i := range n.Inputs {
		go func(i *Input) {
			defer q.Done()
			r := i.Result()
			mu.Lock()
			v += r
			mu.Unlock()
		}(i)
	}
	q.Wait()
	v += n.Bias
	return n.Functions.Activation(v)
}

func (n *Neuron) Learn(expected float64, learningRate float64) {
	out := n.Value()
	err := -(expected - out)                  // actual-expected
	derivative := n.Functions.Derivative(out) // derivative of output

	for _, i := range n.Inputs {
		change := err * derivative * i.N.Value()
		i.PendingChange = learningRate * change    // track pending change; to apply after back propagation is done
		i.N.Learn(expected*i.Weight, learningRate) // pass along weighted expectation + learning rate
	}
}

type NeuralNetwork [][]*Neuron

func (nn NeuralNetwork) Output(in []float64) ([]float64, error) {
	if len(in) != len(nn[0]) {
		return nil, fmt.Errorf("input didn't match first layer; expected len '%v', got '%v'", len(nn[0]), len(in))
	}

	for idx, n := range nn[0] {
		n.Preset = &in[idx]
	}

	var result []float64
	last := nn[len(nn)-1]
	for _, o := range last {
		result = append(result, o.Value())
	}
	return result, nil
}

func NewNetwork(neuronCreate *func(*Neuron) *Neuron, layers ...int) NeuralNetwork {
	var l [][]*Neuron

	if neuronCreate == nil {
		neuronCreate = &WithSigmoid
	}

	for idx, size := range layers {
		n := make([]*Neuron, size)
		init := 1.0
		for nidx, _ := range n {
			a := &Neuron{
				Bias: 0,
			}
			if idx == 0 {
				a.Preset = &init
			}
			if neuronCreate != nil {
				a = (*neuronCreate)(a)
			}
			n[nidx] = a
		}

		if idx > 0 {
			prev := l[idx-1]
			for _, ne := range n {
				ne.Bias = rand.Float64()
				for _, p := range prev {
					in := Input{
						N:      p,
						Weight: rand.Float64(),
					}
					ne.Inputs = append(ne.Inputs, &in)
				}
			}

		}
		l = append(l, n)
	}
	return l
}

func (nn NeuralNetwork) Train(
	input []float64,
	expected []float64,
	learningRate float64,
	rounds int,
) error {
	if len(nn[0]) != len(input) {
		return fmt.Errorf(
			"input doesn't match first layer (want len '%v', got len '%v')",
			len(nn[0]),
			len(input),
		)
	}

	if len(nn[len(nn)-1]) != len(expected) {
		return fmt.Errorf(
			"expected output doesn't match last layer (want len '%v', got len '%v')",
			len(nn[len(nn)-1]),
			len(expected),
		)
	}

	if rounds < 1 {
		rounds = 1
	}

	for n := 0; n < rounds; n++ {
		for idx, n := range nn[len(nn)-1] {
			n.Learn(expected[idx], learningRate) // start learning for all recursive
		}
		nn.Update() // apply all pending weight changes
	}

	return nil
}

func (nn NeuralNetwork) Update() {
	for _, l := range nn {
		for _, n := range l {
			for _, i := range n.Inputs {
				i.UpdateWeight()
			}
		}
	}
}
