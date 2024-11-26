package qndnn

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"sync"
	"time"
)

type Input struct {
	N      *Neuron `json:"-"`
	Weight float64 `json:"weight"`

	changeLock    sync.Mutex
	PendingChange float64 `json:"-"`
}

func (i *Input) Result() float64 {
	return i.N.Value() * i.Weight
}

func (i *Input) UpdateWeight() {
	i.Weight -= i.PendingChange
	i.PendingChange = 0
}

type Neuron struct {
	Inputs    []*Input        `json:"inputs"`
	Bias      float64         `json:"bias"`
	Functions NeuronFunctions `json:"-"`

	Preset *float64 `json:"preset"` // mostly used for input definition
}

func (n *Neuron) Input() float64 {
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
	return v
}

func (n *Neuron) Value() float64 {
	if n.Preset != nil {
		return *n.Preset
	}

	return n.Functions.Activation(n.Input())
}

func (n *Neuron) Learn(delta float64, weight float64, learningRate float64) {
	derivative := n.Functions.Derivative(n.Input()) // derivative of output
	d := derivative * weight * delta

	var wg sync.WaitGroup
	wg.Add(len(n.Inputs))
	for _, i := range n.Inputs {
		go func(i *Input) {
			defer wg.Done()

			pc := learningRate * d * i.N.Value()
			i.changeLock.Lock()
			i.PendingChange += pc // track pending change; to apply after back propagation is done
			i.changeLock.Unlock()
			i.N.Learn(d, i.Weight, learningRate) // pass along weighted expectation + learning rate
		}(i)
	}
	wg.Wait()
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

func NewNeuralNet(neuronCreate *func(*Neuron) *Neuron, layers ...int) NeuralNetwork {
	var l [][]*Neuron

	if neuronCreate == nil {
		neuronCreate = WithSigmoid()
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

type Expectations struct {
	Input  []float64
	Output []float64
}

type Strategy func(errs []float64) bool

func RoundStrategy(rounds int) Strategy {
	if rounds < 0 {
		rounds = 1
	}

	counter := 0
	return func(_ []float64) bool {
		if counter == rounds {
			return false
		} else {
			counter++
			return true
		}
	}
}

func ThresholdStrategy(errorThreshold float64, stopAfter time.Duration) Strategy {
	start := time.Now()
	end := start.Add(stopAfter)
	return func(errs []float64) bool {
		if stopAfter > 0 && time.Now().After(end) {
			return false
		}

		cumulated := 0.0
		for _, err := range errs {
			cumulated += math.Abs(err)
		}

		if len(errs) > 0 && // this is important, because on first run we don't have errors yet; so it would stop immediately
			cumulated <= errorThreshold {
			return false
		} else {
			return true
		}
	}
}

func WithLoggingStrategy(out io.Writer, strategy Strategy) Strategy {
	f := strategy
	return func(errs []float64) bool {
		cum := 0.0
		for _, err := range errs {
			cum += math.Abs(err)
		}
		// we ignore errors, since this strategy is rather for user info
		_, _ = fmt.Fprintf(out, "%s – cumulated error: %.10f\n", time.Now().Format(time.DateTime), cum)
		return f(errs)
	}
}

func (nn NeuralNetwork) Train(
	expectations []Expectations,
	learningRate float64,
	strategy Strategy,
) error {
	for _, e := range expectations {
		if len(nn[0]) != len(e.Input) {
			return fmt.Errorf(
				"input doesn't match first layer (want len '%v', got len '%v')",
				len(nn[0]),
				len(e.Input),
			)
		}

		if len(nn[len(nn)-1]) != len(e.Output) {
			return fmt.Errorf(
				"expected output doesn't match last layer (want len '%v', got len '%v')",
				len(nn[len(nn)-1]),
				len(e.Output),
			)
		}
	}

	var errs []float64
	for {
		// based on strategy, abort or continue
		if !strategy(errs) {
			return nil
		}

		for _, e := range expectations {
			errs = []float64{}
			// set input
			for idx, i := range e.Input {
				nn[0][idx].Preset = &i
			}

			// iterate through all output nodes, comparing result with expectation
			for idx, n := range nn[len(nn)-1] {
				in := n.Input()
				out := n.Value()
				expected := e.Output[idx]
				err := out - expected
				delta := err * n.Functions.Derivative(in)
				errs = append(errs, err)

				n.Learn(delta, 1.0, learningRate) // start learning for all recursive; delta is taken full
			}

			nn.Update() // apply all pending weight changes
		}
	}
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

func (nn NeuralNetwork) Serialize() (string, error) {
	out, err := json.Marshal(nn)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(out), nil
}

func NewNeuralNetFromSerialized(neuronCreate *func(*Neuron) *Neuron, serialized string) (NeuralNetwork, error) {
	content, err := base64.StdEncoding.DecodeString(serialized)
	if err != nil {
		return nil, err
	}

	if neuronCreate == nil {
		neuronCreate = WithSigmoid()
	}

	net := NeuralNetwork{}
	err = json.Unmarshal(content, &net)
	if err != nil {
		return nil, err
	}

	for idx, l := range net {
		for _, n := range l {
			n = (*neuronCreate)(n)
		}

		if idx == 0 {
			continue // skip input layer for reconnecting
		}

		before := net[idx-1]
		for _, n := range l {
			for iidx, in := range before {
				n.Inputs[iidx].N = in // reconnect neurons
			}
		}

	}

	return net, nil
}
