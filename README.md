# go-qndnn
> quick 'n' dirty neural network (for practical use)

## Introduction

This package contains a simple Go implementation for DNNs; for practical everyday-use in common use-cases. It is neither
heavily optimized to be the best DNN package around, nor does this package contain an exhaustive variety of mathematical
functions. It supports Sigmoid, Tanh and ReLU. It leverages Go primitives.

```go
nn := qndnn.NewNeuralNetwork(nil, 4, 3, 3, 1) // sigmoid is default; input (4), hidden1 (3), hidden2 (3), output (1)
// qndnn.NewNeuralNetwork(qndnn.WithRelu(), 4, 3, 3, 1) // – to use with relu
// qndnn.NewNeuralNetwork(qndnn.WithTanh(), 4, 3, 3, 1) // - to use with tanh

// to retrieve output with input values
out, err := nn.Output([]float64{1, 2, 3, 4})

// to train on expectations
err = nn.Train(
	[]Expectation{
        {
            Input: []float64{1, 2, 3, 4},
            Output: []float64{.42},
        },
    }, 
	0.01, // learning rate
	1000, // rounds
)
```
