package main

import (
	"fmt"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	nn := qndnn.NewNeuralNet(nil, 3, 3, 2, 1) // input (3), hidden1 (2), hidden2 (2), out (1)

	out, _ := nn.Output( // returns error if input is wrong dimension
		[]float64{1, 2, 3}, // input
	)
	fmt.Println(out)
	// example out: [0.7460341666363554]

	_ = nn.Train( // returns error if input or output is wrong dimension
		[]qndnn.Expectations{
			{
				[]float64{1, 2, 3}, // on input ...
				[]float64{.42},     // ... expected output
			},
		},
		.5,    // learning rate eta
		10000, // rounds of learning
	)

	out, _ = nn.Output([]float64{1, 2, 3})
	fmt.Println(out)
	// example out: [0.4200000000000005]
}
