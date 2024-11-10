package main

import (
	"fmt"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	nn := qndnn.NewNeuralNet(qndnn.WithRelu(), 3, 2, 2, 1) // input (3), hidden1 (2), hidden2 (2), out (1)

	out, _ := nn.Output( // returns error if input is wrong dimension
		[]float64{1, 2, 3}, // input
	)
	fmt.Println(out)
	// demo out: [3.1304876598996585]

	_ = nn.Train( // returns error if input or output is wrong dimension
		[]qndnn.Expectations{
			{
				[]float64{1, 2, 3}, // on input ...
				[]float64{42},      // ... expected output
			},
		},
		.0001,  // learning rate eta
		100000, // rounds of learning
	)

	out, _ = nn.Output([]float64{1, 2, 3})
	fmt.Println(out)
	// demo out: [42.06386375634217]
}
