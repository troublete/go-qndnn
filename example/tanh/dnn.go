package main

import (
	"fmt"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	nn := qndnn.NewNetwork(&qndnn.WithTanh, 3, 16, 8, 1) // input (3), hidden1 (8), hidden2 (8), out (1)

	out, _ := nn.Output( // returns error if input is wrong dimension
		[]float64{1, 2, 3}, // input
	)
	fmt.Println(out)

	_ = nn.Train( // returns error if input or output is wrong dimension
		[]float64{1, 2, 3}, // on input ...
		[]float64{.42},     // ... expected output
		.75,                // learning rate eta
		10000,              // rounds of learning
	)

	out, _ = nn.Output([]float64{1, 2, 3})
	fmt.Println(out)
}
