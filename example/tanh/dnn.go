package main

import (
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	nn := qndnn.NewNeuralNet(qndnn.WithTanh(), 3, 2, 2, 1) // input (3), hidden1 (2), hidden2 (2), out (1)

	out, _ := nn.Output( // returns error if input is wrong dimension
		[]float64{1, 2, 3}, // input
	)
	fmt.Println(out)
	// demo out: [0.8655444560780231]

	start := time.Now()
	slog.Info("start", start.Format(time.DateTime))
	_ = nn.Train( // returns error if input or output is wrong dimension
		[]qndnn.Expectations{
			{
				[]float64{1, 2, 3}, // on input ...
				[]float64{.42},     // ... expected output
			},
			{
				[]float64{5, 2, 3}, // on input ...
				[]float64{.56},     // ... expected output
			},
		},
		.25, // learning rate eta
		qndnn.WithLoggingStrategy(os.Stdout, qndnn.ThresholdStrategy(0.001, -1)), // learn until errors are < 0.1%
	)
	slog.Info("done", "s", time.Since(start).Seconds())

	out, _ = nn.Output([]float64{1, 2, 3})
	fmt.Println(out)
	// demo out: [0.42]

	out, _ = nn.Output([]float64{5, 2, 3})
	fmt.Println(out)
	// demo out: [0.56]
}
