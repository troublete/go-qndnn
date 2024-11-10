package main

import (
	"bytes"
	"flag"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	activation := flag.String("activation", "sigmoid", "activation function to use (must be 'sigmoid|tanh|relu')")
	file := flag.String("file", "./mynet.qndnn", "file path to the stored qndnn file")
	input := flag.String("input", "", "input in csv form")
	flag.Parse()

	content, err := os.ReadFile(*file)
	if err != nil {
		slog.Error("couldn't read file", "err", err)
		os.Exit(1)
	}

	in := []float64{}
	for _, v := range strings.Split(*input, ",") {
		tv := strings.TrimSpace(v)
		pv, err := strconv.ParseFloat(tv, 64)
		if err != nil {
			slog.Error("failed to parse float", "err", err)
			os.Exit(1)
		}
		in = append(in, pv)
	}

	f := qndnn.WithSigmoid()
	switch *activation {
	case "tanh":
		f = qndnn.WithTanh()
	case "relu":
		f = qndnn.WithRelu()
	}

	buf := bytes.NewBuffer(content)
	nn, err := qndnn.NewNeuralNetFromSerialized(f, buf.String())
	if err != nil {
		slog.Error("couldn't read network", "err", err)
		os.Exit(1)
	}

	out, err := nn.Output(in)
	if err != nil {
		slog.Error("failed to generate output", "err", err)
		os.Exit(1)
	}

	slog.Info("output", "v", out)
}
