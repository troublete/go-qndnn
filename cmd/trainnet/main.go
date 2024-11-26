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
	activation := flag.String("activation", "relu", "activation function to use (must be 'sigmoid|tanh|relu')")
	file := flag.String("file", "./mynet.qndnn", "file path to the stored qndnn file")
	input := flag.String("input", "", "input in csv form")
	output := flag.String("expected", "", "expected output in csv form")
	rounds := flag.Int("n", 1024, "number of rounds to learn")
	learningRate := flag.Float64("learning-rate", 0.5, "")
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

	out := []float64{}
	for _, v := range strings.Split(*output, ",") {
		tv := strings.TrimSpace(v)
		pv, err := strconv.ParseFloat(tv, 64)
		if err != nil {
			slog.Error("failed to parse float", "err", err)
			os.Exit(1)
		}
		out = append(out, pv)
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

	err = nn.Train([]qndnn.Expectations{
		{
			Input:  in,
			Output: out,
		},
	}, *learningRate, qndnn.RoundStrategy(*rounds))
	if err != nil {
		slog.Error("couldn't train network", "err", err)
		os.Exit(1)
	}

	ser, err := nn.Serialize()
	if err != nil {
		slog.Error("couldn't serialize", "err", err)
		os.Exit(1)
	}

	outBuf := bytes.NewBufferString(ser)
	err = os.WriteFile(*file, outBuf.Bytes(), os.ModePerm)
	if err != nil {
		slog.Error("couldn't write file", "err", err)
		os.Exit(1)
	}

	slog.Info("done")
}
