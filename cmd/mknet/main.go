package main

import (
	"bytes"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/troublete/go-qndnn/qndnn"
)

func main() {
	activation := flag.String("activation", "sigmoid", "activation function to use (must be 'sigmoid|tanh|relu')")
	layers := flag.String("layers", "", "csv value for layer sizes (e.g. input=4,hidden1=4,hidden2=4,output=2 == '4, 4, 4, 1'")
	name := flag.String("name", "mynet.qndnn", "name of the net, to be used as filename")
	flag.Parse()

	layerSizes := []int{}
	for _, l := range strings.Split(*layers, ",") {
		layerSize := strings.TrimSpace(l)
		ls, err := strconv.ParseFloat(layerSize, 64)
		if err != nil {
			slog.Error("can't parse layer size", "in", layerSize, "err", err)
			os.Exit(1)
		}

		layerSizes = append(layerSizes, int(ls))
	}

	slog.Info("creating net", "layer configuration", layerSizes, "activation", activation)
	f := qndnn.WithSigmoid()
	switch *activation {
	case "tanh":
		f = qndnn.WithTanh()
	case "relu":
		f = qndnn.WithRelu()
	}

	nn := qndnn.NewNeuralNet(f, layerSizes...)
	serialized, err := nn.Serialize()
	if err != nil {
		slog.Error("can't serialize net", "err", err)
		os.Exit(1)
	}

	cwd, err := os.Getwd()
	if err != nil {
		slog.Error("can't fetch cwd", "err", err)
		os.Exit(1)
	}

	buf := bytes.NewBufferString(serialized)
	path := fmt.Sprintf("%s/%s", cwd, *name)
	err = os.WriteFile(path, buf.Bytes(), os.ModePerm)
	if err != nil {
		slog.Error("couldn't write file", "err", err)
		os.Exit(1)
	}
	slog.Info("done", "file", path)
}
