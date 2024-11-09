package qndnn

import (
	"fmt"
	"testing"
)

func Test_NewNetwork(t *testing.T) {
	nn := NewNetwork(nil, 1, 2, 1)
	input := nn[0][0]
	if *input.Preset != 1 || input.Value() != 1 {
		t.Error("expected input to be 1")
	}

	if input.Bias != 0 {
		t.Error("expected input bias to be 0")
	}

	if len(nn[0]) != 1 || len(nn[1]) != 2 || len(nn[2]) != 1 {
		t.Error("failed to create layers in requested size")
	}

	for _, n := range nn[1] {
		if n.Bias == 0 || n.Inputs[0].Weight == 0 {
			t.Error("failed to set bias and weight in hidden layer")
		}
	}

	for _, n := range nn[2] {
		if n.Bias == 0 || n.Inputs[0].Weight == 0 {
			t.Error("failed to set bias and weight int output layer")
		}
	}

	out, err := nn.Output([]float64{1})
	if err != nil {
		t.Error(err)
	}

	// hidden layer
	n2w := nn[1][0].Inputs[0].Weight
	n2b := nn[1][0].Bias

	n3w := nn[1][1].Inputs[0].Weight
	n3b := nn[1][1].Bias

	// output
	n4w1 := nn[2][0].Inputs[0].Weight
	n4w2 := nn[2][0].Inputs[1].Weight

	n4b := nn[2][0].Bias

	n2v := Sigmoid((n2w * 1) + n2b)
	n3v := Sigmoid((n3w * 1) + n3b)

	n41 := Sigmoid(((n4w1 * n2v) + (n4w2 * n3v)) + n4b)

	if n41 != out[0] {
		t.Errorf("network failed, expected '%v' got '%v", n41, out)
	}
}

func Test_Output(t *testing.T) {
	nn := NewNetwork(nil, 1, 3, 1)
	_, err := nn.Output([]float64{1, 2, 3})
	if err == nil {
		t.Error("expected error, didn't get one")
	}
}

func Test_Train(t *testing.T) {
	t.Run("successful", func(t *testing.T) {
		learningRate := 0.5

		nn := NewNetwork(nil, 1, 2, 1)
		out, err := nn.Output([]float64{5})
		if err != nil {
			t.Error(err)
		}

		bw1 := nn[1][0].Inputs[0].Weight
		w1r := nn[1][0].Value()
		vw1 := nn[1][0].Inputs[0].N.Value()
		bw2 := nn[1][1].Inputs[0].Weight
		w2r := nn[1][1].Value()
		vw2 := nn[1][1].Inputs[0].N.Value()

		bw3 := nn[2][0].Inputs[0].Weight
		vw3 := nn[2][0].Inputs[0].N.Value()
		bw4 := nn[2][0].Inputs[1].Weight
		vw4 := nn[2][0].Inputs[1].N.Value()

		err = nn.Train([]Expectations{{[]float64{5}, []float64{.666}}}, learningRate, 1)
		if err != nil {
			t.Error(err)
		}

		aw1 := nn[1][0].Inputs[0].Weight
		aw2 := nn[1][1].Inputs[0].Weight
		aw3 := nn[2][0].Inputs[0].Weight
		aw4 := nn[2][0].Inputs[1].Weight

		w4out := out[0]
		w4err := -(.666 - w4out)
		w4der := DerivativeSigmoid(w4out)
		w4change := w4err * w4der * vw4
		calcw4 := bw4 - (w4change * learningRate)
		if fmt.Sprintf("%.10f", aw4) != fmt.Sprintf("%.10f", calcw4) {
			t.Errorf("w4 adjustent incorrect, should be '%v' is '%v", calcw4, aw4)
		}

		w3out := out[0]
		w3err := -(.666 - w3out)
		w3der := DerivativeSigmoid(w3out)
		w3change := w3err * w3der * vw3
		calcw3 := bw3 - (w3change * learningRate)
		if fmt.Sprintf("%.10f", aw3) != fmt.Sprintf("%.10f", calcw3) {
			t.Errorf("w3 adjustent incorrect, should be '%v' is '%v", calcw3, aw3)
		}

		w2out := w2r
		w2err := -((.666 * bw4) - w2out)
		w2der := DerivativeSigmoid(w2out)
		w2change := w2err * w2der * vw2
		calcw2 := bw2 - (w2change * learningRate)
		if fmt.Sprintf("%.10f", aw2) != fmt.Sprintf("%.10f", calcw2) {
			t.Errorf("w2 adjustent incorrect, should be '%v' is '%v", calcw2, aw2)
		}

		w1out := w1r
		w1err := -((.666 * bw3) - w1out)
		w1der := DerivativeSigmoid(w1out)
		w1change := w1err * w1der * vw1
		calcw1 := bw1 - (w1change * learningRate)
		if fmt.Sprintf("%.10f", aw1) != fmt.Sprintf("%.10f", calcw1) {
			t.Errorf("w1 adjustent incorrect, should be '%v' is '%v", calcw1, aw1)
		}
	})

	t.Run("error-nous run input", func(t *testing.T) {
		nn := NewNetwork(nil, 1, 2, 1)
		err := nn.Train([]Expectations{{[]float64{1, 2}, []float64{1}}}, .5, 1)
		if err == nil {
			t.Error("expected error got none")
		}
	})

	t.Run("error-nous run output", func(t *testing.T) {
		nn := NewNetwork(nil, 1, 2, 1)
		err := nn.Train([]Expectations{{[]float64{1}, []float64{1, 2}}}, .5, 1)
		if err == nil {
			t.Error("expected error got none")
		}
	})

	t.Run("setting negative rounds", func(t *testing.T) {
		nn := NewNetwork(nil, 1, 2, 1)
		err := nn.Train([]Expectations{{[]float64{1}, []float64{1}}}, .5, -5)
		if err != nil {
			t.Error(err)
		}
	})
}

func Test_UpdateWeight(t *testing.T) {
	i := &Input{
		Weight:        .55,
		PendingChange: -.001,
	}
	i.UpdateWeight()

	if i.Weight != (.55-(-.001)) || i.PendingChange != 0.0 {
		t.Error("updating weight failed")
	}
}
