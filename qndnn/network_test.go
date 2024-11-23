package qndnn

import (
	"encoding/base64"
	"math"
	"testing"
)

func Test_NewNetwork(t *testing.T) {
	nn := NewNeuralNet(nil, 1, 2, 1)
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
	nn := NewNeuralNet(nil, 1, 3, 1)
	_, err := nn.Output([]float64{1, 2, 3})
	if err == nil {
		t.Error("expected error, didn't get one")
	}
}

func Test_Train(t *testing.T) {
	t.Run("successful", func(t *testing.T) {
		learningRate := 0.5

		nn := NewNeuralNet(nil, 1, 2, 1)
		out, err := nn.Output([]float64{5})
		if err != nil {
			t.Error(err)
		}

		/*
		 *   w1 / o2 \ w3
		 *    o1      o4
		 *   w2 \ o3 / w4
		 *
		 *   o1 == n[0][0]
		 *   o2 == n[1][0]
		 *   o3 == n[1][1]
		 *   o4 == n[2][0]
		 */

		expected := .666
		o := out[0]
		i := nn[2][0].Input()
		deltaIn := (o - expected) * DerivativeSigmoid(i)
		w1 := nn[1][0].Inputs[0].Weight
		w2 := nn[1][1].Inputs[0].Weight
		w3 := nn[2][0].Inputs[0].Weight
		w4 := nn[2][0].Inputs[1].Weight
		o2in := nn[1][0].Input()
		o3in := nn[1][1].Input()
		o4in := nn[2][0].Input()
		o1out := nn[0][0].Value()
		o2out := nn[1][0].Value()
		o3out := nn[1][1].Value()

		err = nn.Train([]Expectations{{[]float64{5}, []float64{expected}}}, learningRate, 1)
		if err != nil {
			t.Error(err)
		}

		w3chg := learningRate * (deltaIn * 1.0 * DerivativeSigmoid(o4in)) * o2out
		w3new := w3 - w3chg
		if w3new != nn[2][0].Inputs[0].Weight {
			t.Errorf("w3 was calculated wrongly; wanted %v, got %v", w3new, nn[2][0].Inputs[0].Weight)
		}

		w4chg := learningRate * (deltaIn * 1.0 * DerivativeSigmoid(o4in)) * o3out
		w4new := w4 - w4chg
		if w4new != nn[2][0].Inputs[1].Weight {
			t.Errorf("w4 was calculated wrongly; wanted %v, got %v", w4new, nn[2][0].Inputs[1].Weight)
		}

		w2chg := learningRate * (DerivativeSigmoid(o3in) * w4 * (deltaIn * 1.0 * DerivativeSigmoid(o4in))) * o1out
		w2new := w2 - w2chg
		if w2new != nn[1][1].Inputs[0].Weight {
			t.Errorf("w2 was calculated wrongly; wanted %v, got %v", w2new, nn[1][1].Inputs[0].Weight)
		}

		w1chg := learningRate * (DerivativeSigmoid(o2in) * w3 * (deltaIn * 1.0 * DerivativeSigmoid(o4in))) * o1out
		w1new := w1 - w1chg
		if w1new != nn[1][0].Inputs[0].Weight {
			t.Errorf("w1 was calculated wrongly; wanted %v, got %v", w1new, nn[1][0].Inputs[0].Weight)
		}
	})

	t.Run("error-nous run input", func(t *testing.T) {
		nn := NewNeuralNet(nil, 1, 2, 1)
		err := nn.Train([]Expectations{{[]float64{1, 2}, []float64{1}}}, .5, 1)
		if err == nil {
			t.Error("expected error got none")
		}
	})

	t.Run("error-nous run output", func(t *testing.T) {
		nn := NewNeuralNet(nil, 1, 2, 1)
		err := nn.Train([]Expectations{{[]float64{1}, []float64{1, 2}}}, .5, 1)
		if err == nil {
			t.Error("expected error got none")
		}
	})

	t.Run("setting negative rounds", func(t *testing.T) {
		nn := NewNeuralNet(nil, 1, 2, 1)
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

func Test_Serialize(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		nn := NewNeuralNet(WithRelu(), 2, 2, 1)
		out1, err := nn.Output([]float64{2, 3})
		if err != nil {
			t.Error(err)
		}
		content, err := nn.Serialize()
		if err != nil {
			t.Error(err)
		}
		nn, err = NewNeuralNetFromSerialized(WithRelu(), content)
		if err != nil {
			t.Error(err)
		}
		out2, err := nn.Output([]float64{2, 3})
		if err != nil {
			t.Error(err)
		}

		for idx, o := range out1 {
			if o != out2[idx] {
				t.Errorf("failed to compare output #%v: wanted %v, got %v", idx, o, out2[idx])
			}
		}
	})

	t.Run("success without custom activation", func(t *testing.T) {
		nn := NewNeuralNet(nil, 2, 2, 1)
		out1, err := nn.Output([]float64{2, 3})
		if err != nil {
			t.Error(err)
		}
		content, err := nn.Serialize()
		if err != nil {
			t.Error(err)
		}
		nn, err = NewNeuralNetFromSerialized(nil, content)
		if err != nil {
			t.Error(err)
		}
		out2, err := nn.Output([]float64{2, 3})
		if err != nil {
			t.Error(err)
		}

		for idx, o := range out1 {
			if o != out2[idx] {
				t.Errorf("failed to compare output #%v: wanted %v, got %v", idx, o, out2[idx])
			}
		}
	})

	t.Run("json error serialize", func(t *testing.T) {
		nn := NewNeuralNet(nil, 1, 2, 1)
		nn[1][0].Bias = math.Inf(1)
		content, err := nn.Serialize()
		if content != "" {
			t.Error("expected no output")
		}

		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("json error serialize", func(t *testing.T) {
		nn := NewNeuralNet(nil, 1, 2, 1)
		nn[1][0].Bias = math.Inf(1)
		content, err := nn.Serialize()
		if content != "" {
			t.Error("expected no output")
		}

		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("base64 deserialize error", func(t *testing.T) {
		s := base64.StdEncoding.EncodeToString([]byte("test"))
		n, err := NewNeuralNetFromSerialized(nil, s)
		if n != nil {
			t.Error("expected no output")
		}

		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("base64 deserialize error", func(t *testing.T) {
		n, err := NewNeuralNetFromSerialized(nil, `👀`)
		if n != nil {
			t.Error("expected no output")
		}

		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("json deserialize error", func(t *testing.T) {
		n, err := NewNeuralNetFromSerialized(nil, base64.StdEncoding.EncodeToString([]byte("nothing worth while")))
		if n != nil {
			t.Error("expected no output")
		}

		if err == nil {
			t.Error("expected error")
		}
	})
}
