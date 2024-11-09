package qndnn

import (
	"fmt"
	"testing"
)

func Test_Sigmoid(t *testing.T) {
	t.Run("regular", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{-4, 0.017986209962091562},
			{1, 0.7310585786300049},
			{42, 1},
		} {
			t.Run(fmt.Sprintf("%v_%v", tc.in, tc.out), func(t *testing.T) {
				out := Sigmoid(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%v', got '%v'", tc.out, out)
				}
			})
		}
	})

	t.Run("derivative", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{0.017986209962091562, 0.24997978210580288},
			{0.7310585786300049, 0.2193618640098077},
		} {
			t.Run(fmt.Sprintf("%v", tc.in), func(t *testing.T) {
				out := DerivativeSigmoid(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%.2f', got '%.2f'", tc.out, out)
				}
			})
		}
	})
}

func Test_Relu(t *testing.T) {
	t.Run("regular", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{-4, 0},
			{1, 1},
			{42, 42},
		} {
			t.Run(fmt.Sprintf("%v_%v", tc.in, tc.out), func(t *testing.T) {
				out := Relu(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%v', got '%v'", tc.out, out)
				}
			})
		}
	})

	t.Run("derivative", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{-4, 0},
			{0, 0},
			{42, 1},
		} {
			t.Run(fmt.Sprintf("%v", tc.in), func(t *testing.T) {
				out := DerivativeRelu(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%.2f', got '%.2f'", tc.out, out)
				}
			})
		}
	})
}

func Test_HyperbolicTangent(t *testing.T) {
	t.Run("regular", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{-4, -0.9993292997390669},
			{0, 0},
			{1, 0.7615941559557649},
			{42, 1},
		} {
			t.Run(fmt.Sprintf("%v_%v", tc.in, tc.out), func(t *testing.T) {
				out := HyperbolicTangent(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%v', got '%v'", tc.out, out)
				}
			})
		}
	})

	t.Run("derivative", func(t *testing.T) {
		for _, tc := range []struct{ in, out float64 }{
			{-4, 0.0013409506830260876},
			{0, 1},
			{1, 0.41997434161402614},
			{42, 0},
		} {
			t.Run(fmt.Sprintf("%v", tc.in), func(t *testing.T) {
				out := DerivativeHyperbolicTangent(tc.in)
				if out != tc.out {
					t.Errorf("failed; expected '%.100f', got '%.100f'", tc.out, out)
				}
			})
		}
	})
}

func Test_CallHelper(t *testing.T) {
	t.Run("sigmoid", func(t *testing.T) {
		f := WithSigmoid()
		n := &Neuron{}
		n = (*f)(n)
		r := n.Functions.Activation(1)
		if r != 0.7310585786300049 {
			t.Error("failed to use activation sigmoid")
		}
		r = n.Functions.Derivative(0.7310585786300049)
		if r != 0.2193618640098077 {
			t.Error("failed to use derivative sigmoid")
		}
	})

	t.Run("relu", func(t *testing.T) {
		f := WithRelu()
		n := &Neuron{}
		n = (*f)(n)
		r := n.Functions.Activation(-4)
		if r != 0 {
			t.Error("failed to use activation relu")
		}
		r = n.Functions.Derivative(0)
		if r != 0 {
			t.Error("failed to use derivative relu")
		}
	})

	t.Run("tanh", func(t *testing.T) {
		f := WithTanh()
		n := &Neuron{}
		n = (*f)(n)
		r := n.Functions.Activation(1)
		if r != 0.7615941559557649 {
			t.Error("failed to use activation tanh")
		}
		r = n.Functions.Derivative(0)
		if r != 1.0 {
			t.Error("failed to use derivative tanh")
		}
	})
}
