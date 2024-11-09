package qndnn

import (
	"fmt"
	"testing"
)

func Test_Sigmoid(t *testing.T) {
	for _, tc := range []struct{ in, out float64 }{
		{-4, 0.017986209962091562},
		{1, 0.7310585786300049},
		{42, 1},
	} {
		t.Run(fmt.Sprintf("%v_%v", tc.in, tc.out), func(t *testing.T) {
			out := sigmoid(tc.in)
			if out != tc.out {
				t.Errorf("failed; expected '%v', got '%v'", tc.out, out)
			}
		})
	}
}

func Test_DerivativeSigmoid(t *testing.T) {
	for _, tc := range []struct{ in, out float64 }{
		{0.017986209962091562, 0.24997978210580288},
		{0.7310585786300049, 0.2193618640098077},
	} {
		t.Run(fmt.Sprintf("%v", tc.in), func(t *testing.T) {
			out := derivativeSigmoid(tc.in)
			if out != tc.out {
				t.Errorf("failed; expected '%.2f', got '%.2f'", tc.in, out)
			}
		})
	}
}
