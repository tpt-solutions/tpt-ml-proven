// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

/*
#cgo CFLAGS: -I${SRCDIR}/../c/include
#include <stdlib.h>
#include "tpt_ml_proven.h"
*/
import "C"

import (
	"errors"
	"runtime"
)

// StumpForest is a bagged ensemble of single-split decision stumps, for classification.
type StumpForest struct {
	h *C.struct_tpml_TpmlStumpForest
}

// NewStumpForest returns an unfitted forest of nEstimators stumps, seeded by seed.
func NewStumpForest(nEstimators int, seed uint64) (*StumpForest, error) {
	h := C.tpml_stump_forest_new(C.uintptr_t(nEstimators), C.uint64_t(seed))
	if h == nil {
		return nil, errors.New("mlproven: invalid StumpForest parameters")
	}
	f := &StumpForest{h: h}
	runtime.SetFinalizer(f, (*StumpForest).Close)
	return f, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (f *StumpForest) Close() {
	if f.h != nil {
		C.tpml_stump_forest_free(f.h)
		f.h = nil
	}
}

// Fit trains on the rows of x and their class labels y.
func (f *StumpForest) Fit(x [][]float64, y []float64) error {
	if f.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := checkTargets(y, rows); err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_stump_forest_fit(f.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(y))
	}); err != nil {
		return err
	}
	runtime.KeepAlive(data)
	runtime.KeepAlive(y)
	return nil
}

// Predict returns the predicted class label for each row of x.
func (f *StumpForest) Predict(x [][]float64) ([]float64, error) {
	if f.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_stump_forest_predict(f.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return out, nil
}
