// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

/*
#cgo CFLAGS: -I${SRCDIR}/../c/include
#include <stdlib.h>
#include "tpt_ml_proven.h"
*/
import "C"

import "runtime"

// FitFloat32 clusters the rows of x held as float32. The model is single precision, which halves
// the memory the training data needs. Predict, Labels and Centers still take and return float64,
// and their input is narrowed to float32 where the model needs it.
func (km *KMeans) FitFloat32(x [][]float32) error {
	if km.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_fit_f32(km.h, dataPtr32(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	km.rows, km.cols = rows, cols
	runtime.KeepAlive(data)
	return nil
}

// FitFloat32 computes the principal axes of the rows of x held as float32. The model is single
// precision. Transform still takes float64 input, which is narrowed.
func (p *PCA) FitFloat32(x [][]float32) error {
	if p.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_pca_fit_f32(p.h, dataPtr32(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	p.cols = cols
	runtime.KeepAlive(data)
	return nil
}
