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

// DBSCAN is density-based clustering. Noise points get the label -1.
type DBSCAN struct {
	h    *C.struct_tpml_TpmlDbscan
	rows int
}

// NewDBSCAN returns an unfitted DBSCAN with neighbourhood radius eps and minSamples.
func NewDBSCAN(eps float64, minSamples int) (*DBSCAN, error) {
	h := C.tpml_dbscan_new(C.double(eps), C.uintptr_t(minSamples))
	if h == nil {
		return nil, errors.New("mlproven: invalid DBSCAN parameters")
	}
	d := &DBSCAN{h: h}
	runtime.SetFinalizer(d, (*DBSCAN).Close)
	return d, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (d *DBSCAN) Close() {
	if d.h != nil {
		C.tpml_dbscan_free(d.h)
		d.h = nil
	}
}

// Fit clusters the rows of x.
func (d *DBSCAN) Fit(x [][]float64) error {
	if d.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_dbscan_fit(d.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	d.rows = rows
	runtime.KeepAlive(data)
	return nil
}

// Labels returns the cluster of each training row, with -1 for noise.
func (d *DBSCAN) Labels() ([]int, error) {
	if d.h == nil {
		return nil, ErrClosed
	}
	if d.rows == 0 {
		return nil, ErrNotFitted
	}
	return fetchLabels(d.rows, func(out *C.int64_t, n C.uintptr_t) C.enum_tpml_TpmlStatus {
		return C.tpml_dbscan_labels(d.h, out, n)
	})
}

// HDBSCAN is hierarchical density-based clustering. Noise points get the label -1.
type HDBSCAN struct {
	h    *C.struct_tpml_TpmlHdbscan
	rows int
}

// NewHDBSCAN returns an unfitted HDBSCAN. A minSamples of zero means the same as minClusterSize.
func NewHDBSCAN(minClusterSize, minSamples int) (*HDBSCAN, error) {
	h := C.tpml_hdbscan_new(C.uintptr_t(minClusterSize), C.uintptr_t(minSamples))
	if h == nil {
		return nil, errors.New("mlproven: invalid HDBSCAN parameters")
	}
	m := &HDBSCAN{h: h}
	runtime.SetFinalizer(m, (*HDBSCAN).Close)
	return m, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (m *HDBSCAN) Close() {
	if m.h != nil {
		C.tpml_hdbscan_free(m.h)
		m.h = nil
	}
}

// Fit clusters the rows of x.
func (m *HDBSCAN) Fit(x [][]float64) error {
	if m.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_hdbscan_fit(m.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	m.rows = rows
	runtime.KeepAlive(data)
	return nil
}

// Labels returns the cluster of each training row, with -1 for noise.
func (m *HDBSCAN) Labels() ([]int, error) {
	if m.h == nil {
		return nil, ErrClosed
	}
	if m.rows == 0 {
		return nil, ErrNotFitted
	}
	return fetchLabels(m.rows, func(out *C.int64_t, n C.uintptr_t) C.enum_tpml_TpmlStatus {
		return C.tpml_hdbscan_labels(m.h, out, n)
	})
}
