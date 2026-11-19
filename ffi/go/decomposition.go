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

// TruncatedSVD is a linear projection onto the leading singular vectors, in f64. Unlike PCA it
// does not centre the data, so it suits sparse-style inputs.
type TruncatedSVD struct {
	h          *C.struct_tpml_TpmlTruncatedSvd
	components int
	cols       int
}

// NewTruncatedSVD returns an unfitted TruncatedSVD keeping nComponents vectors.
func NewTruncatedSVD(nComponents int) (*TruncatedSVD, error) {
	h := C.tpml_truncated_svd_new(C.uintptr_t(nComponents))
	if h == nil {
		return nil, errors.New("mlproven: invalid TruncatedSVD parameters")
	}
	s := &TruncatedSVD{h: h, components: nComponents}
	runtime.SetFinalizer(s, (*TruncatedSVD).Close)
	return s, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (s *TruncatedSVD) Close() {
	if s.h != nil {
		C.tpml_truncated_svd_free(s.h)
		s.h = nil
	}
}

// Fit computes the leading singular vectors of the rows of x.
func (s *TruncatedSVD) Fit(x [][]float64) error {
	if s.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_truncated_svd_fit(s.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	s.cols = cols
	runtime.KeepAlive(data)
	return nil
}

// Transform projects the rows of x onto the components.
func (s *TruncatedSVD) Transform(x [][]float64) ([][]float64, error) {
	if s.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows*s.components)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_truncated_svd_transform(s.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return unflatten(out, s.components), nil
}

// SingularValues returns the singular values, largest first.
func (s *TruncatedSVD) SingularValues() ([]float64, error) {
	if s.h == nil {
		return nil, ErrClosed
	}
	if s.cols == 0 {
		return nil, ErrNotFitted
	}
	out := make([]float64, s.components)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_truncated_svd_singular_values(s.h, dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// NMF is non-negative matrix factorisation X ≈ W·H, in f64. Inputs must be non-negative.
type NMF struct {
	h          *C.struct_tpml_TpmlNmf
	components int
	cols       int
}

// NMFOption configures an NMF estimator.
type NMFOption func(*nmfConfig)

type nmfConfig struct {
	maxIter int
	tol     float64
	seed    uint64
}

// WithNMFMaxIter sets the iteration limit (default 200).
func WithNMFMaxIter(n int) NMFOption { return func(c *nmfConfig) { c.maxIter = n } }

// WithNMFTol sets the convergence tolerance (default 1e-4).
func WithNMFTol(tol float64) NMFOption { return func(c *nmfConfig) { c.tol = tol } }

// WithNMFSeed sets the seed for the random initialisation (default 0).
func WithNMFSeed(seed uint64) NMFOption { return func(c *nmfConfig) { c.seed = seed } }

// NewNMF returns an unfitted NMF with nComponents factors.
func NewNMF(nComponents int, opts ...NMFOption) (*NMF, error) {
	cfg := nmfConfig{maxIter: 200, tol: 1e-4, seed: 0}
	for _, opt := range opts {
		opt(&cfg)
	}
	h := C.tpml_nmf_new(C.uintptr_t(nComponents), C.uintptr_t(cfg.maxIter), C.double(cfg.tol), C.uint64_t(cfg.seed))
	if h == nil {
		return nil, errors.New("mlproven: invalid NMF parameters")
	}
	n := &NMF{h: h, components: nComponents}
	runtime.SetFinalizer(n, (*NMF).Close)
	return n, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (n *NMF) Close() {
	if n.h != nil {
		C.tpml_nmf_free(n.h)
		n.h = nil
	}
}

// Fit factorises the non-negative rows of x.
func (n *NMF) Fit(x [][]float64) error {
	if n.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_nmf_fit(n.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	n.cols = cols
	runtime.KeepAlive(data)
	return nil
}

// Transform returns the factor W for the rows of x, one row per input row.
func (n *NMF) Transform(x [][]float64) ([][]float64, error) {
	if n.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows*n.components)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_nmf_transform(n.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return unflatten(out, n.components), nil
}

// Components returns the learned factors H, one row per component.
func (n *NMF) Components() ([][]float64, error) {
	if n.h == nil {
		return nil, ErrClosed
	}
	if n.cols == 0 {
		return nil, ErrNotFitted
	}
	flat := make([]float64, n.components*n.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_nmf_components(n.h, dataPtr(flat), C.uintptr_t(len(flat)))
	}); err != nil {
		return nil, err
	}
	return unflatten(flat, n.cols), nil
}
