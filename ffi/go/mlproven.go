// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

// Package mlproven provides the estimators of tpt-ml-proven for Go, through the C ABI in ffi/c.
//
// Data is a slice of rows; every row must have the same length. Estimators hold native
// memory, so call Close when done (a finalizer is a fallback, not a guarantee).
//
// Build the static library before building this package:
//
//	cargo build --release --manifest-path ../c/Cargo.toml
//
// On Windows the library is built for the GNU target, as cgo links with MinGW (see
// link_windows.go).
package mlproven

/*
#cgo CFLAGS: -I${SRCDIR}/../c/include
#include <stdlib.h>
#include "tpt_ml_proven.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// ErrNotFitted is returned by methods that need a fitted estimator.
var ErrNotFitted = errors.New("mlproven: estimator is not fitted; call Fit first")

// ErrClosed is returned by methods called after Close.
var ErrClosed = errors.New("mlproven: estimator is closed")

// StatusError is a failure reported by the native library.
type StatusError struct {
	Code    int
	Message string
}

func (e *StatusError) Error() string { return "mlproven: " + e.Message }

// Is reports whether target is ErrNotFitted for a not-fitted status.
func (e *StatusError) Is(target error) bool {
	return target == ErrNotFitted && e.Code == int(C.tpml_TpmlStatus_NotFitted)
}

// callC runs f on a locked OS thread, so the thread-local error message read afterwards
// belongs to f's call.
func callC(f func() C.enum_tpml_TpmlStatus) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	st := f()
	if st == C.tpml_TpmlStatus_Ok {
		return nil
	}
	return &StatusError{Code: int(st), Message: C.GoString(C.tpml_last_error_message())}
}

// flatten checks that x is a non-empty rectangular matrix and copies it row-major.
func flatten[T float32 | float64](x [][]T) ([]T, int, int, error) {
	rows := len(x)
	if rows == 0 {
		return nil, 0, 0, errors.New("mlproven: input has no rows")
	}
	cols := len(x[0])
	if cols == 0 {
		return nil, 0, 0, errors.New("mlproven: input rows are empty")
	}
	data := make([]T, 0, rows*cols)
	for i, row := range x {
		if len(row) != cols {
			return nil, 0, 0, fmt.Errorf("mlproven: row %d has %d values, want %d", i, len(row), cols)
		}
		data = append(data, row...)
	}
	return data, rows, cols, nil
}

// unflatten reshapes a row-major buffer into rows of length cols.
func unflatten(flat []float64, cols int) [][]float64 {
	out := make([][]float64, len(flat)/cols)
	for i := range out {
		out[i] = flat[i*cols : (i+1)*cols : (i+1)*cols]
	}
	return out
}

func toInts(buf []C.uintptr_t) []int {
	out := make([]int, len(buf))
	for i, v := range buf {
		out[i] = int(v)
	}
	return out
}

func dataPtr(data []float64) *C.double {
	return (*C.double)(unsafe.Pointer(&data[0]))
}

func dataPtr32(data []float32) *C.float {
	return (*C.float)(unsafe.Pointer(&data[0]))
}

// checkTargets checks that y has one value per row of the matrix.
func checkTargets(y []float64, rows int) error {
	if len(y) != rows {
		return fmt.Errorf("mlproven: %d targets for %d rows", len(y), rows)
	}
	return nil
}

// fetchLabels runs a native call that writes one int64 label per row, and converts them to int.
func fetchLabels(rows int, fill func(out *C.int64_t, n C.uintptr_t) C.enum_tpml_TpmlStatus) ([]int, error) {
	buf := make([]C.int64_t, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return fill(&buf[0], C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	out := make([]int, rows)
	for i, v := range buf {
		out[i] = int(v)
	}
	return out, nil
}

// ---------------------------------------------------------------- KMeans

// KMeansOption configures a KMeans estimator.
type KMeansOption func(*kmeansConfig)

type kmeansConfig struct {
	maxIter int
	tol     float64
	seed    uint64
	threads int
}

// WithMaxIter sets the iteration limit (default 300).
func WithMaxIter(n int) KMeansOption { return func(c *kmeansConfig) { c.maxIter = n } }

// WithTol sets the convergence tolerance (default 1e-4).
func WithTol(tol float64) KMeansOption { return func(c *kmeansConfig) { c.tol = tol } }

// WithSeed sets the seed for the k-means++ start (default 0).
func WithSeed(seed uint64) KMeansOption { return func(c *kmeansConfig) { c.seed = seed } }

// WithThreads sets the worker count (default 1). Results do not depend on it.
func WithThreads(n int) KMeansOption { return func(c *kmeansConfig) { c.threads = n } }

// KMeans is k-means clustering by Lloyd's algorithm, in f64.
type KMeans struct {
	h    *C.struct_tpml_TpmlKMeans
	k    int
	rows int
	cols int
}

// NewKMeans returns an unfitted KMeans with k clusters.
func NewKMeans(k int, opts ...KMeansOption) (*KMeans, error) {
	cfg := kmeansConfig{maxIter: 300, tol: 1e-4, seed: 0, threads: 1}
	for _, opt := range opts {
		opt(&cfg)
	}
	h := C.tpml_kmeans_new(
		C.uintptr_t(k),
		C.uintptr_t(cfg.maxIter),
		C.double(cfg.tol),
		C.uint64_t(cfg.seed),
		C.uintptr_t(cfg.threads),
	)
	if h == nil {
		return nil, errors.New("mlproven: invalid KMeans parameters")
	}
	km := &KMeans{h: h, k: k}
	runtime.SetFinalizer(km, (*KMeans).Close)
	return km, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (km *KMeans) Close() {
	if km.h != nil {
		C.tpml_kmeans_free(km.h)
		km.h = nil
	}
}

// Fit clusters the rows of x.
func (km *KMeans) Fit(x [][]float64) error {
	if km.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_fit(km.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	km.rows, km.cols = rows, cols
	runtime.KeepAlive(data)
	return nil
}

// Labels returns the cluster of each training row.
func (km *KMeans) Labels() ([]int, error) {
	if km.h == nil {
		return nil, ErrClosed
	}
	buf := make([]C.uintptr_t, km.rows)
	if len(buf) == 0 {
		return nil, ErrNotFitted
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_labels(km.h, &buf[0], C.uintptr_t(len(buf)))
	}); err != nil {
		return nil, err
	}
	return toInts(buf), nil
}

// Predict returns the nearest centre for each row of x.
func (km *KMeans) Predict(x [][]float64) ([]int, error) {
	if km.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	buf := make([]C.uintptr_t, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_predict(km.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), &buf[0], C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return toInts(buf), nil
}

// Centers returns the fitted centres, one row per cluster.
func (km *KMeans) Centers() ([][]float64, error) {
	if km.h == nil {
		return nil, ErrClosed
	}
	if km.cols == 0 {
		return nil, ErrNotFitted
	}
	flat := make([]float64, km.k*km.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_centers(km.h, dataPtr(flat), C.uintptr_t(len(flat)))
	}); err != nil {
		return nil, err
	}
	return unflatten(flat, km.cols), nil
}

// Inertia returns the sum of squared distances from each training row to its centre.
func (km *KMeans) Inertia() (float64, error) {
	if km.h == nil {
		return 0, ErrClosed
	}
	var out C.double
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_inertia(km.h, &out)
	}); err != nil {
		return 0, err
	}
	return float64(out), nil
}

// NIter returns the number of Lloyd iterations the last Fit used.
func (km *KMeans) NIter() (int, error) {
	if km.h == nil {
		return 0, ErrClosed
	}
	var out C.uintptr_t
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_kmeans_n_iter(km.h, &out)
	}); err != nil {
		return 0, err
	}
	return int(out), nil
}

// ---------------------------------------------------------------- PCA

// PCA is principal component analysis, in f64.
type PCA struct {
	h          *C.struct_tpml_TpmlPca
	components int
	cols       int
}

// NewPCA returns an unfitted PCA keeping nComponents components.
func NewPCA(nComponents int) (*PCA, error) {
	h := C.tpml_pca_new(C.uintptr_t(nComponents))
	if h == nil {
		return nil, errors.New("mlproven: invalid PCA parameters")
	}
	p := &PCA{h: h, components: nComponents}
	runtime.SetFinalizer(p, (*PCA).Close)
	return p, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (p *PCA) Close() {
	if p.h != nil {
		C.tpml_pca_free(p.h)
		p.h = nil
	}
}

// Fit computes the principal axes of the rows of x.
func (p *PCA) Fit(x [][]float64) error {
	if p.h == nil {
		return ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return err
	}
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_pca_fit(p.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols))
	}); err != nil {
		return err
	}
	p.cols = cols
	runtime.KeepAlive(data)
	return nil
}

// Transform projects the rows of x onto the components.
func (p *PCA) Transform(x [][]float64) ([][]float64, error) {
	if p.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows*p.components)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_pca_transform(p.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return unflatten(out, p.components), nil
}

// Components returns the principal axes, one row per component.
func (p *PCA) Components() ([][]float64, error) {
	if p.h == nil {
		return nil, ErrClosed
	}
	if p.cols == 0 {
		return nil, ErrNotFitted
	}
	flat := make([]float64, p.components*p.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_pca_components(p.h, dataPtr(flat), C.uintptr_t(len(flat)))
	}); err != nil {
		return nil, err
	}
	return unflatten(flat, p.cols), nil
}

// InverseTransform maps rows of scores (one column per component) back to the feature space.
func (p *PCA) InverseTransform(z [][]float64) ([][]float64, error) {
	if p.h == nil {
		return nil, ErrClosed
	}
	if p.cols == 0 {
		return nil, ErrNotFitted
	}
	data, rows, cols, err := flatten(z)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows*p.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_pca_inverse_transform(p.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return unflatten(out, p.cols), nil
}
