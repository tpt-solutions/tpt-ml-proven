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

// Ridge is linear regression with an L2 penalty, in f64.
type Ridge struct {
	h    *C.struct_tpml_TpmlRidge
	cols int
}

// NewRidge returns an unfitted Ridge with penalty alpha.
func NewRidge(alpha float64) (*Ridge, error) {
	h := C.tpml_ridge_new(C.double(alpha))
	if h == nil {
		return nil, errors.New("mlproven: invalid Ridge parameters")
	}
	r := &Ridge{h: h}
	runtime.SetFinalizer(r, (*Ridge).Close)
	return r, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (r *Ridge) Close() {
	if r.h != nil {
		C.tpml_ridge_free(r.h)
		r.h = nil
	}
}

// Fit regresses y on the rows of x.
func (r *Ridge) Fit(x [][]float64, y []float64) error {
	if r.h == nil {
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
		return C.tpml_ridge_fit(r.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(y))
	}); err != nil {
		return err
	}
	r.cols = cols
	runtime.KeepAlive(data)
	runtime.KeepAlive(y)
	return nil
}

// Predict returns the predicted value for each row of x.
func (r *Ridge) Predict(x [][]float64) ([]float64, error) {
	if r.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_ridge_predict(r.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return out, nil
}

// Coef returns the fitted coefficients, one per feature.
func (r *Ridge) Coef() ([]float64, error) {
	if r.h == nil {
		return nil, ErrClosed
	}
	if r.cols == 0 {
		return nil, ErrNotFitted
	}
	out := make([]float64, r.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_ridge_coef(r.h, dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Intercept returns the fitted intercept.
func (r *Ridge) Intercept() (float64, error) {
	if r.h == nil {
		return 0, ErrClosed
	}
	if r.cols == 0 {
		return 0, ErrNotFitted
	}
	var out C.double
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_ridge_intercept(r.h, &out)
	}); err != nil {
		return 0, err
	}
	return float64(out), nil
}

// Lasso is linear regression with an L1 penalty, in f64.
type Lasso struct {
	h    *C.struct_tpml_TpmlLasso
	cols int
}

// NewLasso returns an unfitted Lasso with penalty alpha, coordinate-descent limit maxIter and
// tolerance tol.
func NewLasso(alpha float64, maxIter int, tol float64) (*Lasso, error) {
	h := C.tpml_lasso_new(C.double(alpha), C.uintptr_t(maxIter), C.double(tol))
	if h == nil {
		return nil, errors.New("mlproven: invalid Lasso parameters")
	}
	l := &Lasso{h: h}
	runtime.SetFinalizer(l, (*Lasso).Close)
	return l, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (l *Lasso) Close() {
	if l.h != nil {
		C.tpml_lasso_free(l.h)
		l.h = nil
	}
}

// Fit regresses y on the rows of x.
func (l *Lasso) Fit(x [][]float64, y []float64) error {
	if l.h == nil {
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
		return C.tpml_lasso_fit(l.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(y))
	}); err != nil {
		return err
	}
	l.cols = cols
	runtime.KeepAlive(data)
	runtime.KeepAlive(y)
	return nil
}

// Predict returns the predicted value for each row of x.
func (l *Lasso) Predict(x [][]float64) ([]float64, error) {
	if l.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_lasso_predict(l.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return out, nil
}

// Coef returns the fitted coefficients, one per feature. Lasso sets many to exactly zero.
func (l *Lasso) Coef() ([]float64, error) {
	if l.h == nil {
		return nil, ErrClosed
	}
	if l.cols == 0 {
		return nil, ErrNotFitted
	}
	out := make([]float64, l.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_lasso_coef(l.h, dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Intercept returns the fitted intercept.
func (l *Lasso) Intercept() (float64, error) {
	if l.h == nil {
		return 0, ErrClosed
	}
	if l.cols == 0 {
		return 0, ErrNotFitted
	}
	var out C.double
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_lasso_intercept(l.h, &out)
	}); err != nil {
		return 0, err
	}
	return float64(out), nil
}

// ElasticNet is linear regression with a mix of L1 and L2 penalties, in f64.
type ElasticNet struct {
	h    *C.struct_tpml_TpmlElasticNet
	cols int
}

// NewElasticNet returns an unfitted ElasticNet with penalty alpha and mixing l1Ratio in [0, 1]
// (1 is pure Lasso, 0 is pure ridge-like shrinkage).
func NewElasticNet(alpha, l1Ratio float64, maxIter int, tol float64) (*ElasticNet, error) {
	h := C.tpml_elastic_net_new(C.double(alpha), C.double(l1Ratio), C.uintptr_t(maxIter), C.double(tol))
	if h == nil {
		return nil, errors.New("mlproven: invalid ElasticNet parameters")
	}
	e := &ElasticNet{h: h}
	runtime.SetFinalizer(e, (*ElasticNet).Close)
	return e, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (e *ElasticNet) Close() {
	if e.h != nil {
		C.tpml_elastic_net_free(e.h)
		e.h = nil
	}
}

// Fit regresses y on the rows of x.
func (e *ElasticNet) Fit(x [][]float64, y []float64) error {
	if e.h == nil {
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
		return C.tpml_elastic_net_fit(e.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(y))
	}); err != nil {
		return err
	}
	e.cols = cols
	runtime.KeepAlive(data)
	runtime.KeepAlive(y)
	return nil
}

// Predict returns the predicted value for each row of x.
func (e *ElasticNet) Predict(x [][]float64) ([]float64, error) {
	if e.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_elastic_net_predict(e.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return out, nil
}

// Coef returns the fitted coefficients, one per feature.
func (e *ElasticNet) Coef() ([]float64, error) {
	if e.h == nil {
		return nil, ErrClosed
	}
	if e.cols == 0 {
		return nil, ErrNotFitted
	}
	out := make([]float64, e.cols)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_elastic_net_coef(e.h, dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// Intercept returns the fitted intercept.
func (e *ElasticNet) Intercept() (float64, error) {
	if e.h == nil {
		return 0, ErrClosed
	}
	if e.cols == 0 {
		return 0, ErrNotFitted
	}
	var out C.double
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_elastic_net_intercept(e.h, &out)
	}); err != nil {
		return 0, err
	}
	return float64(out), nil
}

// Logistic is logistic regression, binary or multinomial, in f64.
type Logistic struct {
	h *C.struct_tpml_TpmlLogistic
}

// NewLogistic returns an unfitted logistic regression with inverse regularisation strength c.
func NewLogistic(c float64, maxIter int, tol float64) (*Logistic, error) {
	h := C.tpml_logistic_new(C.double(c), C.uintptr_t(maxIter), C.double(tol))
	if h == nil {
		return nil, errors.New("mlproven: invalid logistic regression parameters")
	}
	l := &Logistic{h: h}
	runtime.SetFinalizer(l, (*Logistic).Close)
	return l, nil
}

// Close releases the native estimator. It is safe to call more than once.
func (l *Logistic) Close() {
	if l.h != nil {
		C.tpml_logistic_free(l.h)
		l.h = nil
	}
}

// Fit trains on the rows of x and their class labels y. Two distinct labels give a binary
// model; more give a multinomial one.
func (l *Logistic) Fit(x [][]float64, y []float64) error {
	if l.h == nil {
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
		return C.tpml_logistic_fit(l.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(y))
	}); err != nil {
		return err
	}
	runtime.KeepAlive(data)
	runtime.KeepAlive(y)
	return nil
}

// NClasses returns the number of classes seen in training.
func (l *Logistic) NClasses() (int, error) {
	if l.h == nil {
		return 0, ErrClosed
	}
	var out C.uintptr_t
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_logistic_n_classes(l.h, &out)
	}); err != nil {
		return 0, err
	}
	return int(out), nil
}

// Predict returns the predicted class label for each row of x.
func (l *Logistic) Predict(x [][]float64) ([]float64, error) {
	if l.h == nil {
		return nil, ErrClosed
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_logistic_predict(l.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(rows))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return out, nil
}

// PredictProba returns, for each row of x, the probability of each class. Classes are in
// ascending order.
func (l *Logistic) PredictProba(x [][]float64) ([][]float64, error) {
	if l.h == nil {
		return nil, ErrClosed
	}
	classes, err := l.NClasses()
	if err != nil {
		return nil, err
	}
	data, rows, cols, err := flatten(x)
	if err != nil {
		return nil, err
	}
	out := make([]float64, rows*classes)
	if err := callC(func() C.enum_tpml_TpmlStatus {
		return C.tpml_logistic_predict_proba(l.h, dataPtr(data), C.uintptr_t(rows), C.uintptr_t(cols), dataPtr(out), C.uintptr_t(len(out)))
	}); err != nil {
		return nil, err
	}
	runtime.KeepAlive(data)
	return unflatten(out, classes), nil
}
