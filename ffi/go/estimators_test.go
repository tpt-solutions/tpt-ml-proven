// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

import (
	"errors"
	"math"
	"testing"
)

func TestDBSCANSeparatesBlobsAndMarksNoise(t *testing.T) {
	x := append(blobs(), []float64{50, -50})
	d, err := NewDBSCAN(0.5, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if err := d.Fit(x); err != nil {
		t.Fatal(err)
	}
	labels, err := d.Labels()
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != labels[1] || labels[0] != labels[2] || labels[3] != labels[4] || labels[0] == labels[3] {
		t.Fatalf("labels %v do not separate the blobs", labels)
	}
	if labels[6] != -1 {
		t.Fatalf("isolated point has label %d, want -1", labels[6])
	}
}

func TestHDBSCANSeparatesBlobs(t *testing.T) {
	m, err := NewHDBSCAN(3, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.Fit(blobs()); err != nil {
		t.Fatal(err)
	}
	labels, err := m.Labels()
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != labels[1] || labels[0] != labels[2] || labels[3] != labels[4] || labels[0] == labels[3] {
		t.Fatalf("labels %v do not separate the blobs", labels)
	}
}

func TestTruncatedSVDShapesAndDescendingValues(t *testing.T) {
	x := [][]float64{
		{1, 2, 3}, {2, 4, 6}, {3, 5, 9}, {4, 8, 12}, {5, 9, 15},
	}
	s, err := NewTruncatedSVD(2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Fit(x); err != nil {
		t.Fatal(err)
	}
	z, err := s.Transform(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(z) != 5 || len(z[0]) != 2 {
		t.Fatalf("transform has shape %dx%d, want 5x2", len(z), len(z[0]))
	}
	values, err := s.SingularValues()
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 2 || values[0] < values[1] {
		t.Fatalf("singular values %v are not two values, largest first", values)
	}
}

func TestNMFFactorsAreNonNegativeAndShaped(t *testing.T) {
	x := [][]float64{
		{1, 0, 2}, {2, 0, 4}, {0, 3, 1}, {0, 6, 2}, {1, 3, 3},
	}
	n, err := NewNMF(2, WithNMFSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()
	if err := n.Fit(x); err != nil {
		t.Fatal(err)
	}
	w, err := n.Transform(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 5 || len(w[0]) != 2 {
		t.Fatalf("W has shape %dx%d, want 5x2", len(w), len(w[0]))
	}
	h, err := n.Components()
	if err != nil {
		t.Fatal(err)
	}
	if len(h) != 2 || len(h[0]) != 3 {
		t.Fatalf("H has shape %dx%d, want 2x3", len(h), len(h[0]))
	}
	for _, row := range append(w, h...) {
		for _, v := range row {
			if v < 0 {
				t.Fatalf("factor value %v is negative", v)
			}
		}
	}
}

// A line y = 2x + 1 with a small offset to make the fit non-trivial.
func linearData() ([][]float64, []float64) {
	x := [][]float64{{0}, {1}, {2}, {3}, {4}, {5}}
	y := []float64{1, 3, 5, 7, 9, 11}
	return x, y
}

func TestRidgeRecoversLine(t *testing.T) {
	x, y := linearData()
	r, err := NewRidge(1e-9)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := r.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	coef, err := r.Coef()
	if err != nil {
		t.Fatal(err)
	}
	intercept, err := r.Intercept()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(coef[0]-2) > 1e-3 || math.Abs(intercept-1) > 1e-3 {
		t.Fatalf("coef %v intercept %v, want 2 and 1", coef, intercept)
	}
	pred, err := r.Predict([][]float64{{10}})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(pred[0]-21) > 1e-2 {
		t.Fatalf("predict(10) = %v, want 21", pred[0])
	}
}

func TestLassoZeroesWithLargePenalty(t *testing.T) {
	x, y := linearData()
	l, err := NewLasso(1000, 1000, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if err := l.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	coef, err := l.Coef()
	if err != nil {
		t.Fatal(err)
	}
	if coef[0] != 0 {
		t.Fatalf("coef %v, want exactly zero under a large penalty", coef)
	}
}

func TestElasticNetFitsLine(t *testing.T) {
	x, y := linearData()
	e, err := NewElasticNet(1e-4, 0.5, 5000, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if err := e.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	coef, err := e.Coef()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(coef[0]-2) > 0.05 {
		t.Fatalf("coef %v, want about 2", coef)
	}
}

func TestLogisticSeparatesClasses(t *testing.T) {
	x := [][]float64{{0}, {0.5}, {1}, {4}, {4.5}, {5}}
	y := []float64{0, 0, 0, 1, 1, 1}
	l, err := NewLogistic(10, 1000, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if err := l.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	n, err := l.NClasses()
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("NClasses = %d, want 2", n)
	}
	pred, err := l.Predict([][]float64{{0.2}, {4.8}})
	if err != nil {
		t.Fatal(err)
	}
	if pred[0] != 0 || pred[1] != 1 {
		t.Fatalf("predict %v, want [0 1]", pred)
	}
	proba, err := l.PredictProba([][]float64{{0.2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(proba[0]) != 2 || math.Abs(proba[0][0]+proba[0][1]-1) > 1e-9 {
		t.Fatalf("probabilities %v do not sum to one over two classes", proba[0])
	}
}

func TestStumpForestSeparatesClasses(t *testing.T) {
	x := [][]float64{{0}, {0.5}, {1}, {4}, {4.5}, {5}}
	y := []float64{0, 0, 0, 1, 1, 1}
	f, err := NewStumpForest(25, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Fit(x, y); err != nil {
		t.Fatal(err)
	}
	pred, err := f.Predict([][]float64{{0.2}, {4.8}})
	if err != nil {
		t.Fatal(err)
	}
	if pred[0] != 0 || pred[1] != 1 {
		t.Fatalf("predict %v, want [0 1]", pred)
	}
}

func TestTargetCountIsChecked(t *testing.T) {
	x, _ := linearData()
	r, err := NewRidge(1)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if err := r.Fit(x, []float64{1, 2}); err == nil {
		t.Fatal("mismatched target count was accepted")
	}
}

func TestUnfittedNewEstimatorsReportNotFitted(t *testing.T) {
	r, err := NewRidge(1)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := r.Coef(); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("Coef before Fit: %v, want ErrNotFitted", err)
	}
	d, err := NewDBSCAN(0.5, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if _, err := d.Labels(); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("Labels before Fit: %v, want ErrNotFitted", err)
	}
}

func TestInvalidNewEstimatorParametersAreRejected(t *testing.T) {
	if _, err := NewDBSCAN(0, 2); err == nil {
		t.Fatal("eps = 0 was accepted")
	}
	if _, err := NewHDBSCAN(1, 0); err == nil {
		t.Fatal("min_cluster_size = 1 was accepted")
	}
	if _, err := NewRidge(-1); err == nil {
		t.Fatal("negative alpha was accepted")
	}
	if _, err := NewStumpForest(0, 0); err == nil {
		t.Fatal("zero estimators were accepted")
	}
}
