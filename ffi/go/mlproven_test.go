// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

import (
	"errors"
	"math"
	"testing"
)

// Two blobs of three points each, two features.
func blobs() [][]float64 {
	return [][]float64{
		{0, 0}, {0.1, 0}, {0, 0.1},
		{10, 10}, {10.1, 10}, {10, 10.1},
	}
}

func TestKMeansRecoversBlobs(t *testing.T) {
	km, err := NewKMeans(2, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if err := km.Fit(blobs()); err != nil {
		t.Fatal(err)
	}
	labels, err := km.Labels()
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != labels[1] || labels[0] != labels[2] || labels[3] != labels[4] || labels[0] == labels[3] {
		t.Fatalf("labels %v do not separate the blobs", labels)
	}
	predicted, err := km.Predict([][]float64{{0.05, 0.05}, {10.05, 10.05}})
	if err != nil {
		t.Fatal(err)
	}
	if predicted[0] != labels[0] || predicted[1] != labels[3] {
		t.Fatalf("predict %v disagrees with labels %v", predicted, labels)
	}
	centers, err := km.Centers()
	if err != nil {
		t.Fatal(err)
	}
	if len(centers) != 2 || len(centers[0]) != 2 {
		t.Fatalf("centers have shape %dx%d, want 2x2", len(centers), len(centers[0]))
	}
	inertia, err := km.Inertia()
	if err != nil {
		t.Fatal(err)
	}
	if !(inertia >= 0) || math.IsNaN(inertia) {
		t.Fatalf("inertia %v is not a non-negative number", inertia)
	}
}

func TestPCAShapesAndOrthonormalAxes(t *testing.T) {
	x := [][]float64{
		{1, 2, 3}, {2, 4, 6.5}, {3, 5.5, 9}, {4, 8, 12.5}, {5, 9.5, 15},
	}
	p, err := NewPCA(2)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Fit(x); err != nil {
		t.Fatal(err)
	}
	z, err := p.Transform(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(z) != 5 || len(z[0]) != 2 {
		t.Fatalf("transform has shape %dx%d, want 5x2", len(z), len(z[0]))
	}
	axes, err := p.Components()
	if err != nil {
		t.Fatal(err)
	}
	if len(axes) != 2 || len(axes[0]) != 3 {
		t.Fatalf("components have shape %dx%d, want 2x3", len(axes), len(axes[0]))
	}
	for i := range axes {
		norm := 0.0
		for _, v := range axes[i] {
			norm += v * v
		}
		if math.Abs(norm-1) > 1e-9 {
			t.Fatalf("component %d has squared norm %v, want 1", i, norm)
		}
	}
}

func TestUnfittedAccessIsNotFitted(t *testing.T) {
	km, err := NewKMeans(2)
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if _, err := km.Labels(); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("Labels before Fit: %v, want ErrNotFitted", err)
	}
	if _, err := km.Centers(); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("Centers before Fit: %v, want ErrNotFitted", err)
	}
}

func TestBadInputIsRejectedBeforeCrossingTheBoundary(t *testing.T) {
	km, err := NewKMeans(2)
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if err := km.Fit([][]float64{{1, 2}, {3}}); err == nil {
		t.Fatal("ragged rows were accepted")
	}
	if err := km.Fit(nil); err == nil {
		t.Fatal("empty input was accepted")
	}
	if err := km.Fit([][]float64{{1, math.NaN()}, {2, 3}}); err == nil {
		t.Fatal("NaN input was accepted")
	}
}

func TestInvalidParametersAreRejected(t *testing.T) {
	if _, err := NewKMeans(0); err == nil {
		t.Fatal("k = 0 was accepted")
	}
	if _, err := NewPCA(0); err == nil {
		t.Fatal("n_components = 0 was accepted")
	}
}

func TestCloseIsIdempotentAndBlocksUse(t *testing.T) {
	km, err := NewKMeans(2)
	if err != nil {
		t.Fatal(err)
	}
	km.Close()
	km.Close()
	if err := km.Fit(blobs()); !errors.Is(err, ErrClosed) {
		t.Fatalf("Fit after Close: %v, want ErrClosed", err)
	}
}
