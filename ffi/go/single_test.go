// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

import (
	"math"
	"testing"
)

func blobs32() [][]float32 {
	return [][]float32{
		{0, 0}, {0.1, 0}, {0, 0.1},
		{10, 10}, {10.1, 10}, {10, 10.1},
	}
}

func TestKMeansFitFloat32SeparatesBlobs(t *testing.T) {
	km, err := NewKMeans(2, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if err := km.FitFloat32(blobs32()); err != nil {
		t.Fatal(err)
	}
	labels, err := km.Labels()
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != labels[1] || labels[0] != labels[2] || labels[3] != labels[4] || labels[0] == labels[3] {
		t.Fatalf("labels %v do not separate the blobs", labels)
	}
	// Predict takes float64, which the single-precision model narrows.
	predicted, err := km.Predict([][]float64{{0.05, 0.05}, {10.05, 10.05}})
	if err != nil {
		t.Fatal(err)
	}
	if predicted[0] != labels[0] || predicted[1] != labels[3] {
		t.Fatalf("predict %v disagrees with labels %v", predicted, labels)
	}
}

func TestPCAFitFloat32MatchesFloat64(t *testing.T) {
	x64 := [][]float64{
		{1, 2, 3}, {2, 4, 6.5}, {3, 5.5, 9}, {4, 8, 12.5}, {5, 9.5, 15},
	}
	x32 := make([][]float32, len(x64))
	for i, row := range x64 {
		x32[i] = make([]float32, len(row))
		for j, v := range row {
			x32[i][j] = float32(v)
		}
	}

	p64, err := NewPCA(2)
	if err != nil {
		t.Fatal(err)
	}
	defer p64.Close()
	if err := p64.Fit(x64); err != nil {
		t.Fatal(err)
	}
	want, err := p64.Transform(x64)
	if err != nil {
		t.Fatal(err)
	}

	p32, err := NewPCA(2)
	if err != nil {
		t.Fatal(err)
	}
	defer p32.Close()
	if err := p32.FitFloat32(x32); err != nil {
		t.Fatal(err)
	}
	got, err := p32.Transform(x64)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		for j := range want[i] {
			// Components are defined up to sign, so compare magnitudes.
			if d := math.Abs(math.Abs(got[i][j]) - math.Abs(want[i][j])); d > 1e-3 {
				t.Fatalf("transform[%d][%d] = %v, want %v (difference %v)", i, j, got[i][j], want[i][j], d)
			}
		}
	}
}

func TestFitFloat32RejectsRaggedInput(t *testing.T) {
	km, err := NewKMeans(2)
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if err := km.FitFloat32([][]float32{{1, 2}, {3}}); err == nil {
		t.Fatal("ragged float32 rows were accepted")
	}
}
