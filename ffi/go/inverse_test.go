// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 TPT Solutions

package mlproven

import (
	"errors"
	"math"
	"testing"
)

func TestPCAInverseTransformRecoversRankOneData(t *testing.T) {
	// The second column is twice the first, so one component reconstructs every row.
	x := [][]float64{{1, 2}, {2, 4}, {3, 6}, {4, 8}}
	p, err := NewPCA(1)
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
	back, err := p.InverseTransform(z)
	if err != nil {
		t.Fatal(err)
	}
	if len(back) != len(x) || len(back[0]) != 2 {
		t.Fatalf("inverse has shape %dx%d, want 4x2", len(back), len(back[0]))
	}
	for i := range x {
		for j := range x[i] {
			if math.Abs(back[i][j]-x[i][j]) > 1e-9 {
				t.Fatalf("back[%d][%d] = %v, want %v", i, j, back[i][j], x[i][j])
			}
		}
	}
}

func TestPCAInverseTransformBeforeFitIsNotFitted(t *testing.T) {
	p, err := NewPCA(1)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if _, err := p.InverseTransform([][]float64{{1}}); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("InverseTransform before Fit: %v, want ErrNotFitted", err)
	}
}

func TestKMeansNIterIsReportedAfterFit(t *testing.T) {
	km, err := NewKMeans(2, WithSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	defer km.Close()
	if _, err := km.NIter(); !errors.Is(err, ErrNotFitted) {
		t.Fatalf("NIter before Fit: %v, want ErrNotFitted", err)
	}
	if err := km.Fit(blobs()); err != nil {
		t.Fatal(err)
	}
	n, err := km.NIter()
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Fatalf("NIter = %d, want at least 1", n)
	}
}
