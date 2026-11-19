# tpt-ml-proven (Go)

Go bindings for KMeans and PCA, through the C ABI in `ffi/c` (cgo).
Module: `github.com/tpt-solutions/tpt-ml-proven/ffi/go`. Requires Go 1.21 or later and a C
compiler for cgo.

Build the native library first:

```sh
cargo build --release --manifest-path ../c/Cargo.toml                       # Linux, macOS
cargo build --release --manifest-path ../c/Cargo.toml --target x86_64-pc-windows-gnu   # Windows, MinGW
go test ./...
```

On Windows the static library must be the GNU build, because cgo links with MinGW. Install the
target with `rustup target add x86_64-pc-windows-gnu`, and point cgo at a MinGW `gcc`.

```go
km, err := mlproven.NewKMeans(2, mlproven.WithSeed(1))
if err != nil { /* ... */ }
defer km.Close()
if err := km.Fit(x); err != nil { /* ... */ }   // x is [][]float64, one slice per row
labels, _ := km.Labels()
centers, _ := km.Centers()

p, _ := mlproven.NewPCA(2)
defer p.Close()
p.Fit(x)
z, _ := p.Transform(x)
```

- Data is `[][]float64`. Rows must be non-empty and of equal length.
- Errors wrap the native status. `errors.Is(err, mlproven.ErrNotFitted)` checks for a call before `Fit`.
- Estimators hold native memory. Call `Close`. A finalizer is a fallback only.
- The Phase 8 estimators are not yet in Go. The C ABI covers all of them.
