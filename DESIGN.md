# Top-K k-NN on the WSE-2 — Design Memo

**Author:** Jin Song · **Distances:** squared L2 · **Final cycles:** 691,534

## 1. Routing Topology — Chain Reduction

```
EW (all rows parallel, colors 0/1 ping-pong):
  row y:  PE(P-1,y) ─color_a→ PE(P-2,y) ─color_b→ … → PE(0,y)

NS (col 0 only, colors 4/5 ping-pong):
  col 0:  PE(0,P-1) ─color_a→ PE(0,P-2) ─color_b→ … → PE(0,0)  ← result
```

Four colors checkerboarded so each hop is point-to-point. Each hop carries **2K u32** — K `(dist_bits, idx)` pairs interleaved. Role flags are comptime booleans; compiler DCEs unused branches per case binary. Chain over tree: at P=4, chain = 3 hops vs tree's 2, but fabric is <1% of total — simpler routing outweighs the saved hop.

## 2. Per-PE Algorithm

**Distance — dot-product expansion.** `||D-q||² = ||D||² − 2·D·q + ||q||²`. Host pre-computes `||D[r]||²` (free on device). Per column j: `@fmacs(dist, dist, D_col_j, -2·q[j])` — one DSD call vs three. Drops DSD count from `3·d_dim` to `d_dim + 2`.

**Cycles per element:** dominant loop is @fmacs at ~1 cycle/element × rows_per_pe extent. Baseline: 32 cols × 128 rows ≈ 4K cycles, ~31 cycles/row.

**Top-K — three comptime paths:** heap-sort for K==rows_per_pe (O(K log K)), max-heap for K≥32 (O(N·log K)), insertion sort for K<32 (worst O(N·K)). Compiler DCEs unused paths.

**Multi-PE redistribution (gated on `K < rows_per_pe`).** Expands to 4×4 grid to use idle PEs, reducing per-PE rows 4×. Skips on k_large to preserve SORT_ALL fast path.

**SRAM (k_large worst):** D 16KB + buffers 12KB + dist+norms 2KB = **~28KB of 48KB**.

## 3. Fabric Bandwidth

Any active edge carries exactly **2K wavelets**. Worst case (k_large, K=256): **512 wavelets per edge**. Whole-grid: EW = P×(P−1)×2K, NS = (P−1)×2K.

**Bottleneck is not fabric.** k_large: heap_sort_asc ~203K cycles (56%), fabric ~2K. Fabric <0.5% in all cases. The kernel is sort-bound on k_large, compute-bound elsewhere.

## 4. Tie-Breaking Proof

`lex_less(d1, idx1, d2, idx2)` defines a strict total order (distance ascending, index ascending). Indices are globally unique. `merge_received()` selects K-smallest from two sorted lists under this order. K-min under a total order is associative + commutative — **output is uniquely determined regardless of reduction order or wavelet timing**. Determinism follows from the algebra, not fabric scheduling. Padded rows use `1.0e20` → +inf, always lex-loses.

## 5. What 2× More Time Would Buy

1. **Microthread-parallel heap_sort on k_large.** k_large's heap_sort_asc is 203K cycles (56% of total). Sort two halves of K=256 concurrently on WSE-2's two microthreads, then merge. Estimated ~115K savings if the microthread API supports concurrent array access.

2. **Vectorized bitonic sort via `@fmins`/`@fmaxs`.** Replace scalar heap_sort with a DSD-pipelined bitonic network. This is the only path to break below the scalar comparison floor on k_large. Initial attempt deadlocked the simulator; needs deeper SDK debugging to identify the constraint.
