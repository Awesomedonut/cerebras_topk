# Top-K k-NN on the WSE-2 — Design Memo

**Author:** Jin Song · **Distances:** squared L2 · **Final cycles:** 691,534

## 1. Routing Topology — Chain Reduction

```
EW (all rows parallel, colors 0/1 ping-pong):
  row y:  PE(P-1,y) ─color_a→ PE(P-2,y) ─color_b→ … → PE(0,y)

NS (col 0 only, colors 4/5 ping-pong):
  col 0:  PE(0,P-1) ─color_a→ PE(0,P-2) ─color_b→ … → PE(0,0)  ← result
```

Four colors checkerboarded so each hop is point-to-point. Each hop carries 2K u32 — K `(dist_bits, idx)` pairs interleaved. Role flags are comptime booleans; compiler DCEs unused branches per case binary. Chain over tree: at P=4, chain = 3 hops vs tree's 2, but fabric is <1% of total — simpler routing outweighs the saved hop.

## 2. Per-PE Algorithm

**Distance — column-wise dot-product expansion.** Using `||D-q||² = ||D||² − 2·D·q + ||q||²`:
- Host pre-computes `||D[r]||²` (free on device — D is fixed at memcpy time).
- Per column j: `@fmacs(dist, dist, D_col_j, -2·q[j])` — one DSD call per column with q[j] as scalar broadcast.
- Final: `dist += ||q||²` (one scalar add).

This drops per-PE compute DSD count from `3·d_dim` (sub + sqr + accumulate) to `d_dim + 2`, saving ~10% on compute-dominated cases.

**Top-K selection — three comptime paths:**

| Condition | Algorithm | Complexity |
|---|---|---|
| `K == rows_per_pe` | Floyd heapify + heap-sort | O(K log K) |
| `K ≥ 32` | Max-heap replace-root + sift | O(N · log K) |
| `K < 32` | Insertion sort, early-exit | Worst O(N · K) |

**Multi-PE redistribution (gated on `K < rows_per_pe`).** Layout uses 4×4 grid when the gate fires, else PxP. Reduces per-PE work 4× on cases that would otherwise leave 12 of 16 PEs idle. Gate skipped on k_large because going to 4×4 there breaks the SORT_ALL fast path — heap_sort cost is set by K, not rows; +200K of extra chain hop overhead overwhelms compute savings.

**SRAM budget (k_large worst case):** D_shard 16KB + local/merge/send/recv 10KB + dist_buf 1KB + row_norms 1KB + q 64B = **~28KB of 48KB**.

## 3. Measured Cycles + Bandwidth

| Case | P | K | d | rows/PE | Cycles | Layout |
|---|---|---|---|---|---|---|
| baseline | 4 | 16 | 32 | 128 | 121,471 | 4×4 |
| k_eq_1 | 2 | 1 | 32 | 256→64 | 44,489 | 4×4 (multi-PE) |
| k_large | 2 | 256 | 16 | 256 | 359,534 | 2×2 (gate skips) |
| uneven | 4 | 16 | 32 | 64 | 76,554 | 4×4 |
| all_equal | 2 | 16 | 16 | 256→64 | 51,577 | 4×4 (multi-PE) |
| duplicates | 2 | 8 | 16 | 256→64 | 37,909 | 4×4 (multi-PE) |
| **TOTAL** | | | | | **691,534** | |

**Where cycles go (k_large, the bottleneck case):** compute ~3K, **heap_sort_asc ~203K (56%)**, chain ~2K, simulator/dispatch overhead ~150K. The kernel is sort-bound on k_large, compute-bound elsewhere. Fabric never bottlenecks: `idle_ce_cycles ≈ 50` per case across all six.

## 4. Tie-Breaking Proof

`lex_less(d1, idx1, d2, idx2)` defines a strict total order. Indices are globally unique (PE-linear offset + within-PE row). `merge_received()` selects K-smallest from a union of two sorted lists under this order. K-min under a total order is associative + commutative — **output is uniquely determined regardless of reduction order or wavelet timing**. Determinism follows from the algebra, not from fabric scheduling. Padded rows carry distance `1.0e20`; squared L2 overflows to +inf and lex-loses to every real row.

## 5. What Didn't Work + What 2× More Time Would Buy

**11 attempts, 2 wins, 9 reverts (full log: EXPERIMENTS.md).** Scalar sort alternatives (bitonic +267K, mergesort +122K, radix +54K) all regressed — heap_sort_asc is the scalar floor on WSE-2 and no scalar restructuring beats it. fp16 distance broke tie-break on random data (np.array_equal fails). Vectorized bitonic via `@fmins`/`@fmaxs` deadlocked the sim. Ungated multi-PE regressed k_large (+117K) by losing SORT_ALL; K_LOCAL+multi-PE regressed (+33K) because 4 extra chain hops × ~50K each exceeded the sort savings. DSR pre-loading was neutral — the compiler already optimizes tight loops with constant DSDs.

**With more time:** (1) **Microthread-parallel heap_sort halves** on k_large — sort two halves of K=256 concurrently then merge. ~115K savings if WSE-2 microthread API supports it. (2) **Vectorized bitonic with `@fmins`/`@fmaxs`** — only path to attack k_large's heap_sort floor; needs deeper SDK debugging to avoid the sim deadlock.
