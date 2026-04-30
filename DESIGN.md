# Top-K k-NN on the WSE-2 — Design Memo

**Author:** Jin Song · **Start:** Monday April 27th, 2026 · **Distances:** squared L2

## 1. Routing Topology — Chain Reduction

```
  EW (all rows parallel, colors 0/1 ping-pong):
  row y:  PE(P-1,y) ─color_a→ PE(P-2,y) ─color_b→ … → PE(0,y)

  NS (col 0 only, colors 4/5 ping-pong):
  col 0:  PE(0,P-1) ─color_a→ PE(0,P-2) ─color_b→ … → PE(0,0)  ← result
```

Four colors, checkerboard: each PE receives on one color and sends on another. Each hop carries **2K u32** — K `(dist_bits, idx)` pairs packed interleaved. Role flags (`is_ew_sender`, etc.) are comptime booleans from layout.csl; compiler dead-code-eliminates unused branches per case binary.

Chain over tree: for P=4, chain = 3 serial hops vs tree's 2, but fabric is <1% of total cycles. The simplicity of chain merge (one code path, no multi-level synchronization) outweighs the extra hop.

## 2. Local Top-K — Adaptive Comptime Dispatch

**Distance: column-wise strided DSDs.** d_dim iterations of rows_per_pe-extent ops instead of rows_per_pe iterations of d_dim-extent ops — fewer DSD setups, longer pipeline runs.

**Top-K selection:**

| Condition | Algorithm | Complexity |
|---|---|---|
| `K == rows_per_pe` | Floyd heapify + heap-sort | O(K log K) |
| `K ≥ 32` | Max-heap replace-root + sift | O(N · log K) |
| `K < 32` | Insertion sort, early-exit | Worst-case O(N · K) |

**Memory budget (k_large, worst case):** D_shard 16KB + local/merge/send/recv 10KB + dist_buf+tmp_buf 2KB + diff 64B = **~28KB of 48KB.**

## 3. Measured Cycles + Bandwidth

| Case | P | K | d | rows/PE | Cycles | Fabric wavelets |
|---|---|---|---|---|---|---|
| baseline | 4 | 16 | 32 | 128 | 132,620 | 480 |
| k_eq_1 | 2 | 1 | 32 | 256 | 91,106 | 4 |
| k_large | 2 | 256 | 16 | 256 | 373,662 | 1,536 |
| uneven | 4 | 16 | 32 | 64 | 81,366 | 480 |
| all_equal | 2 | 16 | 16 | 256 | 73,447 | 64 |
| duplicates | 2 | 8 | 16 | 256 | 68,585 | 32 |

Fabric wavelets formula: EW = P × (P−1) × 2K; NS = (P−1) × 2K. Any active edge carries exactly 2K wavelets.

**Where cycles go (k_large estimate):** ~12K distance DSD ops, ~64K heap_sort_asc (255 sift-downs × ~250 cycles each), ~2K fabric — useful work ~78K of 373K measured (21%). The remaining ~295K is DSD setup overhead (48 invocations), async `@fmovs` round-trips, and simulator dispatch. The kernel is **setup-and-coordination-bound**, not compute-bound. Fabric is never the bottleneck (<0.5% of cycles).

## 4. Tie-Breaking Proof

`lex_less(d1, idx1, d2, idx2)` defines a strict total order (distance ascending, index ascending). Global indices are unique. `merge_received()` selects the K-smallest from the union of two sorted lists under this order. Taking K-smallest under a total order is associative and commutative — **the output is uniquely determined regardless of reduction order or wavelet arrival timing.** Correctness follows from the algebra, not from fabric timing.

Uneven padding: host fills invalid rows with `1.0e20`, whose squared distance overflows to +inf — always lex-loses.

## 5. What Didn't Work + What 2× More Time Would Buy

**Attempted and reverted:** K==1 scalar fast path (regressed k_eq_1 by +5K cycles — compiler optimizes K=1 loops better than manual scalar tracking). @map for all d_dim values (regressed d_dim=16 cases by 12% — per-element call overhead exceeds savings at small d).

**With more time:** (1) Mixed-precision distances (fp16 storage + fp32 accumulation) — needs per-case tolerance validation, gains uncertain without measurement. (2) Overlap rightmost PE's chain send with downstream compute — ceiling is ~1K cycles (the chain's total fabric cost), so <1% on k_large.
