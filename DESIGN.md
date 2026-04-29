# Top-K k-NN on the WSE-2 — Design Memo

**Start:** Monday April 27th, 2026 · **Architecture:** chain-reduction merge · **Distances:** squared L2

## 1. Routing Topology

```
  EW chain (all rows in parallel, colors 0/1 ping-pong):
  row y:  PE(P-1,y) ─color_a→ PE(P-2,y) ─color_b→ … → PE(0,y)

  NS chain (column 0 only, colors 4/5 ping-pong):
  col 0:  PE(0,P-1) ─color_a→ PE(0,P-2) ─color_b→ … → PE(0,0)  ← result
```

Four colors, checkerboard: each PE receives on one color and sends on another. Each hop carries **2K u32 wavelets** — K `(dist_bits, idx)` pairs packed interleaved. Per-PE role flags (`is_ew_sender`, etc.) are comptime booleans; the compiler dead-code-eliminates unused branches.

## 2. Local Top-K Algorithm

**Distance — column-wise strided DSDs.** Instead of one row at a time (rows_per_pe × d_dim-extent DSD ops), I process one column at a time (d_dim × rows_per_pe-extent ops). A strided DSD walks column j; `@fsubs` broadcasts `q[j]`, `@fmuls` squares, `@fadds` accumulates into `dist_buf`. Fewer setups, larger extents, better pipeline utilization.

**Top-K selection — three comptime paths:**

| Condition | Algorithm | Complexity |
|---|---|---|
| `K == rows_per_pe` | Floyd heapify + heap-sort | O(K log K) |
| `K ≥ 32` | Max-heap replace-root + sift-down | O(rows_per_pe · log K) |
| `K < 32` | Insertion sort, early-exit | Worst-case O(rows_per_pe · K) |

Gated by `const SORT_ALL` / `const USE_HEAP` — compiler eliminates unused paths.

**Cycles per element:** For baseline (128 rows, d=32), column-wise distance: 32 iters × 3 DSD ops × extent 128 ≈ 12K cycles, or **~3 cycles per D element**. Insertion sort adds ~2K. Total ≈ 14K per PE, ~109 cycles/row.

## 3. Fabric Bandwidth

Any active edge carries exactly **2K wavelets**; unused edges carry zero. Whole-grid totals: EW = P rows × (P−1) hops × 2K; NS = (P−1) hops × 2K. Baseline (P=4, K=16): EW = 4×3×32 = 384, NS = 3×32 = 96, total **480 wavelets**. k_large (P=2, K=256): EW = 2×1×512 = 1024, NS = 1×512 = 512, total **1536 wavelets**.

**Bottleneck is compute.** Baseline measured 132K cycles; fabric is ~480 wavelets. k_large measured 373K cycles; fabric is ~1.5K wavelets. Fabric is never on the critical path.

## 4. Tie-Breaking Proof

`lex_less(d1, idx1, d2, idx2)` defines a strict total order (distance ascending, index ascending). Global indices are unique. `merge_received()` selects the K-smallest elements from the union of two sorted top-K lists under this total order. Since taking the K-smallest under a total order is both associative and commutative, **any reduction order yields the same result** — the output does not depend on wavelet arrival order. The fixed chain topology additionally guarantees deterministic execution timing, making the output bit-identical across runs.

Uneven padding: host fills invalid rows with `1.0e20`, whose squared distance overflows to +inf — always lex-loses.

## 5. What 2× More Time Would Buy

1. **Pipelined chain merge.** Overlap the rightmost PE's send with downstream PEs' compute. Hides one chain-hop latency behind computation — roughly **5–10%** on k_large.

2. **Mixed-precision distances.** fp16 storage + fp32 accumulation halves DSD cycles, but fp16 error grows with magnitude. Squared L2 over random fp32 vectors can reach hundreds, so mixed precision needs per-case validation. Estimated **15–25%** on compute if precision holds.
