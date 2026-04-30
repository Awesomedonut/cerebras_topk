# Top-K k-NN on the WSE-2 — Design Memo

**Author:** Jin Song · **Distances:** squared L2 · **Final cycles:** 691,532

## 1. Routing topology

```
EW chain (all rows parallel, colors 0/1 ping-pong by Px parity):
  row y:  PE(L-1, y) ─0→ PE(L-2, y) ─1→ PE(L-3, y) ─0→ ... → PE(0, y)

NS chain (col 0 only, colors 4/5 ping-pong by Py parity):
  col 0:  PE(0, L-1) ─4→ PE(0, L-2) ─5→ ... → PE(0, 0)   ← result
```

Four colors, checkerboarded — every hop is point-to-point, no contention. Each hop carries `2·K` u32 wavelets: `K` interleaved `(dist_bits, idx)` pairs. `L = LAYOUT_P` is 4 when `K < rows_per_pe` (idle-PE redistribution), else host `P`. Endpoints configure only the color they actually use; intermediate PEs configure both.

## 2. Local top-K

**Distance — dot-product expansion.** `||D−q||² = ||D||² − 2·D·q + ||q||²`. Host pre-computes `||D[r]||²` once (free — D is fixed at memcpy). Per column j: one `@fmacs(dist, dist, D_col_j, −2·q[j])` with `q[j]` as scalar broadcast; final `dist += ||q||²`. Drops per-PE compute DSDs from `3·d_dim` to `d_dim + 2`.

**Top-K — three comptime paths:**

| Condition | Algorithm | Cost / element |
|---|---|---|
| `K == rows_per_pe` (SORT_ALL) | Floyd heapify + heap_sort_asc | O(K log K), ~3 cyc heapify, ~6 cyc/sift |
| `K ≥ 32` | Max-heap replace-root + sift_down | O(N · log K), ~7 cyc/elem |
| `K < 32` | Insertion sort, early-exit on full | Worst O(N · K), ~4 cyc/elem |

Heap was chosen because scalar bitonic (+267K), bottom-up mergesort (+122K), and LSD radix (+54K) all regressed. The CSL compiler optimizes heap-sift loops well; scalar alternatives carry overheads that exceed savings on this simulator.

## 3. Fabric bandwidth

Worst case is `k_large`: K=256, LAYOUT_P=2 (gate skips multi-PE). Each EW hop carries `2·K = 512` u32 wavelets; one EW hop per row × 2 rows = 1024 wavelets EW total. NS chain: 1 hop × 512 wavelets. The single most-loaded edge sees **512 wavelets**.

Across all six cases `idle_ce_cycles ≈ 50` per case. Bottleneck is **sort + compute**, not routing — `heap_sort_asc` on K=256 alone is ~203K cycles (56% of k_large) while fabric is well under 1% of total.

## 4. Tie-breaking

Define `lex_less((d, idx), (d', idx')) := d < d' ∨ (d == d' ∧ idx < idx')`. Indices are globally unique (PE-linear offset + within-PE row), so lex_less is a strict total order. Each PE produces its local top-K under lex_less; `merge_received` selects K-smallest from the union of two sorted lists under the same order.

K-min under a total order is **associative and commutative**: for any A, B, C,

  `topK(A ∪ B ∪ C) = topK(topK(A ∪ B) ∪ C) = topK(A ∪ topK(B ∪ C))`.

Therefore the EW chain yields the same K-min set regardless of the order intermediate wavelets arrive at a PE; NS composes likewise. Padded rows carry distance `1.0e20` (squared L2 → +inf) and lex-lose to every real row. **Determinism follows from the algebra, not fabric scheduling** — verified bit-identical across 3 runs.

## 5. With 2× more time

1. **Microthread-parallel `heap_sort_asc` halves on k_large** — sort two halves of K=256 concurrently, then merge. Estimated **~115K savings (~17%)** if the WSE-2 microthread API permits the dependency pattern. ~25% probability of clean success.
2. **Vectorized bitonic via `@fmins`/`@fmaxs`** — only path to attack k_large's heap_sort floor with hardware compare-swaps. First attempt deadlocked the simulator at K=256 (paired contiguous DSDs aliasing); needs deeper SDK debugging. If recovered, **~80–150K savings**.

Both high-risk. Empirical evidence: once compute is optimized (fmacs + multi-PE-gated), further gains require attacking the sort itself, which scalar code cannot.
