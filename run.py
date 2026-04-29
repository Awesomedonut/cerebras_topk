#!/usr/bin/env cs_python
"""Host driver for the Top-K k-NN kernel.

Usage:
  cs_python run.py --name <build_dir> --case <case_name>

Prints "PASS: <case_name>" on success.
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

# Search for reference.py: same dir, parent dir, or challenge root
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here.parent, _here.parent / "cerebras-kernel-challenge"]:
    if (_candidate / "reference.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from reference import topk_reference, ALL_CASES  # noqa: E402

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime      # noqa: E402
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType   # noqa: E402
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder      # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="cslc output directory")
parser.add_argument("--case", required=True, help="test case name")
parser.add_argument("--cmaddr", default=None, help="IP:port for CS system")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load compile params from out.json
# ---------------------------------------------------------------------------
print(f"[1/7] Loading compile params from {args.name}/out.json ...")
with open(f"{args.name}/out.json", encoding="utf-8") as f:
    compile_data = json.load(f)
params = compile_data["params"]
P = int(params["P"])
d_dim = int(params["d_dim"])
rows_per_pe = int(params["rows_per_pe"])
K = int(params["K"])

print(f"       Case: {args.case}")
print(f"       P={P}  d_dim={d_dim}  rows_per_pe={rows_per_pe}  K={K}")
print(f"       PE grid: {P}x{P} = {P*P} PEs")
print(f"       Total PE slots: {P*P*rows_per_pe} rows")
print(f"       Sort path: {'HEAPIFY + HEAP SORT (K == rows_per_pe)' if K >= rows_per_pe else 'MAX-HEAP (K>=32)' if K >= 32 else 'INSERTION SORT (K<32)'}")

# ---------------------------------------------------------------------------
# Generate test case (matches reference.py exactly)
# ---------------------------------------------------------------------------
print(f"[2/7] Generating test case '{args.case}' ...")
CASE_MAP = {
    "baseline":   0,
    "k_eq_1":     1,
    "k_large":    2,
    "uneven":     3,
    "all_equal":  4,
    "duplicates": 5,
}
assert args.case in CASE_MAP, f"Unknown case: {args.case}"
case = ALL_CASES[CASE_MAP[args.case]]()
D = case["D"]
q = case["q"]
N = D.shape[0]

print(f"       D shape: {D.shape}  (N={N}, d={d_dim})")
print(f"       q shape: {q.shape}")
print(f"       q[:5] = {q[:5].tolist()}")

# ---------------------------------------------------------------------------
# Pad D to fill P*P*rows_per_pe rows. Padded rows get huge values so their
# squared-L2 distance is effectively +inf and they never enter top-K.
# ---------------------------------------------------------------------------
total_rows = P * P * rows_per_pe
padded_count = total_rows - N
D_padded = np.full((total_rows, d_dim), 1.0e20, dtype=np.float32)
D_padded[:N] = D

print(f"       Padded: {N} real rows + {padded_count} padding rows = {total_rows} total")
if padded_count > 0:
    print(f"       Padding rows filled with 1e20 (squared dist → +inf)")

# Show PE shard assignment
print(f"       PE shard assignment (ROW_MAJOR: PE linear idx = py*P + px):")
for py in range(min(P, 2)):
    for px in range(min(P, 2)):
        start = (py * P + px) * rows_per_pe
        end = start + rows_per_pe
        valid = min(rows_per_pe, max(0, N - start))
        print(f"         PE({px},{py}): rows [{start}..{end-1}], {valid}/{rows_per_pe} valid")
    if P > 2:
        print(f"         ... ({P-2} more columns)")
if P > 2:
    print(f"       ... ({P-2} more rows)")

# ---------------------------------------------------------------------------
# Set up runner
# ---------------------------------------------------------------------------
print(f"[3/7] Creating SdkRuntime from {args.name} ...")
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

D_sym    = runner.get_id("D_shard")
q_sym    = runner.get_id("q")
idx_sym  = runner.get_id("final_idxs")
dist_sym = runner.get_id("final_dists")

runner.load()
runner.run()
print(f"       Simulator loaded and running")

# ---------------------------------------------------------------------------
# H2D: copy D shards to all PEs (ROW_MAJOR: PE linear idx = py*P + px)
# ---------------------------------------------------------------------------
print(f"[4/7] H2D: copying D shards to {P*P} PEs ({rows_per_pe * d_dim} floats/PE) ...")
t0 = time.time()
runner.memcpy_h2d(
    D_sym,
    D_padded.flatten(),
    0, 0, P, P,
    rows_per_pe * d_dim,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)
t1 = time.time()
print(f"       D copied: {D_padded.nbytes:,} bytes in {t1-t0:.2f}s")

# H2D: broadcast q to every PE
print(f"       H2D: copying q to {P*P} PEs ({d_dim} floats/PE) ...")
runner.memcpy_h2d(
    q_sym,
    np.tile(q, P * P),
    0, 0, P, P,
    d_dim,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)
print(f"       q copied: {q.nbytes * P * P:,} bytes")

# ---------------------------------------------------------------------------
# Launch kernel
# ---------------------------------------------------------------------------
print(f"[5/7] Launching kernel 'compute' ...")
print(f"       Phase 1: Each PE computes squared-L2 distances + local top-{K}")
print(f"         - Distance: DSD-vectorized @fsubs + @fmuls over {d_dim} dims")
print(f"         - Top-K: {'heapify + heap sort (O(K log K))' if K >= rows_per_pe else f'max-heap replace (O(N·log {K}))' if K >= 32 else f'insertion sort (O(N·{K}))'}")
print(f"       Phase 2: Row-reduce (east→west), {P-1} merge steps, {2*K} wavelets/step")
print(f"       Phase 3: Col-reduce (south→north), {P-1} merge steps on col 0")
print(f"       Phase 4: PE(0,0) writes result_indices + result_distances")

t0 = time.time()
runner.launch("main", nonblock=False)
t1 = time.time()
print(f"       Kernel finished in {t1-t0:.2f}s (simulator wall time)")

# ---------------------------------------------------------------------------
# D2H: read results from PE(0,0)
# ---------------------------------------------------------------------------
print(f"[6/7] D2H: reading {K} indices + {K} distances from PE(0,0) ...")
result_indices_u32 = np.zeros(K, dtype=np.uint32)
result_distances = np.zeros(K, dtype=np.float32)

runner.memcpy_d2h(
    result_indices_u32,
    idx_sym,
    0, 0, 1, 1,
    K,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)
runner.memcpy_d2h(
    result_distances,
    dist_sym,
    0, 0, 1, 1,
    K,
    streaming=False,
    order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False,
)

runner.stop()

# Cast u32 → i32 for comparison with oracle
result_indices = result_indices_u32.astype(np.int32)

# Read cycle count from sim_stats.json if available
cycles = "N/A"
try:
    with open("sim_stats.json") as f:
        sim_stats = json.load(f)
    cycles = sim_stats.get("cycle_count", "N/A")
except Exception:
    pass
print(f"       Simulator cycles: {cycles}")

# ---------------------------------------------------------------------------
# Validate against oracle
# ---------------------------------------------------------------------------
print(f"[7/7] Validating against NumPy oracle ...")
expected_idx, expected_dist = topk_reference(D, q, K, squared=True)

# Show top results
show_n = min(10, K)
print(f"       Top-{show_n} results:")
print(f"       {'Rank':<6} {'Got Idx':<10} {'Exp Idx':<10} {'Match':<7} {'Got Dist':<16} {'Exp Dist':<16} {'Diff':<12}")
print(f"       {'----':<6} {'-------':<10} {'-------':<10} {'-----':<7} {'--------':<16} {'--------':<16} {'----':<12}")
all_idx_match = True
all_dist_close = True
for i in range(show_n):
    got_i = result_indices[i]
    exp_i = expected_idx[i]
    got_d = result_distances[i]
    exp_d = expected_dist[i]
    idx_ok = "✓" if got_i == exp_i else "✗"
    dist_diff = abs(got_d - exp_d)
    dist_ok = dist_diff <= 1e-3
    if got_i != exp_i:
        all_idx_match = False
    if not dist_ok:
        all_dist_close = False
    print(f"       {i:<6} {got_i:<10} {exp_i:<10} {idx_ok:<7} {got_d:<16.8f} {exp_d:<16.8f} {dist_diff:<12.2e}")

if K > show_n:
    # Check remaining silently
    remaining_idx_ok = np.array_equal(result_indices[show_n:], expected_idx[show_n:])
    remaining_dist_ok = np.allclose(result_distances[show_n:], expected_dist[show_n:], atol=1e-3, rtol=0)
    print(f"       ... ({K - show_n} more entries: idx={'✓' if remaining_idx_ok else '✗'}, dist={'✓' if remaining_dist_ok else '✗'})")
    if not remaining_idx_ok:
        all_idx_match = False
    if not remaining_dist_ok:
        all_dist_close = False

print()
try:
    np.testing.assert_array_equal(result_indices, expected_idx)
    np.testing.assert_allclose(result_distances, expected_dist, atol=1e-3, rtol=0)
except Exception as e:
    print(f"FAIL: {args.case}")
    print(str(e))
    sys.exit(1)

print(f"PASS: {args.case}")
