#!/usr/bin/env cs_python
"""
================================================================================
run.py — Host driver for the Top-K k-NN kernel
================================================================================

WHAT THIS FILE DOES (runs on Linux CPU, NOT on the wafer):
   1. Generates the test case's D + q (same generator as the grader)
   2. Sets up SdkRuntime — Cerebras' Python library that talks to the wafer
   3. Sends D shards to all 16 PEs (memcpy_h2d)
   4. Sends q replicated to all 16 PEs
   5. Calls runner.launch("main") — runs the kernel on the wafer
   6. Reads back final K dists + idxs from PE(0,0) (memcpy_d2h)
   7. Compares against NumPy oracle from reference.py
   8. Prints "PASS: <case>" if correct (the grader greps for this string)

NOTE: this script must run with `cs_python` (not regular `python3`).
`cs_python` is the SDK-provided Python that has the cerebras runtime library.

Docs: https://sdk.cerebras.net/api-docs/sdkruntime-api
================================================================================
"""
import argparse
import json
import sys
import numpy as np

# SdkRuntime: the bridge between Python (host) and the wafer (device).
# Provided by the Cerebras SDK; only available inside cs_python.
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder
)

# Import the same case generators the official grader uses.
# This guarantees we test against the exact inputs the grader will use.
sys.path.insert(0, '.')
from reference import (
    topk_reference,
    make_baseline, make_k_eq_1, make_k_large,
    make_uneven, make_all_equal, make_duplicates,
)

# Map case names → generator functions (called via CASE_MAKERS[args.case]())
CASE_MAKERS = {
    'baseline':   make_baseline,
    'k_eq_1':     make_k_eq_1,
    'k_large':    make_k_large,
    'uneven':     make_uneven,
    'all_equal':  make_all_equal,
    'duplicates': make_duplicates,
}


def shard_D_for_pes(D: np.ndarray, P: int, rows_per_pe: int, d_dim: int):
    """Reshape D into the layout the wafer expects.

    The kernel expects PE(col, row) to own rows D_padded[(row*P + col)*rows_per_pe : ...].
    For memcpy_h2d with ROW_MAJOR ordering on a P×P grid, the array must be
    indexed as data[height=P][width=P][rows_per_pe * d_dim].

    Padding: if N (D.shape[0]) < P*P*rows_per_pe (e.g. uneven case with N=1009),
    pad with 1e6 — those padded rows produce huge distances that never enter
    any top-K, so we don't need device-side handling.
    """
    N_total = P * P * rows_per_pe
    N_actual = D.shape[0]
    assert N_actual <= N_total, f"D has {N_actual} rows, grid holds {N_total}"

    if N_actual < N_total:
        pad = np.full((N_total - N_actual, d_dim), 1e6, dtype=np.float32)
        D_padded = np.vstack([D, pad])
    else:
        D_padded = D

    # Reshape into the (P, P, rows_per_pe * d_dim) layout that ROW_MAJOR memcpy
    # will deliver to PE(col, row) correctly.
    out = np.zeros((P, P, rows_per_pe * d_dim), dtype=np.float32)
    for row in range(P):
        for col in range(P):
            pe_linear = row * P + col
            shard = D_padded[pe_linear * rows_per_pe : (pe_linear + 1) * rows_per_pe]
            out[row, col, :] = shard.ravel()
    return out.ravel()


def main():
    # -------------------------------------------------------------------------
    # CLI: --name = compile output dir (e.g. out_baseline), --case = test name
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compile output dir")
    parser.add_argument("--case", required=True, choices=list(CASE_MAKERS))
    parser.add_argument("--cmaddr", default=None, help="real wafer IP:port (None=simulator)")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Generate the test inputs (same generator as grader)
    # -------------------------------------------------------------------------
    case = CASE_MAKERS[args.case]()
    D, q, K, P = case['D'], case['q'], case['K'], case['P']
    d_dim = D.shape[1]

    # Read compile-time params (rows_per_pe) from the cslc-emitted out.json
    with open(f"{args.name}/out.json") as f:
        compile_data = json.load(f)
    rows_per_pe = int(compile_data['params']['rows_per_pe'])

    # -------------------------------------------------------------------------
    # 2. Set up the runtime: connects to simulator (or real wafer if cmaddr set)
    #    cmaddr=None  → uses the local fabric simulator (Apptainer container)
    #    cmaddr=ip:port → connects to a real Cerebras CS-2 over the network
    # -------------------------------------------------------------------------
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

    # Get host-side handles for the symbols pe.csl exported.
    # These names must match @export_symbol(..., "name") on the device side.
    sym_D           = runner.get_id("D_shard")
    sym_q           = runner.get_id("q")
    sym_final_dists = runner.get_id("final_dists")
    sym_final_idxs  = runner.get_id("final_idxs")
    sym_perf_log    = runner.get_id("perf_log")

    runner.load()    # Push the compiled ELF binaries to all 16 PEs
    runner.run()     # Start the runtime; each PE waits for tasks to fire

    # -------------------------------------------------------------------------
    # 3. Send D shards. memcpy_h2d arguments:
    #      sym, host_array, start_col, start_row, width, height, elements_per_pe,
    #      streaming=False, order, data_type, nonblock
    # `nonblock=False` makes Python wait until the device has received the data.
    # -------------------------------------------------------------------------
    D_flat = shard_D_for_pes(D, P, rows_per_pe, d_dim)
    runner.memcpy_h2d(
        sym_D, D_flat,
        0, 0, P, P,                 # ROI: from (col=0, row=0), spans P×P
        rows_per_pe * d_dim,         # elements per PE
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # -------------------------------------------------------------------------
    # 4. Send q replicated to ALL PEs.
    #    np.tile(q, P*P) makes a flat array [q, q, q, ...] of length P*P*d_dim,
    #    which ROW_MAJOR memcpy delivers as the same q to each PE.
    # -------------------------------------------------------------------------
    q_replicated = np.tile(q, P * P).astype(np.float32)
    runner.memcpy_h2d(
        sym_q, q_replicated,
        0, 0, P, P,
        d_dim,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # -------------------------------------------------------------------------
    # 5. Launch the kernel. This blocks until the device's main() finishes.
    #    "Finish" = device calls sys_mod.unblock_cmd_stream() in the EXIT task.
    #    If pe.csl forgets that call, this line hangs forever.
    # -------------------------------------------------------------------------
    runner.launch("main", nonblock=False)

    # -------------------------------------------------------------------------
    # 6. Read back the final K results from PE(0,0) only.
    #    ROI is 1×1 starting at (col=0, row=0).
    # -------------------------------------------------------------------------
    final_dists = np.zeros(K, dtype=np.float32)
    final_idxs = np.zeros(K, dtype=np.uint32)

    runner.memcpy_d2h(
        final_dists, sym_final_dists,
        0, 0, 1, 1,                  # ROI: only PE(0,0)
        K,                           # K elements
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )
    runner.memcpy_d2h(
        final_idxs, sym_final_idxs,
        0, 0, 1, 1, K,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )

    # Read perf_log from ALL PEs: 6 timestamps × P*P PEs.
    # Use flat 1-D buffer; reshape after to (P, P, 6) — col-major over PEs.
    perf_flat = np.zeros(P * P * 6, dtype=np.uint32)
    runner.memcpy_d2h(
        perf_flat, sym_perf_log,
        0, 0, P, P, 6,
        streaming=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
    )
    perf_all = perf_flat.reshape(P, P, 6)

    runner.stop()    # Disconnect from the device

    # Dump perf_log to a sidecar JSON for the harness to pick up.
    # Phase deltas per PE (TSC handles wraparound via uint32 subtraction).
    import json as _json
    perf_phases = ["compute", "pack", "x_gather", "y_gather", "merge"]
    perf_data = {
        "case": args.case,
        "P": int(P), "K": int(K), "d_dim": int(d_dim), "rows_per_pe": int(rows_per_pe),
        "per_pe": []
    }
    for col in range(P):
        for row in range(P):
            ts = perf_all[col, row]
            # Use uint32 subtraction (handles wraparound). Convert to int for safe diffs.
            deltas = [int(np.uint32(ts[i+1]) - np.uint32(ts[i])) for i in range(5)]
            perf_data["per_pe"].append({
                "col": col, "row": row,
                "raw_ts": [int(t) for t in ts],
                "deltas": dict(zip(perf_phases, deltas)),
                "total_compute_to_merge": int(np.uint32(ts[5]) - np.uint32(ts[0])),
            })
    with open(f"perf_{args.case}.json", "w") as f:
        _json.dump(perf_data, f, indent=2)

    # -------------------------------------------------------------------------
    # 7. Verify against the NumPy oracle (reference.py).
    # -------------------------------------------------------------------------
    expected_idxs, expected_dists = topk_reference(D, q, K)

    # Cast u32 → i32 to compare apples-to-apples with the oracle's int32 idxs
    final_idxs_i32 = final_idxs.astype(np.int32)

    if not np.array_equal(final_idxs_i32, expected_idxs):
        print(f"FAIL idxs:")
        print(f"  got:      {final_idxs_i32[:10]}")
        print(f"  expected: {expected_idxs[:10]}")
        sys.exit(1)

    # Allow small fp32 tolerance on distances (1e-2 absolute, 1e-3 relative)
    if not np.allclose(final_dists, expected_dists, atol=1e-2, rtol=1e-3):
        print(f"FAIL dists:")
        print(f"  got:      {final_dists[:5]}")
        print(f"  expected: {expected_dists[:5]}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 8. Success — the grader greps stdout for "PASS: <case>"
    # -------------------------------------------------------------------------
    print(f"Got top-{K} indices match NumPy oracle for case '{args.case}'")
    print(f"PASS: {args.case}")


if __name__ == "__main__":
    main()
