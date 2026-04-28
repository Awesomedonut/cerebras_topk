#!/bin/bash
# =============================================================================
# commands.sh — Compile and run the baseline test case
# =============================================================================
#
# This is the smoke-test entry point: builds the kernel + runs it once.
# Used by SPEC §3 as the canonical "build my submission" script.
#
# To run all 6 cases instead, use the official grader:
#   cd tests
#   python3 -m pytest test_correctness.py -v --submission=..
# =============================================================================

# `set -e` aborts immediately on any command failure (catches compile errors)
set -e


# -----------------------------------------------------------------------------
# Step 1: COMPILE — turn layout.csl + pe.csl into device binaries
# -----------------------------------------------------------------------------
# cslc flags explained:
#   --arch=wse2              → target Wafer Scale Engine 2 (CS-2)
#   --fabric-dims=11,6       → simulator fabric size (cols, rows)
#                              Required: width >= P+7, height >= P+1 (memcpy reserves)
#                              For P=4 baseline: P+7=11, P+1=5; we pad to 11,6 for safety.
#   --fabric-offsets=4,1     → user code starts at this corner (memcpy occupies 4,1)
#                              MANDATORY: hardcoded by memcpy infrastructure.
#   --memcpy --channels=1    → enable host↔device data transfer (SdkRuntime, not legacy CSELFRunner)
#                              MANDATORY for SDK 2.10+.
#   --params=KEY:VALUE,...   → set the `param X;` declarations in layout.csl + pe.csl
#                              Each test case uses different values.
#   -o out_baseline          → write compiled ELFs to out_baseline/
#
# Docs: https://sdk.cerebras.net/csl/csl-compiler
# -----------------------------------------------------------------------------
cslc --arch=wse2 layout.csl \
    --fabric-dims=11,6 --fabric-offsets=4,1 \
    --memcpy --channels=1 \
    --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
    -o out_baseline


# -----------------------------------------------------------------------------
# Step 2: RUN — execute the host driver, which talks to the wafer
# -----------------------------------------------------------------------------
# cs_python is Cerebras' Python interpreter (NOT regular python3).
# It has the SdkRuntime library installed so run.py can do memcpy / launch.
#
# run.py args:
#   --name out_baseline    → which compiled kernel to load
#   --case baseline        → which test case (drives D/q generation + verification)
#
# Expected stdout on success:
#   Got top-16 indices match NumPy oracle for case 'baseline'
#   PASS: baseline
# -----------------------------------------------------------------------------
cs_python run.py --name out_baseline --case baseline
