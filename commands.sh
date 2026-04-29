#!/usr/bin/env bash
set -e

# Compile baseline (P=4, d_dim=32, rows_per_pe=128, K=16)
cslc --arch=wse2 layout.csl \
  --fabric-dims=11,6 --fabric-offsets=4,1 \
  --params=P:4,d_dim:32,rows_per_pe:128,K:16 \
  --memcpy --channels=1 \
  -o out_baseline

# Run baseline
cs_python run.py --name out_baseline --case baseline
