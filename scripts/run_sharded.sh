#!/usr/bin/env bash
# Run the attack over several GPUs, one shard per GPU.
# Usage: scripts/run_sharded.sh <config> <gpu> [<gpu> ...]
#   e.g. scripts/run_sharded.sh config/iau_foa_1000.yaml 0 1 2 3
set -eu
cd "$(dirname "$0")/.."
CFG=$1; shift
GPUS=("$@")
NS=${#GPUS[@]}
for SID in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$SID]} python IAU_FOA.py --config "$CFG" \
    model.device=cuda:0 data.num_shards="$NS" data.shard_id="$SID" > /dev/null &
done
wait
