#!/bin/bash
# sudo -n drops the calling shell's environment (no env_keep for these
# vars), so both ncu's own report-save tempdir AND the target python
# process it launches (which inherits ncu's env, not ours) silently fall
# back to /tmp (root disk) as root -- refilling the root filesystem and
# eventually failing with "Failed to save the report to file". Pass the
# needed vars explicitly as part of the sudo'd command instead of relying
# on inheritance.
exec sudo -n env \
  TMPDIR=/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache \
  TRITON_CACHE_DIR=/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/triton \
  TORCHINDUCTOR_CACHE_DIR=/data/Triton-to-tile-IR/Tile_IR_Test/Chronos/dev_tests/.run_cache/inductor \
  /usr/local/cuda-13.1/bin/ncu "$@"
