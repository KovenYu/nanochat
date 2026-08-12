#!/bin/bash
# Machine-local environment for this 4XH200 node.
# $HOME (/sailhome/koven) only has ~13GB free, so every cache that grows
# (dataset shards, checkpoints, uv wheels, HF kernels) is redirected to /svl/u/koven.
# Usage: screen -L -Logfile runs/speedrun.log -S speedrun bash -c 'source runs/env_4xh200.sh && bash runs/speedrun.sh'
# The system CUDA on LD_LIBRARY_PATH (/usr/local/cuda-12.2/lib64) ships libcudnn.so.9.7.1,
# which shadows the cuDNN 9.10.2 bundled in the torch wheel (LD_LIBRARY_PATH is searched
# before DT_RUNPATH), and torch hard-errors on the version mismatch. The torch wheels are
# fully self-contained, so drop system CUDA from the path entirely and keep only OpenBLAS.
export LD_LIBRARY_PATH=/viscam/u/koven/opt/OpenBLAS

export NANOCHAT_BASE_DIR=/svl/u/koven/nanochat_data
export UV_CACHE_DIR=/svl/u/koven/nanochat_cache/uv
export UV_PYTHON_INSTALL_DIR=/svl/u/koven/nanochat_cache/python
export HF_HOME=/svl/u/koven/nanochat_cache/hf
