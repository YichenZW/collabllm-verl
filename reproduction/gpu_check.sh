#!/bin/bash
set -euo pipefail
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
printenv CUDA_VISIBLE_DEVICES
