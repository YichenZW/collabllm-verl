#!/bin/bash
set -euo pipefail
cd /net/scratch2/yichenzw/collabllm
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec .venv/bin/python -m reproduction.budget_service
