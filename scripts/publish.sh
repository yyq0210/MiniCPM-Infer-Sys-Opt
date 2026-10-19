#!/usr/bin/env bash
# Explicit one-shot publication; requires working GitHub credentials.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
git -C /workspace/LeanScale/vllm-plugin-FL push leanscale \
  leanscale/flagos-s2-baseline:refs/heads/leanscale/flagos-s2-baseline
git -C /workspace/LeanScale/FlagGems push leanscale \
  leanscale/flagos-s2-baseline:refs/heads/leanscale/flagos-s2-baseline \
  leanscale/iluvatar-rope-v1:refs/heads/leanscale/iluvatar-rope-v1
git -C /workspace/LeanScale/MiniCPM-Infer-Sys-Opt push origin main:refs/heads/main
git -C /workspace/LeanScale/vllm-plugin-FL ls-remote --heads leanscale leanscale/flagos-s2-baseline
git -C /workspace/LeanScale/FlagGems ls-remote --heads leanscale 'leanscale/*'
git -C /workspace/LeanScale/MiniCPM-Infer-Sys-Opt ls-remote --heads origin main
