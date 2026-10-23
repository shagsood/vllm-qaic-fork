#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

# Install the Cohere2-MoE vLLM-QAIC AOT stack, then serve existing AI200 QPCs.
# Set INSTALL_VLLM_QAIC=0 after the first successful install.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_VLLM_QAIC="${INSTALL_VLLM_QAIC:-1}"

usage() {
    cat <<EOF
Usage:
  $0 --prefill-qpc-path /path/to/prefill/qpc --decode-qpc-path /path/to/decode/qpc

The default AI200 topology is prefill QIDs 0-7 and decode QIDs 8-11.
Set INSTALL_VLLM_QAIC=0 to skip installation after the first run.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$INSTALL_VLLM_QAIC" == "1" ]]; then
    TRANSFORMERS_VERSION_AOT=5.9.0 bash "$ROOT_DIR/scripts/install.sh" aot
fi

exec "$PYTHON_BIN" "$ROOT_DIR/examples/cohere2_moe_ai200_qaic_disagg.py" "$@"
