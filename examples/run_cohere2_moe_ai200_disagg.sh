#!/usr/bin/env bash
# Cohere2-MoE AI200 8P/4D QEff compile + vLLM QAIC disaggregation.
# Run in a QAIC AI200 vLLM environment. See usage: $0 help

set -euo pipefail

MODEL_ID="${MODEL_ID:-CohereLabs/North-Mini-Code-1.0}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ID}"
WORKDIR="${WORKDIR:-$PWD/cohere2-moe-ai200}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$WORKDIR/artifacts}"
QEFF_DIR="${QEFF_DIR:-$WORKDIR/efficient-transformers}"
VLLM_QAIC_DIR="${VLLM_QAIC_DIR:-$WORKDIR/vllm-qaic}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QEFF_HOME="${QEFF_HOME:-$WORKDIR/qeff_cache}"
HF_HOME="${HF_HOME:-$WORKDIR/hf_cache}"
PREFILL_QIDS="${PREFILL_QIDS:-0,1,2,3,4,5,6,7}"
DECODE_QIDS="${DECODE_QIDS:-8,9,10,11}"
CTX_LEN="${CTX_LEN:-1024}"
PREFILL_PORT="${PREFILL_PORT:-18001}"
DECODE_PORT="${DECODE_PORT:-18002}"
PROXY_PORT="${PROXY_PORT:-18000}"
KV_PORT="${KV_PORT:-18565}"
MEASURED_ITERATIONS="${MEASURED_ITERATIONS:-5}"
GENERATION_LEN="${GENERATION_LEN:-32}"
PROMPT="${PROMPT:-$'Explain in one short sentence why key-value caching speeds up autoregressive decoding.\nAnswer:'}"
QEFF_REPO_URL="${QEFF_REPO_URL:-https://github.com/shagsood/efficient-transformers-fork.git}"
VLLM_QAIC_REPO_URL="${VLLM_QAIC_REPO_URL:-https://github.com/shagsood/vllm-qaic-fork.git}"
QEFF_BRANCH="${QEFF_BRANCH:-feature/add-cohere2_moe}"
VLLM_QAIC_BRANCH="${VLLM_QAIC_BRANCH:-feature/cohere2-moe-qaic}"
QAIC_UTIL="${QAIC_UTIL:-/opt/qti-aic/tools/qaic-util}"

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[$(date '+%F %T')] $*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  preflight  Verify AI200/QID availability and required Python tools.
  prepare    Clone/update QEff + Qualcomm vLLM-QAIC and install their pinned vLLM core.
  compile    Export and compile fresh AI200 8-device prefill + 4-device decode QPCs.
  serve      Launch vLLM QAIC disaggregation using QPCs from artifacts/qpcs.env.
  benchmark  Run one warm-up and $MEASURED_ITERATIONS streaming requests through the proxy.
  run        Run preflight, prepare, compile, launch, and benchmark end to end.
  help       Show this message.

Useful overrides:
  WORKDIR, MODEL_DIR, QEFF_DIR, VLLM_QAIC_DIR, PYTHON_BIN,
  PREFILL_QIDS, DECODE_QIDS, CTX_LEN, PREFILL_PORT, DECODE_PORT,
  PROXY_PORT, KV_PORT, QEFF_HOME, HF_HOME.

For an end-to-end run (the service log is saved under ARTIFACT_DIR):
  WORKDIR=/data/cohere2-moe-ai200 $0 run

The run command intentionally stops only the qaic_disagg process tree it starts
after benchmarking; it never touches unrelated QAIC processes.
EOF
}

parse_qids() {
  local group="$1" expected="$2" label="$3"
  local -a ids=()
  local id
  IFS=',' read -r -a ids <<<"$group"
  [[ ${#ids[@]} -eq "$expected" ]] || die "$label needs exactly $expected QIDs; got: $group"
  for id in "${ids[@]}"; do [[ "$id" =~ ^[0-9]+$ ]] || die "Invalid QID '$id' in $label"; done
}

preflight() {
  require_cmd git
  require_cmd curl
  require_cmd "$PYTHON_BIN"
  [[ -x "$QAIC_UTIL" ]] || die "QAIC utility is not executable: $QAIC_UTIL"
  parse_qids "$PREFILL_QIDS" 8 "PREFILL_QIDS"
  parse_qids "$DECODE_QIDS" 4 "DECODE_QIDS"
  local prefill_id decode_id output
  for prefill_id in ${PREFILL_QIDS//,/ }; do
    output="$($QAIC_UTIL -q -d "$prefill_id")"
    grep -q 'Status:Ready' <<<"$output" || die "Prefill QID $prefill_id is not Ready"
    grep -Eqi 'AIC200|Device a110' <<<"$output" || die "Prefill QID $prefill_id is not identified as AI200"
  done
  for decode_id in ${DECODE_QIDS//,/ }; do
    output="$($QAIC_UTIL -q -d "$decode_id")"
    grep -q 'Status:Ready' <<<"$output" || die "Decode QID $decode_id is not Ready"
    grep -Eqi 'AIC200|Device a110' <<<"$output" || die "Decode QID $decode_id is not identified as AI200"
  done
  "$PYTHON_BIN" - <<'PY'
import qaicrt  # noqa: F401
import qaic_disagg  # noqa: F401
print("qaicrt and qaic_disagg imports succeeded")
PY
  note "AI200 preflight passed: prefill=$PREFILL_QIDS decode=$DECODE_QIDS"
}

clone_or_update() {
  local url="$1" branch="$2" directory="$3"
  if [[ -e "$directory" && ! -d "$directory/.git" ]]; then
    die "Refusing to reuse non-git path: $directory"
  fi
  if [[ ! -d "$directory/.git" ]]; then
    git clone --branch "$branch" --single-branch "$url" "$directory"
    return
  fi
  git -C "$directory" diff --quiet || die "Refusing to update dirty checkout: $directory"
  git -C "$directory" fetch origin "$branch"
  git -C "$directory" checkout "$branch"
  git -C "$directory" pull --ff-only origin "$branch"
}

prepare() {
  mkdir -p "$WORKDIR" "$ARTIFACT_DIR" "$QEFF_HOME" "$HF_HOME"
  clone_or_update "$QEFF_REPO_URL" "$QEFF_BRANCH" "$QEFF_DIR"
  clone_or_update "$VLLM_QAIC_REPO_URL" "$VLLM_QAIC_BRANCH" "$VLLM_QAIC_DIR"
  VLLM_TARGET_DEVICE=empty "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps \
    "vllm @ git+https://github.com/shagsood/vllm-fork.git@ee08aabc25446e118492e5c271a4aaf7eeaaf8b1"
  "$PYTHON_BIN" -m pip install -e "$QEFF_DIR" -e "$VLLM_QAIC_DIR"
  note "QEff=$(git -C "$QEFF_DIR" rev-parse HEAD)"
  note "vLLM-QAIC=$(git -C "$VLLM_QAIC_DIR" rev-parse HEAD)"
  "$PYTHON_BIN" - <<'PY'
from transformers import AutoConfig
from vllm.model_executor.models.cohere2_moe import Cohere2MoeForCausalLM  # noqa: F401
config = AutoConfig.from_pretrained("CohereLabs/North-Mini-Code-1.0")
assert config.model_type == "cohere2_moe", config.model_type
assert config.architectures == ["Cohere2MoeForCausalLM"], config.architectures
print("HF config resolves:", config.architectures[0])
PY
}

compile_qpcs() {
  preflight
  [[ -d "$QEFF_DIR" ]] || die "Missing QEff checkout: $QEFF_DIR; run prepare first"
  mkdir -p "$ARTIFACT_DIR" "$QEFF_HOME" "$HF_HOME"
  QEFF_HOME="$QEFF_HOME" HF_HOME="$HF_HOME" ARTIFACT_DIR="$ARTIFACT_DIR" PYTHONPATH="$QEFF_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    MODEL_ID="$MODEL_ID" CTX_LEN="$CTX_LEN" PREFILL_QIDS="$PREFILL_QIDS" DECODE_QIDS="$DECODE_QIDS" \
    "$PYTHON_BIN" - <<'PY'
import os
import shlex
from pathlib import Path

from QEfficient import QEFFAutoModelForCausalLM

model_id = os.environ["MODEL_ID"]
ctx_len = int(os.environ["CTX_LEN"])
prefill_ids = [int(x) for x in os.environ["PREFILL_QIDS"].split(",")]
decode_ids = [int(x) for x in os.environ["DECODE_QIDS"].split(",")]
artifact_dir = Path(os.environ["ARTIFACT_DIR"])

# This is deliberately AI200-specific. The prior QPCs were AI100 and cannot be reused.
model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True)
decode_moe = {"flavour": "expert_parallel", "expert_parallel_chunk_size": 32}
if len(decode_ids) >= 8:
    decode_moe["cores_per_expert"] = 2

decode_qpc = model.compile(
    num_cores=4,
    aic_hw_version="ai200",
    num_devices=len(decode_ids),
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    batch_size=1,
    full_batch_size=1,
    prefill_seq_len=1,
    ctx_len=ctx_len,
    qaic_config={"moe_config": decode_moe},
    mos=1,
    aic_enable_depth_first=True,
    offload_pt_weights=False,
    split_retained_state_io=True,
    retain_full_kv=True,
    use_onnx_subfunctions=False,
)

prefill_qpc = model.compile(
    num_cores=4,
    aic_hw_version="ai200",
    num_devices=len(prefill_ids),
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    batch_size=1,
    full_batch_size=1,
    prefill_seq_len=128,
    ctx_len=ctx_len,
    qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 128}},
    mdp_num_partitions=1,
    split_retained_state_io=True,
    retain_full_kv=True,
    mos=1,
    aic_enable_depth_first=False,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
)

for label, qpc in (("prefill", Path(prefill_qpc)), ("decode", Path(decode_qpc))):
    if not (qpc / "programqpc.bin").is_file():
        raise SystemExit(f"{label} QPC missing programqpc.bin: {qpc}")

artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "qpcs.env").write_text(
    f"export PREFILL_QPC={shlex.quote(str(prefill_qpc))}\n"
    f"export DECODE_QPC={shlex.quote(str(decode_qpc))}\n"
    f"export MODEL_DIR={shlex.quote(model_id)}\n"
    f"export CTX_LEN={ctx_len}\n"
)
print(f"PREFILL_QPC={prefill_qpc}")
print(f"DECODE_QPC={decode_qpc}")
PY
  note "Fresh AI200 QPC paths written to $ARTIFACT_DIR/qpcs.env"
}

serve() {
  preflight
  [[ -f "$ARTIFACT_DIR/qpcs.env" ]] || die "Missing $ARTIFACT_DIR/qpcs.env; run compile first"
  # shellcheck disable=SC1090
  source "$ARTIFACT_DIR/qpcs.env"
  [[ -f "$PREFILL_QPC/programqpc.bin" ]] || die "Invalid prefill QPC: $PREFILL_QPC"
  [[ -f "$DECODE_QPC/programqpc.bin" ]] || die "Invalid decode QPC: $DECODE_QPC"
  "$PYTHON_BIN" -c 'from vllm.model_executor.models.cohere2_moe import Cohere2MoeForCausalLM'
  mkdir -p "$ARTIFACT_DIR/compat_site"
  cat >"$ARTIFACT_DIR/compat_site/sitecustomize.py" <<'PY'
"""Compatibility for explicit QAIC disaggregated QPC device groups."""
try:
    import vllm_qaic.model_loader.qaic as loader
    for profile in ("default", "turbo"):
        loader.QAIC_DEVICE_CONFIG[profile]["device_group"] = None
except Exception:
    pass
PY
  QAIC_VISIBLE_DEVICES="$PREFILL_QIDS,$DECODE_QIDS" \
  QEFF_HOME="$QEFF_HOME" HF_HOME="$HF_HOME" \
  PREFILL_QPC="$PREFILL_QPC" DECODE_QPC="$DECODE_QPC" MODEL_DIR="$MODEL_DIR" \
  PREFILL_QIDS="$PREFILL_QIDS" DECODE_QIDS="$DECODE_QIDS" CTX_LEN="$CTX_LEN" \
  PREFILL_PORT="$PREFILL_PORT" DECODE_PORT="$DECODE_PORT" PROXY_PORT="$PROXY_PORT" KV_PORT="$KV_PORT" \
  COMPAT_SITE="$ARTIFACT_DIR/compat_site" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

from vllm.platforms import current_platform

current_platform.package_root = "vllm_qaic"
import qaic_disagg.__main__ as disagg

real_popen = disagg.subprocess.Popen
prefill_port = int(os.environ["PREFILL_PORT"])
decode_port = int(os.environ["DECODE_PORT"])

def patched_popen(command, *args, **kwargs):
    if isinstance(command, (list, tuple)) and list(command[:2]) == ["vllm", "serve"]:
        command = list(command)
        port = int(command[command.index("--port") + 1])
        if "--model-tag" in command:
            index = command.index("--model-tag")
            del command[index:index + 2]
        config_index = command.index("--additional-config") + 1
        config = json.loads(command[config_index])
        if port == decode_port:
            # QAIC_VISIBLE_DEVICES renumbers physical decode QIDs locally.
            config["device_group"] = list(range(len(os.environ["DECODE_QIDS"].split(","))))
        command[config_index] = json.dumps(config)
        visible = {
            prefill_port: os.environ["PREFILL_QIDS"],
            decode_port: os.environ["DECODE_QIDS"],
        }.get(port)
        if visible is not None:
            env = dict(kwargs.get("env") or os.environ)
            env["QAIC_VISIBLE_DEVICES"] = visible
            env["VLLM_QAIC_USE_LEGACY_SLICING_SPEC"] = "1"
            env["PYTHONPATH"] = os.environ["COMPAT_SITE"] + ":" + env.get("PYTHONPATH", "")
            kwargs["env"] = env
            print(f"[cohere2-ai200] port={port} visible={visible} local_group={config.get('device_group')}", flush=True)
        return real_popen(command, *args, **kwargs)
    return real_popen(command, *args, **kwargs)

disagg.subprocess.Popen = patched_popen
prefill_cfg = {
    "qpc_path": os.environ["PREFILL_QPC"],
    "prefill_only": True,
    "split-model-io": True,
    "allow-mxint8-mdp-io": True,
}
decode_cfg = {
    "qpc_path": os.environ["DECODE_QPC"],
    "prefill_only": False,
    "split-model-io": True,
    "allow-mxint8-mdp-io": True,
}
sys.argv = [
    "qaic_disagg",
    "--model", os.environ["MODEL_DIR"],
    "--host", "0.0.0.0",
    "--port", os.environ["PROXY_PORT"],
    "--prefill-port", os.environ["PREFILL_PORT"],
    "--decode-port", os.environ["DECODE_PORT"],
    "--kv-handOff-port", os.environ["KV_PORT"],
    "--prefill-device-group", os.environ["PREFILL_QIDS"],
    "--decode-device-group", os.environ["DECODE_QIDS"],
    "--prefill-pipeline-parallel-size", "1",
    "--prefill-max-num-seqs", "1",
    "--decode-max-num-seqs", "1",
    "--max-model-len", os.environ["CTX_LEN"],
    "--dtype", "float16",
    "--quantization", "mxfp6",
    "--kv-cache-dtype", "mxint8",
    "--prefill-override-qaic-config", json.dumps(prefill_cfg),
    "--decode-override-qaic-config", json.dumps(decode_cfg),
    "-vvv",
]
disagg.main()
PY
}

benchmark() {
  require_cmd curl
  mkdir -p "$ARTIFACT_DIR"
  PROXY_PORT="$PROXY_PORT" ARTIFACT_DIR="$ARTIFACT_DIR" PROMPT="$PROMPT" \
  GENERATION_LEN="$GENERATION_LEN" MEASURED_ITERATIONS="$MEASURED_ITERATIONS" "$PYTHON_BIN" - <<'PY'
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

port = os.environ["PROXY_PORT"]
base_url = f"http://127.0.0.1:{port}"
model = json.load(urllib.request.urlopen(base_url + "/v1/models", timeout=10))["data"][0]["id"]
body = {
    "model": model,
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["GENERATION_LEN"]),
    "ignore_eos": True,
    "temperature": 0.0,
    "top_p": 1.0,
    "stream": True,
    "stream_options": {"include_usage": True},
}

run() {
  preflight
  prepare
  compile_qpcs
  mkdir -p "$ARTIFACT_DIR"
  local server_pid
  serve >"$ARTIFACT_DIR/qaic_disagg.log" 2>&1 &
  server_pid=$!
  trap "kill -TERM $server_pid 2>/dev/null || true; wait $server_pid 2>/dev/null || true" EXIT INT TERM
  local attempt
  for attempt in $(seq 1 60); do
    if curl --fail --silent "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null; then
      note "Disaggregated service is ready (pid=$server_pid)"
      benchmark
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      tail -100 "$ARTIFACT_DIR/qaic_disagg.log" >&2 || true
      die "qaic_disagg exited before becoming ready"
    fi
    sleep 2
  done
  tail -100 "$ARTIFACT_DIR/qaic_disagg.log" >&2 || true
  die "Timed out waiting for qaic_disagg on port $PROXY_PORT"
}

def request(iteration):
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            now = time.perf_counter()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            item = json.loads(payload)
            chunks += 1
            if first is None and any(choice.get("text") for choice in item.get("choices", [])):
                first = now
            if item.get("usage"):
                usage = item["usage"]
    end = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("Streaming response did not contain first text and usage")
    return {
        "iteration": iteration,
        "ttft_s": first - start,
        "total_latency_s": end - start,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "chunks": chunks,
    }

warmup = request(0)
samples = [request(index) for index in range(1, int(os.environ["MEASURED_ITERATIONS"]) + 1)]
for sample in samples:
    sample["effective_prefill_tok_s"] = sample["prompt_tokens"] / sample["ttft_s"]
    sample["decode_tok_s"] = (sample["completion_tokens"] - 1) / (sample["total_latency_s"] - sample["ttft_s"])

def summary(field):
    values = [sample[field] for sample in samples]
    return {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}

result = {
    "model": model,
    "request": body,
    "warmup_excluded": warmup,
    "samples": samples,
    "summary": {
        "ttft_s": summary("ttft_s"),
        "total_latency_s": summary("total_latency_s"),
        "effective_prefill_tok_s": summary("effective_prefill_tok_s"),
        "decode_tok_s": summary("decode_tok_s"),
    },
}
artifact_dir = Path(os.environ["ARTIFACT_DIR"])
(artifact_dir / "benchmark_ai200.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["summary"], indent=2))
print(f"Saved {artifact_dir / 'benchmark_ai200.json'}")
PY
}

case "${1:-help}" in
  preflight) preflight ;;
  prepare) prepare ;;
  compile) compile_qpcs ;;
  serve) serve ;;
  benchmark) benchmark ;;
  run) run ;;
  help|-h|--help) usage ;;
  *) die "Unknown command: ${1:-}" ;;
esac
