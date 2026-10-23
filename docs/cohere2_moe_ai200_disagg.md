# Cohere2-MoE: AI200 vLLM disaggregated export, compile, and serve

This runbook targets `CohereLabs/North-Mini-Code-1.0` with:

```text
Prefill QIDs: 0,1,2,3,4,5,6,7
Decode QIDs:  8,9,10,11
```

Use another disjoint 8+4 QID split if these are occupied. Do not reuse the
AI100 QPCs: compile fresh QPCs on the AI200 host with
`aic_hw_version=ai200`.

## 1. Verify the AI200 host

```bash
/opt/qti-aic/tools/qaic-util -q
```

Confirm that the target QIDs are `Ready` and report AI200/AIC200 firmware.
Load the vendor AI200 Apps/Platform SDK environment for this host, then verify
that `qaic-compile`, `qaicrt`, and `qaic_disagg` are available in the selected
Python environment.

```bash
python -c 'import qaicrt; print(qaicrt.__file__)'
qaic-compile --help >/dev/null
qaic_disagg --help >/dev/null
```

## 2. Pull the model support

Use clean checkouts. Do not pull into a dirty working tree.

```bash
git clone https://github.com/shagsood/efficient-transformers-fork.git efficient-transformers
cd efficient-transformers
git checkout feature/add-cohere2_moe
git pull --ff-only origin feature/add-cohere2_moe
git rev-parse HEAD
# Expected known-good QEff commit: 531c0536
```

Pull the Qualcomm QAIC vLLM backend fork. Its installer pulls the pinned
Cohere2-MoE vLLM core package; do not install `vllm-project/vllm`.

```bash
git clone https://github.com/shagsood/vllm-qaic-fork.git vllm-qaic
cd vllm-qaic
git checkout feature/cohere2-moe-qaic
git pull --ff-only origin feature/cohere2-moe-qaic
git rev-parse HEAD
# Expected QAIC backend commit: see the branch HEAD.
```

Install the QEff fork, pinned Cohere2-MoE vLLM core, and Qualcomm QAIC plugin
into the AI200 environment:

```bash
export QEFF_HOME=/data/qeff_cache
export HF_HOME=/data/hf_cache
pip install -e /path/to/efficient-transformers
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation --no-deps \
  "vllm @ git+https://github.com/shagsood/vllm-fork.git@ee08aabc25446e118492e5c271a4aaf7eeaaf8b1"
pip install -e /path/to/vllm-qaic
```

Verify the model configuration before compiling:

```bash
python - <<'PY'
from transformers import AutoConfig

config = AutoConfig.from_pretrained("CohereLabs/North-Mini-Code-1.0")
assert config.model_type == "cohere2_moe", config.model_type
assert config.architectures == ["Cohere2MoeForCausalLM"], config.architectures
print(config.model_type, config.architectures)
PY
```

## 3. Make the QEff example target AI200

The current `examples/text_generation/cohere2_moe_example.py` explicitly
passes `aic_hw_version="ai100"` and `num_cores=16` in both compile calls. Do
not run it unchanged on AI200.

In both calls, replace:

```python
num_cores=16, aic_hw_version="ai100",
```

with:

```python
num_cores=int(os.environ.get("AIC_NUM_CORES", "4")),
aic_hw_version=os.environ.get("AIC_HW_VERSION", "ai200"),
```

AI200's QEff hardware defaults use four cores. Keep FP16 plus MXFP6 for the
first compile; BF16 is an AI200 capability but is a separate numerical
configuration that must be validated independently.

Commit and push this parameterization to the QEff fork if other AI200 systems
will reuse it.

## 4. Export and compile prefill and decode QPCs

`QEFFAutoModelForCausalLM.compile()` performs ONNX export and QPC compilation.
The example builds separate QPCs for prefill and decode.

```bash
cd /path/to/efficient-transformers

export AIC_HW_VERSION=ai200
export AIC_NUM_CORES=4
export PREFILL_DEVICE_GROUP=0,1,2,3,4,5,6,7
export DECODE_DEVICE_GROUP=8,9,10,11

python examples/text_generation/cohere2_moe_example.py \
  --model-id CohereLabs/North-Mini-Code-1.0 \
  --ctx-len 1024 \
  --prefill-device-group "$PREFILL_DEVICE_GROUP" \
  --decode-device-group "$DECODE_DEVICE_GROUP" \
  --generation-len 64
```

The intended compilation properties are:

| Stage | Devices | Static sequence | Required options |
|---|---:|---:|---|
| Prefill | 8 | 128 | `prefill_only=True`, `enable_chunking=True`, retained state, MXFP6, MXINT8 KV |
| Decode | 4 | 1 | retained state, MXFP6, MXINT8 KV, expert-parallel MoE |

Record the emitted QPC paths:

```bash
export PREFILL_QPC=/absolute/path/to/prefill/qpc
export DECODE_QPC=/absolute/path/to/decode/qpc
export MODEL_DIR=/absolute/path/to/CohereLabs/North-Mini-Code-1.0

test -f "$PREFILL_QPC/programqpc.bin"
test -f "$DECODE_QPC/programqpc.bin"
```

Inspect both `qconfig.json` files. They must identify `aic_hw_version=ai200`;
the prefill QPC must use 8 devices and the decode QPC 4. QPCs compiled for
AI100 or a different device cardinality are invalid for this deployment.

## 5. Configure vLLM disaggregation

Use the precompiled QPCs rather than allowing an accidental on-demand compile:

```bash
export PREFILL_CFG="$(printf \
  '{\"qpc_path\":\"%s\",\"prefill_only\":true,\"split-model-io\":true,\"allow-mxint8-mdp-io\":true}' \
  "$PREFILL_QPC")"

export DECODE_CFG="$(printf \
  '{\"qpc_path\":\"%s\",\"prefill_only\":false,\"split-model-io\":true,\"allow-mxint8-mdp-io\":true}' \
  "$DECODE_QPC")"
```

The launcher arguments are:

```bash
qaic_disagg \
  --model "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 18000 \
  --prefill-port 18001 \
  --decode-port 18002 \
  --kv-handOff-port 18565 \
  --prefill-device-group 0,1,2,3,4,5,6,7 \
  --decode-device-group 8,9,10,11 \
  --prefill-pipeline-parallel-size 1 \
  --prefill-max-num-seqs 1 \
  --decode-max-num-seqs 1 \
  --max-model-len 1024 \
  --dtype float16 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --prefill-override-qaic-config "$PREFILL_CFG" \
  --decode-override-qaic-config "$DECODE_CFG" \
  -vvv
```

## 6. Required child-process device isolation

For the tested vLLM 0.15 / QAIC combination, `qaic_disagg` must start each
worker with only its own physical QIDs visible:

```text
Prefill child: QAIC_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
Decode child:  QAIC_VISIBLE_DEVICES=8,9,10,11
```

The decode process renumbers visible cards locally. Its QAIC additional config
must therefore be `device_group=[0,1,2,3]`, not `[8,9,10,11]`.

For QPCs with legacy `past_key.*` retained-state I/O, set this in both child
workers:

```bash
export VLLM_QAIC_USE_LEGACY_SLICING_SPEC=1
```

The known vLLM 0.15 integration also needed a launcher compatibility wrapper
that:

1. removes obsolete `--model-tag` from child `vllm serve` commands;
2. applies the child-specific `QAIC_VISIBLE_DEVICES` values above;
3. changes the decode config to local IDs `[0,1,2,3]`; and
4. clears the QAIC loader's default device group before it consumes the
   explicit additional config.

Validate whether the chosen AI200 image has fixed these issues before removing
the wrapper. A launcher that exposes all 12 devices to both workers can fail
the QPC topology check.

## 7. Validate the service and benchmark it

```bash
curl -s http://127.0.0.1:18000/v1/models
curl -s http://127.0.0.1:18001/health
curl -s http://127.0.0.1:18002/health
```

Send a warm-up and then five measured requests through the proxy:

```bash
curl -N http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_DIR"'",
    "prompt": "Explain KV caching in one sentence.\\nAnswer:",
    "max_tokens": 32,
    "ignore_eos": true,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Record the QPC paths, SDK and firmware versions, device groups, startup time,
TTFT, prefill rate, decode rate, total latency, prompt tokens, and generated
tokens. Check every used QID returns to `Ready` after shutdown.

## AI100 result that must not be reused as an AI200 claim

The prior AI100 8+4 vLLM run measured 187.4 ms mean TTFT, 90.73 effective
prefill tok/s, and 14.49 decode tok/s. It proves the vLLM model and
disaggregation flow, not AI200 fit, accuracy, or performance.
