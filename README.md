# ML Tuner

The ML tuner is an optional ONNX-based NCCL tuning path for the SPCX tuner plugin. When enabled, it scores NCCL-allowed algorithm/protocol candidates, respects NCCL `IGNORE` entries, applies the model/Python valid-combo list, and selects the best entry for NCCL. If the model chooses `DEFAULT`, the plugin leaves NCCL's table unchanged.

The current model is `nccl_ml_tuner_v9.2.onnx`.

## Requirements

- SPCX NCCL plugin built with ML tuner support (`--enable-onnx`).
- ONNX Runtime shared library compatible with the bundled ONNX Runtime C API headers.
  - Set `NCCL_SPCX_ML_TUNER_ORT_PATH` if the library is not discoverable as `libonnxruntime.so.1` or `libonnxruntime.so`.
- The `nccl_ml_tuner_v9.2.onnx` model file from this repository.
- NCCL configured to load the SPCX tuner plugin in the target job environment.
- For standalone debugging, the `ml_tuner_test` binary from the SPCX plugin package.

## Runtime Environment Variables

`NCCL_SPCX_ML_TUNER_MODEL`

Top-level enable switch. If unset or empty, the ML tuner is disabled and ONNX Runtime is not loaded. Set this to either the full ONNX model path or a model directory. If set to a directory, the C++ tuner resolves `nccl_ml_tuner_v9.2.onnx` inside it.

Installed model path in the SPCX plugin package:

```bash
<prefix>/share/tuner-models/nccl_ml_tuner_v9.2.onnx
```

`NCCL_SPCX_ML_TUNER_ORT_PATH`

Optional path to the ONNX Runtime shared library. If unset, the loader tries default names such as `libonnxruntime.so.1` and `libonnxruntime.so`. If set but invalid, the plugin logs a warning and falls back to the default names.

`NCCL_SPCX_ML_TUNER_CACHE_SIZE`

Controls the internal result-cache size. Default is `128`. Set to `0` to disable result caching. Cache hits suppress the ML tuner pick log; fresh inference logs the pick with `INFO(NCCL_TUNING, ...)`.

`NCCL_SPCX_ML_TUNER_COLLS`

Optional collective filter. Accepted tokens are `ALL`, `AR`, and `AG`, comma-separated and case-insensitive.

Examples:

```bash
NCCL_SPCX_ML_TUNER_COLLS=ALL
NCCL_SPCX_ML_TUNER_COLLS=AR
NCCL_SPCX_ML_TUNER_COLLS=AR,AG
```

When unset, both AllReduce and AllGather are enabled. Invalid tokens warn and fall back to all.

## Debug And Test Commands

Installed test binary:

```bash
ml_tuner_test --model <model-or-dir> --nodes 4 --ranks 256 --size 1M --coll allreduce
```

Useful options:

- `--coll, --collective allreduce|allgather`
- `--nodes N`
- `--ranks N`
- `--size SIZE` with bytes/K/M/G suffix
- `--model PATH`
- `--ew` or `--no-nvlink` for EW-only mode
- `--python-valid-combos` for Python-reference candidate list in the standalone test

Wrapper script example:

```bash
./run_cpp_ml_tuner_test.sh --nodes 4 --ranks 256 --collective allreduce --mode nvlink --size 1M --verbose
```

Common wrapper options include `--collective`, `--nodes`, `--ranks`, `--size`, `--mode nvlink|ew|both`, `--full-range`, `--send-size`, `--verbose`, and `--output`.
