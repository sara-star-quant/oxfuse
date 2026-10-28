# ONNX import

`oxfuse run model.onnx` reads a float32 ONNX model made of a small operator subset and runs it on any backend. The importer is the optional crate `oxfuse-onnx`, behind the CLI feature `onnx`:

    cargo install oxfuse --features onnx
    oxfuse run model.onnx --inputs inputs.json --backend cpu
    oxfuse check model.onnx
    oxfuse inspect model.onnx

A model with a named dimension (`N`, `batch`) needs a value for it: `--dim N=1`. The flag repeats; naming a dimension twice is an error. Without the feature, an `.onnx` file fails with `E107`. A file of many tiny messages costs far more memory than its bytes (a 16 MiB file of empty nodes peaked at 1.2 GB), so the importer counts messages before decoding and refuses more than ten per `max_values` (`E314`).

## What is supported

Default-domain opsets 13 through 21 only. Anything else fails with "re-export with opset 13 or newer". All tensors are float32; `int64` is accepted only as the shape operand of `Reshape`.

| ONNX op | Lowered to | Rules |
|---|---|---|
| `Relu`, `Sigmoid` | same op | |
| `MatMul` | `MatMul` | rank 2 only |
| `Add`, `Mul` | same op | equal shapes or a scalar run as is; a constant operand is expanded to the broadcast shape at import; a non-constant operand that needs broadcasting is an error |
| `Gemm` | `MatMul`, then `Transpose`, `Mul`, `Add` as needed | `alpha`, `beta`, `transA`, `transB`, optional `C` (a constant `C` is scaled by `beta` at import; `beta = 0` drops `C` as ONNX does, even a `C` full of infinity or NaN, and still checks its shape) |
| `Reshape` | `Reshape` | the shape is an int64 initializer or `Constant`; `0` and `-1` are resolved at import; `allowzero=1` with a literal `0` is an error, the IR has no zero-size tensors |
| `Flatten` | `Reshape` | `axis` in `[-rank, rank]` |
| `Transpose` | `Transpose` | default `perm` reverses the axes |
| `Identity`, `Dropout` | no node | `Dropout` is the inference identity; a `mask` output or a `training_mode` input is an error |
| `Constant` | constant | one of `value`, `value_float`, `value_floats`, `value_int`, `value_ints` |

Every attribute a handler does not read is an error that names the node, so nothing is dropped silently. Initializers that are also listed as graph inputs (older IR versions) are constants; the initializer decides their type and shape, the declaration is not read. Unused initializers do not reach the graph.

Not supported: convolution, pooling, normalization, `Softmax`, attention, quantization, control flow, sequences, sparse tensors, external data files, dtypes other than float32, dynamic shapes, general broadcasting.

## Numerics

Lowering never changes a computation. With `alpha = beta = 1` and no transposes `Gemm` is exactly `MatMul` plus `Add`. Any other setting adds one multiplication by a constant (`alpha` is never dropped, only an exact `1.0` skips the multiply, compared by bit pattern) or one `Transpose`, so such a model sits inside the `MatMul` bound of [semantics.md](semantics.md) plus one rounding per added multiplication. Models that run on the GPU backend follow its two-tier contract.

## Limits and untrusted models

A model is untrusted input, like a graph file. The size cap, the tensor element caps and the value cap are the core `Limits`. Nesting is capped by the protobuf decoder (100 messages). Every dimension product and every constant expansion is checked with checked arithmetic against the limits before anything is allocated.

## Errors

| Code | Meaning |
|---|---|
| E300 | the bytes are not a decodable model |
| E301 | the file is over the size limit |
| E302 | messages nest too deeply |
| E303 | opset outside 13 to 21, or missing |
| E304 | a node uses a domain other than the default |
| E305 | unsupported op |
| E306 | unsupported attribute |
| E307 | an element type other than float32 |
| E308 | a malformed tensor (shape, data length, external data) |
| E309 | invalid structure: no graph, unknown or misordered values, no outputs |
| E310 | a value name defined twice |
| E311 | a node-level shape or operand error |
| E312 | a named dimension without `--dim` |
| E313 | an expansion over a limit |
| E314 | more messages than the importer allows (ten per `max_values`, so a file of tiny messages cannot exhaust memory) |

Errors from the IR (`E0xx`) pass through with their own codes. `E107`, `E108` and `E109` come from the CLI.

## How it is checked

`scripts/gen_onnx_fixtures.py` builds 14 small models with the `onnx` package and evaluates each three times: onnxruntime, the `onnx` package's reference evaluator at float32, and the same evaluator on a float64 copy of the model. Generation fails unless all three agree, and the committed expected values are the float64 run. Data are multiples of 0.25, so every fixture without `Sigmoid` is exact in float32 and must match bit for bit; `Sigmoid` fixtures use an absolute and relative tolerance of 1e-6. `just onnx-check` recomputes and compares the committed files. The Rust tests run every fixture on the reference and fast CPU backends (the two must print the same bits) and, in the CLI tests, on the GPU backend when an adapter exists.
