# Run record and compare

`oxfuse run FILE --manifest OUT.json` writes a small JSON record next to the normal output, so two runs can be diffed and a result can be pinned to what produced it. A failed write is an error, and the outputs are still printed.

| Key | Meaning |
|---|---|
| `manifest_version` | 1 |
| `oxfuse_version` | the crate version of the binary |
| `backend` | `reference`, `cpu` or `wgpu` |
| `optimize`, `disabled_passes` | the pass options of the run |
| `os`, `arch` | the platform the binary was built for |
| `simd` | the SIMD level the fast CPU backend picked (`Neon`, `Avx2`, ...) |
| `features` | `parallel`, `wgpu`, `onnx`: which were compiled in |
| `model_sha256`, `inputs_sha256` | SHA-256 of the model file and of the inputs file (`null` without one), as they are when the run ends |
| `outputs` | per output: `name`, `shape`, `sha256` over the little-endian bit patterns of the values, so `-0.0` and `0.0` differ |
| `unix_time` | seconds since the epoch; the only field that changes between identical runs |

Two runs of the same model on the same inputs give equal `outputs` on `reference` and `cpu`; that is the bit contract in `semantics.md`, now one hash compare. Paths, user names and host names are not recorded.

## Comparing two runs

`oxfuse compare A.json B.json [--manifest-a A.manifest.json] [--manifest-b B.manifest.json]` takes the outputs of two runs and prints one line per output (`identical`, or how many values differ, the largest difference in ulp and the first index), then a verdict. A difference is counted by bit pattern, so `-0.0` against `0.0` is a difference of 0 ulp. Outputs must have the same names and shapes; otherwise it is an error (`E111`). A run record must describe the outputs file it is given with (`E112`).

| Verdict | When | Exit |
|---|---|---|
| identical | every bit equal | 0 |
| permitted variation | the values differ, and the records say one run used the `wgpu` backend (the GPU contract allows it: `contract.json`) | 0 |
| contract violation | the values differ, and the records say both runs are the same Oxfuse version on `reference` or `cpu` | 1 (`E110`) |
| unresolved | the values differ, and the records do not settle it: no record, another version, a run from another tool | 1 (`E110`) |

The verdict does not say how large a difference the contract allows; `contract.json` has that per operator and backend. A rigorous bound for a whole graph is future work.
