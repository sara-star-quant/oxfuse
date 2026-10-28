# Run record

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
