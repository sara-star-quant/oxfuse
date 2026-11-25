# Checking another runtime: `oxfuse conform`

You have the outputs another runtime (or another backend) produced for a graph and some inputs. `oxfuse conform` computes the enclosure of every output element (see [analysis.md](analysis.md): the exact real-number result and the reference float32 result, for every input in a box) and says, for every element of your outputs, whether it is **inside**, **outside** or **unresolved**. There is no tolerance parameter.

```sh
oxfuse conform OUTPUTS --graph GRAPH --inputs INPUTS
```

`OUTPUTS` is the JSON `oxfuse run` prints (`{"outputs": [{"name", "shape", "values"}, ...]}`); `GRAPH` is the graph file or ONNX model (`--dim NAME=VALUE` fixes a named dimension, as for the other commands); `INPUTS` is an inputs file whose inputs are points or boxes, exactly as for `analyze`.

## Example

The reference backend's outputs for the example network are inside the enclosure of the same inputs:

```sh
oxfuse run examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json --backend reference > out.json
oxfuse conform out.json --graph examples/tiny_mlp.json --inputs examples/tiny_mlp.inputs.json
```

```
out: 4 of 4 inside, 0 outside, 0 unresolved
verdict: inside
```

A copy with one element moved from 1.0 to 1.5:

```json
{"outputs":[{"name":"out","shape":[2,2],"values":[0.0,1.0,1.5,0.0]}]}
```

```
out: 3 of 4 inside, 1 outside, 0 unresolved; first outside at index 2: 1.5 is not in [0.9999999105930306, 1.0000001192092918]
verdict: outside
```

The report goes to standard output, one line per output with the first element that is outside; the last line is the verdict.

## What the verdicts mean

- **inside**: the value is a finite number between the two ends of the enclosure, ends included. The ends are compared as float64, not rounded to float32.
- **outside**: the value is a finite number past a finite end of the enclosure, whatever the other end is.
- **unresolved**: the enclosure is unbounded on a side (a divisor whose range reaches zero, or a result that can pass the largest float32) and the value is not past the finite end, or the value is not a finite number. Neither is ever a pass.

The run's verdict is outside if any element is outside, else unresolved if any is unresolved, else inside. The exit code is 0 only for inside. For outside and unresolved it is 1, the report is still printed to standard output, and the error line says which (`error[E114]`).

A file that does not match the graph is an error with no verdict: an output of the graph that is missing, an output the graph does not have, a different shape, an output named twice, or a file that is not valid outputs JSON (`error[E115]`, `error[E111]`, `error[E113]`). NaN and infinity are not valid JSON, so a file that contains one is rejected as unreadable instead of getting a verdict.

## What outside means, and does not

The enclosure is the range of the reference arithmetic of this graph: the operator order, the fused multiply-add chain of every `MatMul`, the own `exp`. Outside means the value is not explained by that model. A runtime that sums in another order has its own rounding error, so with a point input, where the enclosure is only a few ulp wide, it can be outside and still be accurate; a wider box gives a wider enclosure. Outside is evidence that the outputs do not follow the reference contract, not a measurement of how wrong they are. Inside means the value lies in the enclosure, not that it came from an input in the box.

It does not run the other runtime, rank anything, or apply a tolerance. The GPU backend is allowed by the contract to differ from the reference ([semantics.md](semantics.md)), so its outputs can be outside without being a defect.

## For comparison: tolerances elsewhere

A tolerance is a number someone chose. The node tests of the ONNX backend suite, for example, build every test case with `rtol=1e-3, atol=1e-7` ([`onnx/backend/test/case/node/__init__.py`](https://github.com/onnx/onnx/blob/v1.23.2/onnx/backend/test/case/node/__init__.py#L327-L328), onnx v1.23.2, read 2026-11-25) and compare with `np.testing.assert_allclose` ([`onnx/backend/test/runner/__init__.py`](https://github.com/onnx/onnx/blob/v1.23.2/onnx/backend/test/runner/__init__.py#L240), same version and date). `oxfuse conform` takes no such number: an element is inside the enclosure or it is not. That is a different question, not a better one: a tolerance says "close enough to a reference", the enclosure says "explained by the reference arithmetic".
