#!/usr/bin/env python3
"""ONNX import fixtures for Oxfuse, checked by two independent oracles.

Builds small models with the `onnx` package, evaluates each one twice at
float32 (onnxruntime and the `onnx` package's pure-Python ReferenceEvaluator)
and once at float64 (ReferenceEvaluator on a double-precision copy), and
refuses to write anything unless the three agree. The committed expected
values are the float64 evaluation. Data are dyadic (multiples of 0.25), so
every non-Sigmoid fixture is exact in float32 and must match bit for bit.

Usage (needs the virtualenv from scripts/requirements-onnx.txt):
  .venv-onnx/bin/python -I scripts/gen_onnx_fixtures.py [--check]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates" / "onnx" / "tests" / "fixtures"
LOOSE = {"abs": 1e-6, "rel": 1e-6}
EXACT = {"abs": 0.0, "rel": 0.0}
OPSET = 13
IR_VERSION = 9


class Disagreement(Exception):
    """Two oracles (or an oracle and its own float64 run) gave different answers."""


def dyadic(rs, shape):
    """Multiples of 0.25 in [-2, 2]."""
    return (rs.randint(-8, 9, size=shape) / 4).astype(np.float32)


def info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


class Fixture:
    def __init__(self, name, nodes, inputs, outputs, inits, feeds, dims=None, sigmoid=False):
        self.name = name
        self.dims = dims or {}
        self.sigmoid = sigmoid
        self.feeds = feeds  # input name -> float32 array, in input order
        def make(output_infos):
            graph = helper.make_graph(
                nodes, name, inputs, output_infos, initializer=inits
            )
            return helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", OPSET)],
                ir_version=IR_VERSION,
            )

        # Graph outputs carry a type and shape; take them from shape inference.
        inferred = onnx.shape_inference.infer_shapes(
            make([helper.make_empty_tensor_value_info(o) for o in outputs]))
        known = {vi.name: vi for vi in [*inferred.graph.value_info, *inputs]}
        self.model = make([known[o] for o in outputs])
        onnx.checker.check_model(self.model)
        self.outputs = outputs

    @property
    def tolerance(self):
        return LOOSE if self.sigmoid else EXACT


def fixtures():
    out = []

    rs = np.random.RandomState(1)
    out.append(Fixture(
        "linear",
        [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Add", ["h", "b"], ["y"])],
        [info("x", [2, 3])], ["y"],
        [const("w", dyadic(rs, (3, 4))), const("b", dyadic(rs, (4,)))],
        {"x": dyadic(rs, (2, 3))},
    ))

    rs = np.random.RandomState(2)
    out.append(Fixture(
        "mlp",
        [
            helper.make_node("Gemm", ["x", "w1", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["a1"]),
            helper.make_node("Gemm", ["a1", "w2", "b2"], ["z"]),
            helper.make_node("Sigmoid", ["z"], ["y"]),
        ],
        [info("x", [4, 8])], ["y"],
        [const("w1", dyadic(rs, (8, 6))), const("b1", dyadic(rs, (6,))),
         const("w2", dyadic(rs, (6, 3))), const("b2", dyadic(rs, (3,)))],
        {"x": dyadic(rs, (4, 8))},
        sigmoid=True,
    ))

    gemm_cases = {
        "gemm_plain": dict(c=(4,)),
        "gemm_no_c": dict(),
        "gemm_trans_b_alpha": dict(trans_b=1, alpha=0.5, c=(2, 4)),
        "gemm_trans_a_b_alpha_beta": dict(trans_a=1, trans_b=1, alpha=-1.5, beta=0.25, c=(1, 4)),
        "gemm_c_scalar_beta_zero": dict(beta=0.0, c=()),
        "gemm_c_column": dict(beta=-0.5, c=(2, 1)),
    }
    for seed, (name, case) in enumerate(gemm_cases.items(), start=10):
        rs = np.random.RandomState(seed)
        m, k, n = 2, 3, 4
        ta, tb = case.get("trans_a", 0), case.get("trans_b", 0)
        a = dyadic(rs, (k, m) if ta else (m, k))
        b = dyadic(rs, (n, k) if tb else (k, n))
        args = ["a", "b"]
        inits = []
        if "c" in case:
            inits.append(const("c", dyadic(rs, case["c"])))
            args.append("c")
        attrs = {"transA": ta, "transB": tb,
                 "alpha": case.get("alpha", 1.0), "beta": case.get("beta", 1.0)}
        out.append(Fixture(
            name, [helper.make_node("Gemm", args, ["y"], **attrs)],
            [info("a", list(a.shape)), info("b", list(b.shape))], ["y"], inits,
            {"a": a, "b": b},
        ))

    rs = np.random.RandomState(20)
    out.append(Fixture(
        "add_mul_broadcast_constants",
        [
            helper.make_node("Add", ["x", "row"], ["a"]),
            helper.make_node("Mul", ["col", "a"], ["m"]),
            helper.make_node("Add", ["m", "s"], ["y"]),
        ],
        [info("x", [3, 4])], ["y"],
        [const("row", dyadic(rs, (4,))), const("col", dyadic(rs, (3, 1))),
         const("s", dyadic(rs, ()))],
        {"x": dyadic(rs, (3, 4))},
    ))

    rs = np.random.RandomState(21)
    out.append(Fixture(
        "reshape_flatten_transpose",
        [
            helper.make_node("Reshape", ["x", "shape"], ["r"]),
            helper.make_node("Transpose", ["r"], ["t"], perm=[1, 0, 2]),
            helper.make_node("Flatten", ["t"], ["f"], axis=2),
            helper.make_node("Transpose", ["f"], ["y"]),
        ],
        [info("x", [2, 3, 4])], ["y"],
        [numpy_helper.from_array(np.array([0, -1, 2], dtype=np.int64), "shape")],
        {"x": dyadic(rs, (2, 3, 4))},
    ))

    rs = np.random.RandomState(22)
    out.append(Fixture(
        "identity_dropout_outputs",
        [
            helper.make_node("Identity", ["x"], ["i"]),
            helper.make_node("Dropout", ["i", "ratio"], ["d"]),
            helper.make_node("Relu", ["d"], ["y"]),
        ],
        [info("x", [2, 3])], ["y", "i", "x"],
        [const("ratio", np.float32(0.5))],
        {"x": dyadic(rs, (2, 3))},
    ))

    rs = np.random.RandomState(23)
    out.append(Fixture(
        "constant_nodes",
        [
            helper.make_node("Constant", [], ["k"], value=numpy_helper.from_array(
                dyadic(rs, (3,)), "kv")),
            helper.make_node("Constant", [], ["s"], value_float=0.75),
            helper.make_node("Add", ["x", "k"], ["a"]),
            helper.make_node("Mul", ["a", "s"], ["y"]),
        ],
        [info("x", [2, 3])], ["y"], [],
        {"x": dyadic(rs, (2, 3))},
    ))

    rs = np.random.RandomState(24)
    out.append(Fixture(
        "named_dimension",
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        [info("x", ["N", 3])], ["y"],
        [const("w", dyadic(rs, (3, 2)))],
        {"x": dyadic(rs, (5, 3))},
        dims={"N": 5},
    ))

    rs = np.random.RandomState(25)
    out.append(Fixture(
        "initializer_listed_as_input",
        [helper.make_node("Add", ["x", "w"], ["y"])],
        [info("x", [2, 2]), info("w", [2, 2])], ["y"],
        [const("w", dyadic(rs, (2, 2)))],
        {"x": dyadic(rs, (2, 2))},
    ))
    return out


def to_double(model):
    """A copy of `model` with every float32 tensor and type turned into float64."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for t in m.graph.initializer:
        if t.data_type == TensorProto.FLOAT:
            t.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(t).astype(np.float64), t.name))
    for vi in list(m.graph.input) + list(m.graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.DOUBLE
    for node in m.graph.node:
        if node.op_type == "Constant" and node.attribute[0].name == "value_float":
            scalar = np.float64(node.attribute[0].f)
            del node.attribute[:]
            node.attribute.append(
                helper.make_attribute("value", numpy_helper.from_array(np.array(scalar), "k")))
        for attr in node.attribute:
            if attr.t.data_type == TensorProto.FLOAT:
                attr.t.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(attr.t).astype(np.float64), attr.t.name))
    return m


def run_reference(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def run_ort(model, feeds):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(
        model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def build(fx, ort_run=run_ort, ref_run=run_reference):
    """The three evaluations, checked against each other. Returns the JSON dict."""
    feeds32 = {k: v for k, v in fx.feeds.items()}
    feeds64 = {k: v.astype(np.float64) for k, v in fx.feeds.items()}
    ort32 = ort_run(fx.model, feeds32)
    ref32 = ref_run(fx.model, feeds32)
    ref64 = ref_run(to_double(fx.model), feeds64)
    tol = fx.tolerance
    for i, name in enumerate(fx.outputs):
        want = np.asarray(ref64[i], dtype=np.float64)
        for label, got in (("onnxruntime", ort32[i]), ("reference evaluator", ref32[i])):
            got = np.asarray(got, dtype=np.float32)
            if got.shape != want.shape:
                raise Disagreement(f"{fx.name}/{name}: {label} shape {got.shape} != {want.shape}")
            if tol is EXACT:
                ok = np.array_equal(got.astype(np.float64), want.astype(np.float32).astype(np.float64))
            else:
                ok = np.allclose(got, want, rtol=2 * tol["rel"], atol=2 * tol["abs"])
            if not ok:
                raise Disagreement(f"{fx.name}/{name}: {label} disagrees with the float64 run")
    return {
        "name": fx.name,
        "opset": OPSET,
        "dims": fx.dims,
        "tolerance": tol,
        "inputs": [
            {"name": k, "shape": list(v.shape), "data": [float(x) for x in v.reshape(-1)]}
            for k, v in fx.feeds.items()
        ],
        "outputs": [
            {"name": n, "shape": list(np.asarray(ref64[i]).shape),
             "data": [float(x) for x in np.asarray(ref64[i], dtype=np.float64).reshape(-1)]}
            for i, n in enumerate(fx.outputs)
        ],
    }


def render(fx, expected):
    return fx.model.SerializeToString(), (json.dumps(expected, indent=1, sort_keys=True) + "\n").encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for fx in fixtures():
        onnx_bytes, json_bytes = render(fx, build(fx))
        for suffix, data in ((".onnx", onnx_bytes), (".expected.json", json_bytes)):
            path = OUT / (fx.name + suffix)
            if args.check:
                if not path.exists() or path.read_bytes() != data:
                    stale.append(path.name)
            else:
                path.write_bytes(data)
    if args.check:
        wanted = {fx.name for fx in fixtures()}
        for path in OUT.iterdir():
            if path.name.split(".")[0] not in wanted:
                stale.append(path.name + " (no generator)")
        if stale:
            print("stale or missing fixtures:", ", ".join(sorted(stale)), file=sys.stderr)
            return 1
        print(f"ok: {len(wanted)} onnx fixtures match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
