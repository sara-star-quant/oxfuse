#!/usr/bin/env python3
"""Independent oracle for Oxfuse.

Evaluates graph files in float64 with NumPy and writes or checks the expected
outputs. It shares no code with the Rust implementation, so agreement between
the two is evidence, not tautology.

Usage:
  python3 -I scripts/gen_expected.py examples [--check]
  python3 -I scripts/gen_expected.py corpus [--count N] [--seed S] [--check]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOOSE = {"abs": 1e-6, "rel": 1e-6}
EXACT = {"abs": 0.0, "rel": 0.0}
DYADIC = [i / 4 for i in range(-8, 9)] + [-0.0]  # -2.0 .. 2.0 by 0.25, plus a signed zero
INEXACT_COUNT = 10  # MatMul graphs whose sums round, checked against a bound


class Inexact(Exception):
    """A graph whose intermediates are not exactly representable in f32."""


def f32(values):
    """JSON number -> f64 -> f32 -> f64, the parsing rule in the numerical contract."""
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


def apply(node, args):
    op = node["op"]
    attrs = node.get("attrs", {})
    if op == "Relu":
        return np.where(args[0] > 0, args[0], 0.0)
    if op == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-args[0]))
    if op == "Add":
        return args[0] + args[1]
    if op == "Mul":
        return args[0] * args[1]
    if op == "MatMul":
        return args[0] @ args[1]
    if op == "Reshape":
        return args[0].reshape(attrs["shape"])
    if op == "Transpose":
        return np.transpose(args[0], attrs["perm"])
    raise SystemExit(f"unsupported op {op}")


def evaluate(graph, inputs):
    env = {}
    for c in graph.get("constants", []):
        env[c["name"]] = f32(c["values"]).reshape(c["shape"])
    for i in graph.get("inputs", []):
        t = inputs[i["name"]]
        env[i["name"]] = f32(t["values"]).reshape(t["shape"])
    nodes = {n["name"]: n for n in graph["nodes"]}

    def value(name):
        if name not in env:
            node = nodes[name]
            env[name] = np.asarray(apply(node, [value(a) for a in node["inputs"]]))
        return env[name]

    outputs = [(name, value(name)) for name in graph["outputs"]]
    return outputs, env


def sigmoid_dependent(graph):
    """Returns a function: does this value depend on a Sigmoid node?"""
    nodes = {n["name"]: n for n in graph["nodes"]}
    memo = {}

    def dep(name):
        if name not in memo:
            node = nodes.get(name)
            memo[name] = node is not None and (
                node["op"] == "Sigmoid" or any(dep(i) for i in node["inputs"])
            )
        return memo[name]

    return dep


def expected_doc(graph, inputs):
    outputs, env = evaluate(graph, inputs)
    dep = sigmoid_dependent(graph)
    for name, value in env.items():
        if not np.all(np.isfinite(value)):
            raise Inexact(f"{name} is not finite")
        exact = np.array_equal(value.astype(np.float32).astype(np.float64), value)
        if not dep(name) and not exact:
            raise Inexact(f"{name} is not exactly representable in f32")
    docs = []
    for name, value in outputs:
        out = {"name": name, "shape": list(value.shape), "values": [float(x) for x in value.reshape(-1)]}
        if dep(name):
            out["tolerance"] = LOOSE  # only outputs that depend on Sigmoid are loose
        docs.append(out)
    return {"tolerance": EXACT, "outputs": docs}


def effective(out, doc):
    return out.get("tolerance", doc["tolerance"])


def close(got, want, tol):
    return got == want or abs(got - want) <= tol["abs"] + tol["rel"] * abs(want)


def same_doc(committed, computed):
    if [o["name"] for o in committed["outputs"]] != [o["name"] for o in computed["outputs"]]:
        return False
    for a, b in zip(committed["outputs"], computed["outputs"]):
        tol = effective(b, computed)
        if effective(a, committed) != tol:
            return False
        if a["shape"] != b["shape"] or len(a["values"]) != len(b["values"]):
            return False
        bound_a, bound_b = a.get("bound"), b.get("bound")
        if (bound_a is None) != (bound_b is None):
            return False
        if bound_b is None:
            if not all(close(x, y, tol) for x, y in zip(a["values"], b["values"])):
                return False
            continue
        # Sums done by a BLAS differ in the last bits between machines, so a
        # bounded output is compared inside its own bound, and the bound to
        # a relative 1e-9.
        if len(bound_a) != len(bound_b):
            return False
        if not all(abs(x - y) <= 1e-9 * abs(y) for x, y in zip(bound_a, bound_b)):
            return False
        if not all(abs(x - y) <= e for x, y, e in zip(a["values"], b["values"], bound_b)):
            return False
    return True


def write_json(path, doc):
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")


# Random graph generation --------------------------------------------------


def pick(rng, items):
    return items[int(rng.integers(0, len(items)))]


def rand_tensor(rng, shape):
    count = int(np.prod(shape, dtype=np.int64))
    return [float(pick(rng, DYADIC)) for _ in range(count)]


def rand_shape(rng):
    return [int(rng.integers(1, 5)) for _ in range(int(rng.integers(0, 4)))]


def rand_reshape(rng, count):
    dims, remaining = [], count
    while remaining > 1 and len(dims) < 3:
        d = pick(rng, [x for x in range(2, remaining + 1) if remaining % x == 0])
        dims.append(d)
        remaining //= d
    if remaining > 1:
        dims.append(remaining)
    dims += [1] * int(rng.integers(0, 2))
    rng.shuffle(dims)
    return [int(d) for d in dims]


def random_graph(rng, allow_sigmoid):
    inputs, constants, nodes, values, data = [], [], [], {}, {}

    def new_input(shape):
        name = f"x{len(inputs)}"
        inputs.append({"name": name, "dtype": "f32", "shape": shape})
        data[name] = {"shape": shape, "values": rand_tensor(rng, shape)}
        values[name] = shape
        return name

    def new_const(shape):
        name = f"c{len(constants)}"
        constants.append(
            {"name": name, "dtype": "f32", "shape": shape, "values": rand_tensor(rng, shape)}
        )
        values[name] = shape
        return name

    for _ in range(int(rng.integers(1, 3))):
        new_input(rand_shape(rng))

    for k in range(int(rng.integers(2, 7))):
        # Favor rank-2 sources and MatMul: random shapes rarely line up, and
        # MatMul is the op whose bugs matter most.
        rank2 = sorted(n for n, s in values.items() if len(s) == 2)
        if rank2 and int(rng.integers(0, 5)) < 2:
            src = pick(rng, rank2)
        else:
            src = pick(rng, sorted(values))
        shape = values[src]
        ops = ["Relu", "Add", "Mul", "Reshape", "Transpose"]
        if allow_sigmoid:
            ops.append("Sigmoid")
        if len(shape) == 2:
            ops += ["MatMul"] * 3
        op = pick(rng, ops)
        node = {"name": f"n{k}", "op": op}
        if op in ("Relu", "Sigmoid"):
            node["inputs"], out = [src], shape
        elif op in ("Add", "Mul"):
            mode = int(rng.integers(0, 3 if shape else 4))
            out = shape
            if mode == 0:
                other = new_const(shape)
            elif mode == 1:
                other = new_const([])
            elif mode == 2:
                other = pick(rng, sorted(n for n, s in values.items() if s == shape))
            else:
                other = pick(rng, sorted(values))
                out = values[other]
            node["inputs"] = [src, other] if int(rng.integers(0, 2)) else [other, src]
        elif op == "MatMul":
            partners = sorted(n for n, s in values.items() if len(s) == 2 and s[0] == shape[1])
            if partners and int(rng.integers(0, 2)):
                other = pick(rng, partners)
            else:
                other = new_const([shape[1], int(rng.integers(1, 5))])
            node["inputs"], out = [src, other], [shape[0], values[other][1]]
        elif op == "Reshape":
            target = rand_reshape(rng, int(np.prod(shape, dtype=np.int64)))
            node["inputs"], node["attrs"], out = [src], {"shape": target}, target
        else:
            perm = [int(p) for p in rng.permutation(len(shape))]
            node["inputs"], node["attrs"] = [src], {"perm": perm}
            out = [shape[p] for p in perm]
        nodes.append(node)
        values[node["name"]] = out

    # Every node nothing else consumes is an output, so the oracle checks all
    # of them; a dead node would be computed by Rust but never compared.
    consumed = {i for node in nodes for i in node["inputs"]}
    outputs = [n["name"] for n in nodes if n["name"] not in consumed]
    if int(rng.integers(0, 2)):
        extra = pick(rng, sorted(values))
        if extra not in outputs:
            outputs.append(extra)
    order = [int(i) for i in rng.permutation(len(nodes))]
    graph = {
        "format_version": 1,
        "inputs": inputs,
        "constants": constants,
        "nodes": [nodes[i] for i in order],
        "outputs": outputs,
    }
    return graph, data


def generate(seed, allow_sigmoid):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        graph, data = random_graph(rng, allow_sigmoid)
        try:
            return graph, data, expected_doc(graph, data)
        except Inexact:
            continue
    raise SystemExit(f"seed {seed}: no exactly representable graph found")


def inexact_matmul(seed):
    """One MatMul over normal-distributed f32 values across six decades.

    The sums round, so the expected file carries a per-element bound
    K * eps * sum(|a_ik * b_kj|) instead of a tolerance.
    """
    rng = np.random.default_rng(seed)
    m, k, n = (pick(rng, [1, 2, 3, 5, 8, 16, 33]) for _ in range(3))

    def tensor(rows, cols):
        scale = 10.0 ** rng.uniform(-3, 3, size=(rows, cols))
        values = rng.standard_normal((rows, cols)) * scale
        return values.astype(np.float32).astype(np.float64)

    a, b = tensor(m, k), tensor(k, n)
    want = a @ b
    bound = k * 2.0**-23 * (np.abs(a) @ np.abs(b))
    graph = {
        "format_version": 1,
        "inputs": [{"name": "x", "dtype": "f32", "shape": [m, k]}],
        "constants": [
            {"name": "w", "dtype": "f32", "shape": [k, n], "values": [float(v) for v in b.reshape(-1)]}
        ],
        "nodes": [{"name": "y", "op": "MatMul", "inputs": ["x", "w"]}],
        "outputs": ["y"],
    }
    data = {"x": {"shape": [m, k], "values": [float(v) for v in a.reshape(-1)]}}
    doc = {
        "tolerance": EXACT,
        "outputs": [
            {
                "name": "y",
                "shape": [m, n],
                "values": [float(v) for v in want.reshape(-1)],
                "bound": [float(v) for v in bound.reshape(-1)],
            }
        ],
    }
    return graph, data, doc


# Commands -----------------------------------------------------------------


def graph_files(directory):
    return [
        p
        for p in sorted(directory.glob("*.json"))
        if not p.name.endswith((".inputs.json", ".expected.json"))
    ]


def cmd_examples(check):
    failed = False
    for path in graph_files(ROOT / "examples"):
        graph = json.loads(path.read_text())
        inputs_path = path.with_name(path.stem + ".inputs.json")
        inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
        computed = expected_doc(graph, inputs)
        expected_path = path.with_name(path.stem + ".expected.json")
        if check:
            committed = json.loads(expected_path.read_text())
            ok = same_doc(committed, computed)
            print(f"{'ok  ' if ok else 'FAIL'} {path.name}")
            failed |= not ok
        else:
            write_json(expected_path, computed)
            print(f"wrote {expected_path.name}")
    return 1 if failed else 0


def compact(doc):
    return json.dumps(doc, separators=(",", ":")) + "\n"


def corpus_files(count, seed):
    files = {}

    def add(stem, graph, data, doc):
        files[f"{stem}.json"] = compact(graph)
        files[f"{stem}.inputs.json"] = compact(data)
        files[f"{stem}.expected.json"] = compact(doc)

    for k in range(count):
        add(f"g{k:03d}", *generate(seed + k, allow_sigmoid=(k % 4 == 3)))
    for k in range(INEXACT_COUNT):
        add(f"h{k:03d}", *inexact_matmul(seed + 1000 + k))
    return files


def cmd_corpus(count, seed, check):
    out = ROOT / "tests" / "corpus"
    files = corpus_files(count, seed)
    existing = {p.name for p in out.glob("[gh]*.json")} if out.exists() else set()
    if check:
        stale = sorted(existing ^ set(files)) + sorted(
            n for n, text in files.items() if n in existing and (out / n).read_text() != text
        )
        print(f"{'ok  ' if not stale else 'FAIL'} corpus ({len(files)} files)")
        for name in stale:
            print(f"     differs: {name}")
        return 1 if stale else 0
    out.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (out / name).unlink()
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"wrote {len(files)} files to {out}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("examples")
    ex.add_argument("--check", action="store_true")
    co = sub.add_parser("corpus")
    co.add_argument("--count", type=int, default=40)
    co.add_argument("--seed", type=int, default=2026)
    co.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.cmd == "examples":
        return cmd_examples(args.check)
    return cmd_corpus(args.count, args.seed, args.check)


if __name__ == "__main__":
    sys.exit(main())
