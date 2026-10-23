# Adding an op

An op touches five places. The compiler lists most of them for you: every `match` on `Op` is exhaustive, so adding a variant breaks the build until each one has an arm.

1. **`crates/core/src/op.rs`**: add the `Op` variant, its `name()` and `arity()`, and its entry in `ALL_NAMES` and in the test's `one_of_each` and `assert_listed`.
2. **`crates/core/src/shape.rs`**: add the shape rule as a function with checked arithmetic, returning `ShapeMismatch` (through `mismatch`) with a message that names the node, the op and the offending shapes. Add the arm to `infer`. Write the rule for the general case and restrict it in the function, so widening later is a one-function change.
3. **`crates/runtime/src/kernels.rs`**: add the reference kernel and the arm in `apply`. Keep it naive: it is the oracle for every optimized kernel.
4. **`crates/cli/src/loader.rs`**: map the JSON name to the variant in `parse_op`. If the op takes attributes, add the key to `Attrs` in `format.rs` and use `require`/`unexpected`.
5. **Evidence**:
   - an example in `examples/` with a hand-computed `.expected.json` (the tolerance is zero for exact ops);
   - an arm in `scripts/gen_expected.py` `apply` and in the random generator, then `just corpus` to regenerate and `just oracle-check` to confirm NumPy agrees with your hand computation and the committed corpus is current;
   - a rejecting case in `tests/invalid/` for each way the shape rule can fail, with its snapshot reviewed;
   - unit tests for the shape rule and the kernel, including rank 0, size-1 dimensions, NaN and infinity where they matter;
   - a row in the README's supported-ops table;
   - an entry for the op in `docs/ir.md` and `docs/semantics.md` (a test fails without it).
   - three rows in `docs/contract.json` (reference, cpu, gpu) with the level, the evidence and a named test for each (a test fails without them);

`cargo test --workspace` fails if an op has no example, no README row or no contract rows. Finish with `just ci` and `just oracle-check`.

A new op needs nothing for the optimization passes. Folding calls the same reference kernel through `ReferenceEvaluator`, CSE keys on the op value (so its attributes must be part of the `Op` variant and derive `Hash`), and DCE only looks at inputs. The property tests in `crates/runtime/tests/optimization_equivalence.rs` generate graphs from a fixed list of ops; add the new op there so it is covered.

## Numerical contract

State which class the op is in before writing the kernel. Exact ops (`Add`, `Mul`, `Relu`, `Reshape`, `Transpose`) must match NumPy bit for bit on exactly representable inputs. Ops that sum or approximate (`MatMul`, `Sigmoid`) need a documented error bound and a test that measures it.
