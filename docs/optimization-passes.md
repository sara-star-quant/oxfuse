# Optimization passes

`oxfuse run` and `oxfuse inspect` run three passes once, in this order: `fold`, `cse`, `dce`. Every pass is exact: the optimized graph computes bit-identical outputs to the unoptimized one on the reference backend, including NaN, infinity and signed zero. Errors are identical too: an output that overflows fails the same way with and without optimization, and an overflowing dead node fails in neither.

The `PassManager` enforces the rules around the passes:

- the graph is verified and checked against the limits before the first pass and after every pass;
- a pass that returns an invalid graph is rejected, naming the pass;
- a pass that changes the graph interface (input names, order and types; output names, order and types) is rejected, naming the pass. Output names are stored in the graph, so merging or folding a value never renames an output.

Usage: `oxfuse inspect FILE` shows the IR before and after each pass. `oxfuse run FILE --dump-ir` prints the same stages to stderr, before the error line when the run fails (if a pass itself fails, only the input stage is shown). `--no-opt` skips every pass. `--disable-pass NAME` skips one (`fold`, `cse`, `dce`); an unknown name is an error even with `--no-opt`.

## fold: constant folding

- **Purpose:** evaluate nodes whose inputs are all constants at compile time.
- **Preconditions:** a verified graph.
- **Rule:** a node with all-constant inputs, an output of at most 2^20 elements and no input over 2^20 elements is evaluated with the injected `ConstEvaluator` and replaced by a constant of the same name. The CLI injects the reference kernels, so the folded value is the value the run-time path would compute.
- **Postconditions:** same values, same names, same outputs. Folded nodes are now constants; the constants they consumed may be unused.
- **Risks:** a different evaluator would change results (use the backend's kernels); folding a huge node spends compile-time memory and time (hence the cap on outputs and inputs; for a `MatMul` whose operands and result each have at most n elements the work is at most n^1.5 multiply-adds, about 2^30 at the cap); a non-finite folded constant is legal and behaves like the run-time NaN or infinity.
- **Tests:** `passes/fold.rs` unit tests; `runtime/tests/folding.rs` against the real kernels; the `fold_chain` snapshot; the property tests; the error-parity test in `cli/tests/optimization.rs`.

## cse: common-subexpression elimination

- **Purpose:** compute a repeated expression once.
- **Rule:** a node with the same op (attributes included) and the same remapped inputs as an earlier node is aliased to it. Merging is transitive because keys use the new ids. Output names are kept, so an output merged into another value still has its own name.
- **Not merged:** inputs, constants (even with equal data), `Add(a, b)` versus `Add(b, a)`. Both are valid extensions, left out until a measured case needs them.
- **Risks:** merging nodes that differ in an attribute; losing an output name.
- **Tests:** `passes/cse.rs` unit tests; the `cse_duplicate` snapshot; the property tests.

## dce: dead-code elimination

- **Purpose:** drop what no output uses.
- **Rule:** keep every graph input, everything an output transitively uses, and nothing else.
- **Risks:** removing an input would change the interface, so inputs are always kept.
- **Tests:** `passes/dce.rs` unit tests; the `dce_dead` snapshot; the property tests.

## Why this order

Folding creates constants and leaves the nodes that fed them unused. CSE sees the folded graph. DCE runs last and collects everything the other two left behind. The pipeline runs once; iterating to a fixpoint is deferred until a measured case needs it, and the idempotence property test checks that one run already reaches the fixpoint on random graphs.

## How the passes are tested

Every graph in `examples/`, `tests/corpus/` and `tests/passes/` runs both optimized and with `--no-opt`, and the output bytes must match. `crates/runtime/tests/optimization_equivalence.rs` generates random valid graphs (inputs and constants include NaN, infinity and `-0.0`) and checks three properties over 256 cases each: bit-identical outputs, an unchanged interface, and idempotence. The generator repeats earlier steps so that duplicates exist, lists several named outputs (one value sometimes twice), and a separate test pins how often each pass has real work to do (folds in about 6 of 10 graphs, merges that reach the result in about 3 of 10). Breaking CSE so that it ignores `Reshape` attributes, or naming outputs after their values, makes the properties fail.

## Adding a pass

Implement `Pass` in `crates/core/src/passes/`, build the result with `Rebuilder` so names and output names carry over, add it to `default_pipeline`, add unit tests, a snapshot graph in `tests/passes/`, and make sure the property tests still pass. A pass that is not exact needs its own documented contract and must not run by default.
