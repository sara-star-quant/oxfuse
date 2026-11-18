//! Lowering: a checked `Model` into an oxfuse `Graph`. Every op maps onto
//! the existing IR ops; `Add`, `Mul`, `Sub` and `Div` broadcast in the IR, so
//! an operand of any kind (an initializer, an input or a computed value) is
//! passed as is.

use crate::{
    ImportOptions, OnnxError,
    model::{Model, Node},
    proto::{AttributeProto, attribute_type},
    tensor::{Dense, read_tensor},
};
use oxfuse_core::{Graph, GraphBuilder, Op, ValueId, shape::numpy_broadcast};
use std::{collections::HashMap, sync::Arc};

/// A value during lowering: a float tensor in the IR, or an int64 vector
/// that may only feed a shape operand.
#[derive(Clone)]
enum Val {
    Tensor(T),
    Ints { dims: Vec<usize>, data: Vec<i64> },
}

#[derive(Clone)]
struct T {
    id: ValueId,
    shape: Vec<usize>,
    /// Present for constants, so Reshape and the Gemm scaling can read them.
    data: Option<Arc<[f32]>>,
}

pub fn lower(model: Model, opts: &ImportOptions) -> Result<Graph, OnnxError> {
    let mut cx = Cx {
        b: GraphBuilder::new(opts.limits.clone()),
        opts,
        values: HashMap::new(),
        initializers: model.initializers.into_iter().collect(),
        total: 0,
    };
    for (name, shape) in &model.inputs {
        let id = cx.b.input(name, shape.clone())?;
        cx.total = cx.total.saturating_add(shape.iter().product::<usize>());
        cx.values.insert(
            name.clone(),
            Val::Tensor(T {
                id,
                shape: shape.clone(),
                data: None,
            }),
        );
    }
    for node in &model.nodes {
        cx.node(node)?;
    }
    let mut outputs = Vec::with_capacity(model.outputs.len());
    for name in &model.outputs {
        let t = cx.tensor(name, name)?;
        outputs.push((name.clone(), t.id));
    }
    Ok(cx.b.finish_named(outputs)?)
}

struct Cx<'a> {
    b: GraphBuilder,
    opts: &'a ImportOptions,
    values: HashMap<String, Val>,
    /// Initializers not yet turned into IR constants (they are created on
    /// first use, so an unused one never reaches the graph).
    initializers: HashMap<String, Dense>,
    /// Elements of every IR value created so far, to check expansions
    /// against `max_total_elements` before they allocate.
    total: usize,
}

fn shape_err(node: &Node, reason: impl Into<String>) -> OnnxError {
    OnnxError::Shape {
        node: node.label.clone(),
        reason: reason.into(),
    }
}

impl Cx<'_> {
    fn get(&mut self, name: &str) -> Result<Val, OnnxError> {
        if let Some(v) = self.values.get(name) {
            return Ok(v.clone());
        }
        let dense = self
            .initializers
            .remove(name)
            .ok_or_else(|| OnnxError::Structure {
                reason: format!("value `{name}` is not defined"),
            })?;
        let val = match dense {
            Dense::F32 { dims, data } => {
                let data: Arc<[f32]> = data.into();
                let id = self.b.constant_shared(name, dims.clone(), data.clone())?;
                self.total = self.total.saturating_add(data.len());
                Val::Tensor(T {
                    id,
                    shape: dims,
                    data: Some(data),
                })
            }
            Dense::I64 { dims, data } => Val::Ints { dims, data },
        };
        self.values.insert(name.to_owned(), val.clone());
        Ok(val)
    }

    /// A float tensor operand; int64 data is rejected naming the value.
    fn tensor(&mut self, what: &str, name: &str) -> Result<T, OnnxError> {
        match self.get(name)? {
            Val::Tensor(t) => Ok(t),
            Val::Ints { .. } => Err(OnnxError::Dtype {
                what: what.to_owned(),
                dtype: "INT64 (only a Reshape shape may be int64)".into(),
            }),
        }
    }

    fn define(&mut self, name: &str, val: Val) {
        self.values.insert(name.to_owned(), val);
    }

    fn emit(&mut self, name: &str, op: Op, args: &[&T], shape: Vec<usize>) -> Result<T, OnnxError> {
        let ids: Vec<ValueId> = args.iter().map(|t| t.id).collect();
        let id = self.b.node(name, op, &ids)?;
        self.total = self.total.saturating_add(shape.iter().product::<usize>());
        Ok(T {
            id,
            shape,
            data: None,
        })
    }

    fn node(&mut self, n: &Node) -> Result<(), OnnxError> {
        let mut attrs = Attrs::new(n)?;
        match n.op.as_str() {
            "Relu" => self.unary(n, Op::Relu)?,
            "Sigmoid" => self.unary(n, Op::Sigmoid)?,
            "MatMul" => self.matmul(n)?,
            "Add" => self.binary(n, Op::Add)?,
            "Mul" => self.binary(n, Op::Mul)?,
            "Sub" => self.binary(n, Op::Sub)?,
            "Div" => self.binary(n, Op::Div)?,
            "Neg" => self.unary(n, Op::Neg)?,
            "Min" => self.fold(n, Op::Min)?,
            "Max" => self.fold(n, Op::Max)?,
            "Clip" => self.clip(n)?,
            "Gemm" => self.gemm(n, &mut attrs)?,
            "Reshape" => self.reshape(n, &mut attrs)?,
            "Flatten" => self.flatten(n, &mut attrs)?,
            "Transpose" => self.transpose(n, &mut attrs)?,
            "Identity" => self.identity(n)?,
            "Dropout" => self.dropout(n, &mut attrs)?,
            "Constant" => self.constant(n, &mut attrs)?,
            other => {
                return Err(OnnxError::UnsupportedOp {
                    node: n.label.clone(),
                    op: other.to_owned(),
                });
            }
        }
        attrs.finish()
    }

    /// Exactly `K` non-empty inputs and one output, returned by name.
    fn arity<'a, const K: usize>(
        &self,
        n: &'a Node,
    ) -> Result<([&'a String; K], &'a String), OnnxError> {
        let inputs: Option<[&String; K]> = if n.inputs.iter().any(String::is_empty) {
            None
        } else {
            n.inputs.iter().collect::<Vec<_>>().try_into().ok()
        };
        let Some(inputs) = inputs else {
            return Err(shape_err(
                n,
                format!(
                    "{} takes {K} input(s), got {}",
                    n.op,
                    n.inputs.iter().filter(|s| !s.is_empty()).count()
                ),
            ));
        };
        let [out] = n.outputs.as_slice() else {
            return Err(shape_err(
                n,
                format!("{} produces 1 output, got {}", n.op, n.outputs.len()),
            ));
        };
        Ok((inputs, out))
    }

    fn unary(&mut self, n: &Node, op: Op) -> Result<(), OnnxError> {
        let ([x_name], out) = self.arity::<1>(n)?;
        let x = self.tensor(x_name, x_name)?;
        let y = self.emit(out, op, &[&x], x.shape.clone())?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    fn matmul(&mut self, n: &Node) -> Result<(), OnnxError> {
        let ([a_name, b_name], out) = self.arity::<2>(n)?;
        let a = self.tensor(a_name, a_name)?;
        let b = self.tensor(b_name, b_name)?;
        let ([m, _], [_, p]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(shape_err(
                n,
                format!(
                    "MatMul supports rank 2 only, got ranks {} and {}",
                    a.shape.len(),
                    b.shape.len()
                ),
            ));
        };
        let shape = vec![*m, *p];
        let y = self.emit(out, Op::MatMul, &[&a, &b], shape)?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    fn binary(&mut self, n: &Node, op: Op) -> Result<(), OnnxError> {
        let ([a_name, b_name], out) = self.arity::<2>(n)?;
        let a = self.tensor(a_name, a_name)?;
        let b = self.tensor(b_name, b_name)?;
        let shape = self.broadcast_shape(n, &a, &b)?;
        let y = self.emit(out, op, &[&a, &b], shape)?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    /// `Min` and `Max` take two or more operands that ONNX broadcasts together.
    /// Both are associative and commutative bit for bit, so folding the operands
    /// pairwise from the left gives the joint result (and the joint shape).
    fn fold(&mut self, n: &Node, op: Op) -> Result<(), OnnxError> {
        let given = n.inputs.iter().filter(|s| !s.is_empty()).count();
        if given != n.inputs.len() || given < 2 {
            return Err(shape_err(
                n,
                format!("{} takes at least 2 inputs, got {given}", n.op),
            ));
        }
        let ([first, rest @ ..], [out]) = (n.inputs.as_slice(), n.outputs.as_slice()) else {
            return Err(shape_err(
                n,
                format!("{} produces 1 output, got {}", n.op, n.outputs.len()),
            ));
        };
        let out = out.clone();
        let mut acc = self.tensor(first, first)?;
        for (i, name) in rest.iter().enumerate() {
            let next = self.tensor(name, name)?;
            let shape = self.broadcast_shape(n, &acc, &next)?;
            // Node names only keep the steps apart; the output keeps its ONNX name.
            acc = self.emit(
                &format!("{out}.fold{}", i + 1),
                op.clone(),
                &[&acc, &next],
                shape,
            )?;
        }
        self.define(&out, Val::Tensor(acc));
        Ok(())
    }

    /// `Clip(x, min, max)` is `Min(Max(x, min), max)`; an omitted bound skips its
    /// step, so with neither the result is `x` itself. Both bounds are scalars.
    /// With `min` above `max` every element becomes `max`, as in ONNX.
    fn clip(&mut self, n: &Node) -> Result<(), OnnxError> {
        let [x_name, bounds @ ..] = n.inputs.as_slice() else {
            return Err(shape_err(n, "Clip takes 1 to 3 inputs"));
        };
        if bounds.len() > 2 || x_name.is_empty() {
            return Err(shape_err(n, "Clip takes 1 to 3 inputs"));
        }
        let [out] = n.outputs.as_slice() else {
            return Err(shape_err(n, "Clip produces 1 output"));
        };
        let out = out.clone();
        let mut y = self.tensor(x_name, x_name)?;
        let steps = [
            (bounds.first(), Op::Max, "min"),
            (bounds.get(1), Op::Min, "max"),
        ];
        let present: Vec<_> = steps
            .into_iter()
            .filter_map(|(name, op, what)| name.filter(|s| !s.is_empty()).map(|s| (s, op, what)))
            .collect();
        for (name, op, what) in &present {
            let bound = self.tensor(name, name)?;
            if !bound.shape.is_empty() {
                return Err(shape_err(
                    n,
                    format!("Clip: {what} must be a scalar, got shape {:?}", bound.shape),
                ));
            }
            let shape = y.shape.clone();
            y = self.emit(&format!("{out}.{what}"), op.clone(), &[&y, &bound], shape)?;
        }
        self.define(&out, Val::Tensor(y));
        Ok(())
    }

    /// The NumPy broadcast of two operand shapes, or the error ONNX gives for
    /// shapes that do not match.
    fn broadcast_shape(&self, n: &Node, a: &T, b: &T) -> Result<Vec<usize>, OnnxError> {
        numpy_broadcast(&a.shape, &b.shape).ok_or_else(|| {
            shape_err(
                n,
                format!(
                    "{}: shapes {:?} and {:?} cannot be broadcast",
                    n.op, a.shape, b.shape
                ),
            )
        })
    }

    /// `Y = alpha * A' * B' + beta * C`. With `alpha = beta = 1` and no
    /// transposes this is exactly `MatMul` then `Add`; every other setting
    /// adds one `Transpose` or one multiplication by a constant, never a
    /// dropped or rounded `alpha` or `beta`.
    fn gemm(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let alpha = attrs.float("alpha")?.unwrap_or(1.0);
        let beta = attrs.float("beta")?.unwrap_or(1.0);
        let trans_a = attrs.flag(n, "transA")?;
        let trans_b = attrs.flag(n, "transB")?;
        let [a_name, b_name, rest @ ..] = n.inputs.as_slice() else {
            return Err(shape_err(n, "Gemm takes 2 or 3 inputs"));
        };
        if rest.len() > 1 || a_name.is_empty() || b_name.is_empty() {
            return Err(shape_err(n, "Gemm takes 2 or 3 inputs"));
        }
        let c_name = rest.first().filter(|s| !s.is_empty());
        let has_c = c_name.is_some();
        let [out] = n.outputs.as_slice() else {
            return Err(shape_err(n, "Gemm produces 1 output"));
        };
        let out = out.clone();
        let mut a = self.tensor(a_name, a_name)?;
        let mut b = self.tensor(b_name, b_name)?;
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(shape_err(
                n,
                format!(
                    "Gemm supports rank 2 only, got ranks {} and {}",
                    a.shape.len(),
                    b.shape.len()
                ),
            ));
        }
        if trans_a {
            a = self.transpose2(n, &format!("{out}.transA"), &a)?;
        }
        if trans_b {
            b = self.transpose2(n, &format!("{out}.transB"), &b)?;
        }
        let ([m, _], [_, p]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(shape_err(n, "Gemm supports rank 2 only"));
        };
        let (m, p) = (*m, *p);
        let scaled = !is_one(alpha);
        // ONNX adds beta * C only when beta is not zero, so a zero beta drops
        // C even where C holds infinity or NaN. C is still checked.
        let adds_c = has_c && !is_zero(beta);
        let c = if let Some(c_name) = c_name {
            let c = self.tensor(c_name, c_name)?;
            let shape = [m, p];
            if numpy_broadcast(&shape, &c.shape).as_deref() != Some(shape.as_slice()) {
                return Err(shape_err(
                    n,
                    format!(
                        "Gemm: C of shape {:?} cannot be broadcast to the output shape {shape:?}",
                        c.shape
                    ),
                ));
            }
            Some(c)
        } else {
            None
        };
        let name = |last: bool, suffix: &str| {
            if last {
                out.clone()
            } else {
                format!("{out}.{suffix}")
            }
        };
        let shape = vec![m, p];
        let mut y = self.emit(
            &name(!scaled && !adds_c, "matmul"),
            Op::MatMul,
            &[&a, &b],
            shape.clone(),
        )?;
        if scaled {
            let k = self.scalar(&format!("{out}.alpha"), alpha)?;
            y = self.emit(&name(!adds_c, "scaled"), Op::Mul, &[&y, &k], shape.clone())?;
        }
        if let Some(mut c) = c.filter(|_| adds_c) {
            if !is_one(beta) {
                c = self.scale(&format!("{out}.beta"), c, beta)?;
            }
            let y = self.emit(&out, Op::Add, &[&y, &c], shape)?;
            self.define(&out, Val::Tensor(y));
        } else {
            self.define(&out, Val::Tensor(y));
        }
        Ok(())
    }

    /// The shape operand must be an int64 initializer or Constant: a value
    /// known at import, so every output shape is known too.
    fn reshape(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let allowzero = attrs.flag(n, "allowzero")?;
        let ([x_name, spec_name], out) = self.arity::<2>(n)?;
        let x = self.tensor(x_name, x_name)?;
        let spec = match self.get(spec_name)? {
            Val::Ints { dims, data } if dims.len() == 1 => data,
            Val::Ints { .. } => {
                return Err(shape_err(
                    n,
                    "Reshape shape operand must be a rank 1 tensor",
                ));
            }
            Val::Tensor(t) if t.data.is_some() => {
                return Err(shape_err(
                    n,
                    "Reshape shape operand must be int64, got float32",
                ));
            }
            Val::Tensor(_) => {
                return Err(shape_err(
                    n,
                    "Reshape shape operand must be an initializer or Constant, not a runtime value",
                ));
            }
        };
        let target = resolve_reshape(n, &x.shape, &spec, allowzero)?;
        let y = self.emit(
            out,
            Op::Reshape {
                shape: target.clone(),
            },
            &[&x],
            target,
        )?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    fn flatten(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let axis = attrs.int("axis")?.unwrap_or(1);
        let ([x_name], out) = self.arity::<1>(n)?;
        let x = self.tensor(x_name, x_name)?;
        let rank = x.shape.len() as i64;
        let split = usize::try_from(if axis < 0 { axis + rank } else { axis })
            .ok()
            .and_then(|a| x.shape.split_at_checked(a));
        let Some((head, tail)) = split else {
            return Err(shape_err(
                n,
                format!("Flatten axis {axis} is out of range for rank {rank}"),
            ));
        };
        let outer: usize = head.iter().product();
        let inner: usize = tail.iter().product();
        let target = vec![outer, inner];
        let y = self.emit(
            out,
            Op::Reshape {
                shape: target.clone(),
            },
            &[&x],
            target,
        )?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    fn transpose(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let perm = attrs.ints("perm")?;
        let ([x_name], out) = self.arity::<1>(n)?;
        let x = self.tensor(x_name, x_name)?;
        let rank = x.shape.len();
        let perm: Vec<usize> = match perm.filter(|p| !p.is_empty()) {
            None => (0..rank).rev().collect(),
            Some(p) => {
                let valid = p.len() == rank
                    && p.iter().all(|v| (0..rank as i64).contains(v))
                    && (0..rank as i64).all(|i| p.contains(&i));
                if !valid {
                    return Err(shape_err(
                        n,
                        format!("Transpose perm {p:?} is not a permutation of 0..{rank}"),
                    ));
                }
                p.iter().map(|v| *v as usize).collect()
            }
        };
        let shape: Option<Vec<usize>> = perm.iter().map(|i| x.shape.get(*i).copied()).collect();
        let Some(shape) = shape else {
            return Err(shape_err(
                n,
                format!("Transpose perm {perm:?} is not a permutation of 0..{rank}"),
            ));
        };
        let y = self.emit(out, Op::Transpose { perm }, &[&x], shape)?;
        self.define(out, Val::Tensor(y));
        Ok(())
    }

    fn scalar(&mut self, name: &str, v: f32) -> Result<T, OnnxError> {
        let data: Arc<[f32]> = vec![v].into();
        let id = self.b.constant_shared(name, vec![], data.clone())?;
        self.total = self.total.saturating_add(1);
        Ok(T {
            id,
            shape: vec![],
            data: Some(data),
        })
    }

    fn transpose2(&mut self, n: &Node, name: &str, t: &T) -> Result<T, OnnxError> {
        let [rows, cols] = t.shape.as_slice() else {
            return Err(shape_err(n, "Gemm supports rank 2 only"));
        };
        let shape = vec![*cols, *rows];
        self.emit(name, Op::Transpose { perm: vec![1, 0] }, &[t], shape)
    }

    /// `k * t`: a constant is scaled at import (the same float32 product the
    /// runtime would compute), a runtime tensor gets a `Mul` node.
    fn scale(&mut self, name: &str, t: T, k: f32) -> Result<T, OnnxError> {
        if let Some(data) = &t.data {
            let scaled: Arc<[f32]> = data.iter().map(|v| k * v).collect::<Vec<_>>().into();
            let id = self
                .b
                .constant_shared(name, t.shape.clone(), scaled.clone())?;
            self.total = self.total.saturating_add(scaled.len());
            return Ok(T {
                id,
                shape: t.shape,
                data: Some(scaled),
            });
        }
        let k = self.scalar(&format!("{name}.k"), k)?;
        let shape = t.shape.clone();
        self.emit(name, Op::Mul, &[&k, &t], shape)
    }

    fn identity(&mut self, n: &Node) -> Result<(), OnnxError> {
        let ([x_name], out) = self.arity::<1>(n)?;
        let v = self.get(x_name)?;
        self.define(out, v);
        Ok(())
    }

    /// Inference-mode Dropout is the identity. `ratio` is ignored at
    /// inference; a `training_mode` input or a `mask` output is an error.
    fn dropout(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        attrs.int("seed")?;
        let ([x_name, ..], [out, rest @ ..]) = (n.inputs.as_slice(), n.outputs.as_slice()) else {
            return Err(shape_err(
                n,
                "Dropout takes 1 to 3 inputs and produces an output",
            ));
        };
        if n.inputs.len() > 3 || x_name.is_empty() {
            return Err(shape_err(n, "Dropout takes 1 to 3 inputs"));
        }
        if n.inputs.get(2).is_some_and(|s| !s.is_empty()) {
            return Err(shape_err(
                n,
                "Dropout training_mode input is not supported (inference only)",
            ));
        }
        if rest.len() > 1 || rest.iter().any(|s| !s.is_empty()) {
            return Err(shape_err(n, "Dropout mask output is not supported"));
        }
        let v = self.get(x_name)?;
        self.define(out, v);
        Ok(())
    }

    fn constant(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let [out] = n.outputs.as_slice() else {
            return Err(shape_err(
                n,
                "Constant takes no inputs and produces 1 output",
            ));
        };
        if !n.inputs.is_empty() {
            return Err(shape_err(
                n,
                "Constant takes no inputs and produces 1 output",
            ));
        }
        let limits = &self.opts.limits;
        let mut found: Vec<Dense> = Vec::new();
        if let Some(t) = attrs.tensor("value")? {
            let mut t = t.clone();
            t.name = out.clone();
            found.push(read_tensor(&t, limits)?);
        }
        if let Some(f) = attrs.float("value_float")? {
            found.push(Dense::F32 {
                dims: vec![],
                data: vec![f],
            });
        }
        if let Some(fs) = attrs.floats("value_floats")? {
            found.push(Dense::F32 {
                dims: vec![fs.len()],
                data: fs,
            });
        }
        if let Some(i) = attrs.int("value_int")? {
            found.push(Dense::I64 {
                dims: vec![],
                data: vec![i],
            });
        }
        if let Some(is) = attrs.ints("value_ints")? {
            found.push(Dense::I64 {
                dims: vec![is.len()],
                data: is,
            });
        }
        let dense = match found.len() {
            1 => found.remove(0),
            0 => return Err(shape_err(n, "Constant has no supported value attribute")),
            _ => return Err(shape_err(n, "Constant sets more than one value attribute")),
        };
        self.initializers.insert(out.clone(), dense);
        self.get(out)?;
        Ok(())
    }
}

/// Resolve `0` and `-1` entries of a Reshape shape against the input shape.
/// All arithmetic is checked; the caller's builder re-verifies the element
/// count.
fn resolve_reshape(
    n: &Node,
    input: &[usize],
    spec: &[i64],
    allowzero: bool,
) -> Result<Vec<usize>, OnnxError> {
    let infer = spec.iter().filter(|v| **v == -1).count();
    if infer > 1 {
        return Err(shape_err(n, "Reshape shape has more than one -1"));
    }
    if allowzero && infer == 1 && spec.contains(&0) {
        return Err(shape_err(
            n,
            "Reshape with allowzero=1 cannot combine 0 and -1 in the shape",
        ));
    }
    let overflow = || shape_err(n, "Reshape element count overflows");
    let mut dims: Vec<Option<usize>> = Vec::with_capacity(spec.len());
    let mut known: usize = 1;
    for (i, v) in spec.iter().enumerate() {
        let d = match *v {
            -1 => None,
            0 if allowzero => {
                return Err(shape_err(
                    n,
                    "Reshape allowzero=1 asks for a zero-size dimension; zero-size tensors are not supported",
                ));
            }
            0 => Some(*input.get(i).ok_or_else(|| {
                shape_err(
                    n,
                    format!(
                        "Reshape shape entry {i} is 0 but the input has rank {}",
                        input.len()
                    ),
                )
            })?),
            // Only a negative entry fails the conversion; -1 and 0 are matched above.
            v => Some(
                usize::try_from(v)
                    .map_err(|_| shape_err(n, format!("Reshape shape entry {i} is {v}")))?,
            ),
        };
        if let Some(d) = d {
            known = known.checked_mul(d).ok_or_else(overflow)?;
        }
        dims.push(d);
    }
    let total = input
        .iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .ok_or_else(overflow)?;
    let inferred = if infer == 1 {
        if known == 0 || total % known != 0 {
            return Err(shape_err(
                n,
                format!(
                    "Reshape cannot infer -1: {total} elements do not divide into dimensions of product {known}"
                ),
            ));
        }
        total / known
    } else {
        0
    };
    Ok(dims.into_iter().map(|d| d.unwrap_or(inferred)).collect())
}

/// True for `+0.0` and `-0.0`.
fn is_zero(v: f32) -> bool {
    v == 0.0
}

/// Bit comparison, so only exactly `1.0` skips the scaling (not `-0.0`, not NaN).
fn is_one(v: f32) -> bool {
    v.to_bits() == 1.0f32.to_bits()
}

/// A node's attributes. A handler takes the ones it understands; `finish`
/// fails on any left over, so nothing is silently ignored.
struct Attrs<'a> {
    node: &'a Node,
    items: Vec<(&'a AttributeProto, bool)>,
}

impl<'a> Attrs<'a> {
    fn new(node: &'a Node) -> Result<Self, OnnxError> {
        let items: Vec<_> = node.attrs.iter().map(|a| (a, false)).collect();
        let this = Self { node, items };
        for (a, _) in &this.items {
            if !a.ref_attr_name.is_empty() || a.g.is_some() {
                return Err(this.unsupported(&a.name));
            }
        }
        Ok(this)
    }

    fn unsupported(&self, attr: &str) -> OnnxError {
        OnnxError::UnsupportedAttr {
            node: self.node.label.clone(),
            op: self.node.op.clone(),
            attr: attr.to_owned(),
        }
    }

    fn take(&mut self, name: &str, ty: i32) -> Result<Option<&'a AttributeProto>, OnnxError> {
        let Some((a, used)) = self.items.iter_mut().find(|(a, _)| a.name == name) else {
            return Ok(None);
        };
        *used = true;
        if a.r#type != 0 && a.r#type != ty {
            return Err(shape_err(
                self.node,
                format!("attribute `{name}` has type {}, expected {ty}", a.r#type),
            ));
        }
        let has_value = match ty {
            attribute_type::INT => a.i.is_some(),
            attribute_type::FLOAT => a.f.is_some(),
            attribute_type::TENSOR => a.t.is_some(),
            attribute_type::INTS => a.r#type == ty || !a.ints.is_empty(),
            attribute_type::FLOATS => a.r#type == ty || !a.floats.is_empty(),
            _ => true,
        };
        if !has_value {
            return Err(shape_err(
                self.node,
                format!("attribute `{name}` has no value of type {ty}"),
            ));
        }
        Ok(Some(*a))
    }

    fn int(&mut self, name: &str) -> Result<Option<i64>, OnnxError> {
        Ok(self.take(name, attribute_type::INT)?.and_then(|a| a.i))
    }

    fn float(&mut self, name: &str) -> Result<Option<f32>, OnnxError> {
        Ok(self.take(name, attribute_type::FLOAT)?.and_then(|a| a.f))
    }

    /// An integer attribute that must be 0 or 1 (default 0).
    fn flag(&mut self, node: &Node, name: &str) -> Result<bool, OnnxError> {
        match self.int(name)? {
            None | Some(0) => Ok(false),
            Some(1) => Ok(true),
            Some(v) => Err(shape_err(
                node,
                format!("attribute `{name}` is {v}, expected 0 or 1"),
            )),
        }
    }

    fn ints(&mut self, name: &str) -> Result<Option<Vec<i64>>, OnnxError> {
        Ok(self
            .take(name, attribute_type::INTS)?
            .map(|a| a.ints.clone()))
    }

    fn floats(&mut self, name: &str) -> Result<Option<Vec<f32>>, OnnxError> {
        Ok(self
            .take(name, attribute_type::FLOATS)?
            .map(|a| a.floats.clone()))
    }

    fn tensor(&mut self, name: &str) -> Result<Option<&'a crate::proto::TensorProto>, OnnxError> {
        Ok(self
            .take(name, attribute_type::TENSOR)?
            .and_then(|a| a.t.as_ref()))
    }

    fn finish(self) -> Result<(), OnnxError> {
        match self.items.iter().find(|(_, used)| !used) {
            Some((a, _)) => Err(self.unsupported(&a.name)),
            None => Ok(()),
        }
    }
}
