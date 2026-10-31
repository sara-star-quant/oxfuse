//! Lowering: a checked `Model` into an oxfuse `Graph`. Every op maps onto
//! the existing IR ops; `Add` and `Mul` broadcast in the IR, so an operand of
//! any kind (an initializer, an input or a computed value) is passed as is.

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

    /// Exactly `inputs` non-empty inputs and one output.
    fn arity(&self, n: &Node, inputs: usize) -> Result<(), OnnxError> {
        if n.inputs.len() != inputs || n.inputs.iter().any(String::is_empty) {
            return Err(shape_err(
                n,
                format!(
                    "{} takes {inputs} input(s), got {}",
                    n.op,
                    n.inputs.iter().filter(|s| !s.is_empty()).count()
                ),
            ));
        }
        if n.outputs.len() != 1 {
            return Err(shape_err(
                n,
                format!("{} produces 1 output, got {}", n.op, n.outputs.len()),
            ));
        }
        Ok(())
    }

    fn unary(&mut self, n: &Node, op: Op) -> Result<(), OnnxError> {
        self.arity(n, 1)?;
        let x = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let y = self.emit(&n.outputs[0], op, &[&x], x.shape.clone())?;
        self.define(&n.outputs[0], Val::Tensor(y));
        Ok(())
    }

    fn matmul(&mut self, n: &Node) -> Result<(), OnnxError> {
        self.arity(n, 2)?;
        let a = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let b = self.tensor(&n.inputs[1], &n.inputs[1])?;
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(shape_err(
                n,
                format!(
                    "MatMul supports rank 2 only, got ranks {} and {}",
                    a.shape.len(),
                    b.shape.len()
                ),
            ));
        }
        let shape = vec![a.shape[0], b.shape[1]];
        let y = self.emit(&n.outputs[0], Op::MatMul, &[&a, &b], shape)?;
        self.define(&n.outputs[0], Val::Tensor(y));
        Ok(())
    }

    fn binary(&mut self, n: &Node, op: Op) -> Result<(), OnnxError> {
        self.arity(n, 2)?;
        let a = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let b = self.tensor(&n.inputs[1], &n.inputs[1])?;
        let shape = self.broadcast_shape(n, &a, &b)?;
        let y = self.emit(&n.outputs[0], op, &[&a, &b], shape)?;
        self.define(&n.outputs[0], Val::Tensor(y));
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
        let has_c = n.inputs.get(2).is_some_and(|s| !s.is_empty());
        if !(2..=3).contains(&n.inputs.len()) || n.inputs[..2].iter().any(String::is_empty) {
            return Err(shape_err(n, "Gemm takes 2 or 3 inputs"));
        }
        if n.outputs.len() != 1 {
            return Err(shape_err(n, "Gemm produces 1 output"));
        }
        let out = n.outputs[0].clone();
        let mut a = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let mut b = self.tensor(&n.inputs[1], &n.inputs[1])?;
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
            a = self.transpose2(&format!("{out}.transA"), &a)?;
        }
        if trans_b {
            b = self.transpose2(&format!("{out}.transB"), &b)?;
        }
        let scaled = !is_one(alpha);
        // ONNX adds beta * C only when beta is not zero, so a zero beta drops
        // C even where C holds infinity or NaN. C is still checked.
        let adds_c = has_c && !is_zero(beta);
        let c = if has_c {
            let c = self.tensor(&n.inputs[2], &n.inputs[2])?;
            let shape = [a.shape[0], b.shape[1]];
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
        let shape = vec![a.shape[0], b.shape[1]];
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
        self.arity(n, 2)?;
        let x = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let spec = match self.get(&n.inputs[1])? {
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
            &n.outputs[0],
            Op::Reshape {
                shape: target.clone(),
            },
            &[&x],
            target,
        )?;
        self.define(&n.outputs[0], Val::Tensor(y));
        Ok(())
    }

    fn flatten(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let axis = attrs.int("axis")?.unwrap_or(1);
        self.arity(n, 1)?;
        let x = self.tensor(&n.inputs[0], &n.inputs[0])?;
        let rank = x.shape.len() as i64;
        if axis < -rank || axis > rank {
            return Err(shape_err(
                n,
                format!("Flatten axis {axis} is out of range for rank {rank}"),
            ));
        }
        let axis = usize::try_from(if axis < 0 { axis + rank } else { axis })
            .expect("axis checked against the rank");
        let outer: usize = x.shape[..axis].iter().product();
        let inner: usize = x.shape[axis..].iter().product();
        let target = vec![outer, inner];
        let y = self.emit(
            &n.outputs[0],
            Op::Reshape {
                shape: target.clone(),
            },
            &[&x],
            target,
        )?;
        self.define(&n.outputs[0], Val::Tensor(y));
        Ok(())
    }

    fn transpose(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        let perm = attrs.ints("perm")?;
        self.arity(n, 1)?;
        let x = self.tensor(&n.inputs[0], &n.inputs[0])?;
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
        let shape: Vec<usize> = perm.iter().map(|i| x.shape[*i]).collect();
        let y = self.emit(&n.outputs[0], Op::Transpose { perm }, &[&x], shape)?;
        self.define(&n.outputs[0], Val::Tensor(y));
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

    fn transpose2(&mut self, name: &str, t: &T) -> Result<T, OnnxError> {
        let shape = vec![t.shape[1], t.shape[0]];
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
        self.arity(n, 1)?;
        let v = self.get(&n.inputs[0])?;
        self.define(&n.outputs[0], v);
        Ok(())
    }

    /// Inference-mode Dropout is the identity. `ratio` is ignored at
    /// inference; a `training_mode` input or a `mask` output is an error.
    fn dropout(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        attrs.int("seed")?;
        if n.inputs.is_empty() || n.inputs.len() > 3 || n.inputs[0].is_empty() {
            return Err(shape_err(n, "Dropout takes 1 to 3 inputs"));
        }
        if n.inputs.get(2).is_some_and(|s| !s.is_empty()) {
            return Err(shape_err(
                n,
                "Dropout training_mode input is not supported (inference only)",
            ));
        }
        if n.outputs.len() > 2 || (n.outputs.len() == 2 && !n.outputs[1].is_empty()) {
            return Err(shape_err(n, "Dropout mask output is not supported"));
        }
        let v = self.get(&n.inputs[0])?;
        self.define(&n.outputs[0], v);
        Ok(())
    }

    fn constant(&mut self, n: &Node, attrs: &mut Attrs) -> Result<(), OnnxError> {
        if !n.inputs.is_empty() || n.outputs.len() != 1 {
            return Err(shape_err(
                n,
                "Constant takes no inputs and produces 1 output",
            ));
        }
        let out = &n.outputs[0];
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
