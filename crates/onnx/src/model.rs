//! Model-level checks: opset, names, node order, input shapes. Produces a
//! `Model` the lowering can trust: every name defined once, every node input
//! defined before use, every dimension a positive number.

use crate::{
    ImportOptions, OnnxError,
    proto::{AttributeProto, ModelProto, NodeProto, ValueInfoProto, data_type, dimension::Value},
    tensor::{Dense, dtype_name, read_tensor},
};
use std::collections::{HashMap, HashSet};

/// Opsets this importer is tested against (default domain, inclusive).
pub const OPSET_RANGE: std::ops::RangeInclusive<i64> = 13..=21;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// `name`, or `<op> node <index>` when the model leaves it empty.
    pub label: String,
    pub op: String,
    /// Empty string marks an absent optional input.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<AttributeProto>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    /// Runtime inputs only; initializers are not listed here.
    pub inputs: Vec<(String, Vec<usize>)>,
    pub initializers: Vec<(String, Dense)>,
    pub nodes: Vec<Node>,
    pub outputs: Vec<String>,
}

fn structure(reason: impl Into<String>) -> OnnxError {
    OnnxError::Structure {
        reason: reason.into(),
    }
}

pub fn check_model(m: ModelProto, opts: &ImportOptions) -> Result<Model, OnnxError> {
    check_opset(&m)?;
    let g = m.graph.ok_or_else(|| structure("model has no graph"))?;
    if !g.sparse_initializer.is_empty() {
        return Err(structure("sparse initializers are not supported"));
    }

    let mut names = Names::default();
    let mut inputs: Vec<(String, Vec<usize>)> = Vec::new();
    let mut input_slots: HashMap<String, usize> = HashMap::new();
    let initializer_names: HashSet<&str> = g.initializer.iter().map(|t| t.name.as_str()).collect();
    for (i, vi) in g.input.iter().enumerate() {
        let name = names.define(&vi.name, || format!("input {i}"))?;
        // An initializer also listed as an input is a constant (old IR); the
        // initializer decides its type and shape, so the declaration is not read.
        let shape = if initializer_names.contains(name.as_str()) {
            Vec::new()
        } else {
            read_input_shape(vi, &name, opts)?
        };
        input_slots.insert(name.clone(), inputs.len());
        inputs.push((name, shape));
    }

    let mut initializers: Vec<(String, Dense)> = Vec::new();
    let mut constant_inputs: HashSet<usize> = HashSet::new();
    let mut seen_init: HashSet<String> = HashSet::new();
    for (i, t) in g.initializer.iter().enumerate() {
        let tensor = read_tensor(t, &opts.limits)?;
        if !seen_init.insert(t.name.clone()) {
            return Err(OnnxError::Name {
                reason: format!("`{}` is defined by two initializers", t.name),
            });
        }
        // An initializer that is also a graph input is a constant (old IR).
        let name = match input_slots.get(&t.name) {
            Some(slot) => {
                constant_inputs.insert(*slot);
                t.name.clone()
            }
            None => names.define(&t.name, || format!("initializer {i}"))?,
        };
        initializers.push((name, tensor));
    }
    let inputs: Vec<_> = inputs
        .into_iter()
        .enumerate()
        .filter(|(slot, _)| !constant_inputs.contains(slot))
        .map(|(_, v)| v)
        .collect();

    let produced_later: HashSet<&str> = g
        .node
        .iter()
        .flat_map(|n| n.output.iter().map(String::as_str))
        .collect();
    let mut nodes = Vec::with_capacity(g.node.len());
    for (index, n) in g.node.iter().enumerate() {
        let label = node_label(n, index);
        if !n.domain.is_empty() && n.domain != "ai.onnx" {
            return Err(OnnxError::Domain {
                node: label,
                domain: n.domain.clone(),
            });
        }
        if n.output.is_empty() {
            return Err(structure(format!("node `{label}` has no outputs")));
        }
        for input in n.input.iter().filter(|s| !s.is_empty()) {
            if !names.contains(input) {
                return Err(structure(if produced_later.contains(input.as_str()) {
                    format!(
                        "node `{label}` uses `{input}` before it is defined; nodes must be in topological order"
                    )
                } else {
                    format!("node `{label}` references unknown value `{input}`")
                }));
            }
        }
        let mut outputs = Vec::with_capacity(n.output.len());
        for (k, out) in n.output.iter().enumerate() {
            // An empty name after the first output is an optional output the
            // model leaves out (Dropout's mask); the first gets a generated name.
            if k > 0 && out.is_empty() {
                outputs.push(String::new());
            } else {
                outputs.push(names.define(out, || format!("node {index} output {k}"))?);
            }
        }
        nodes.push(Node {
            label,
            op: n.op_type.clone(),
            inputs: n.input.clone(),
            outputs,
            attrs: n.attribute.clone(),
        });
    }

    if g.output.is_empty() {
        return Err(structure("graph declares no outputs"));
    }
    let mut outputs = Vec::with_capacity(g.output.len());
    for vi in &g.output {
        if vi.name.is_empty() || !names.contains(&vi.name) {
            return Err(structure(format!(
                "graph output `{}` is not produced by any input, initializer or node",
                vi.name
            )));
        }
        if let Some(tt) = vi.r#type.as_ref().and_then(|t| t.tensor_type.as_ref())
            && tt.elem_type != data_type::FLOAT
        {
            return Err(OnnxError::Dtype {
                what: vi.name.clone(),
                dtype: dtype_name(tt.elem_type),
            });
        }
        outputs.push(vi.name.clone());
    }

    Ok(Model {
        inputs,
        initializers,
        nodes,
        outputs,
    })
}

fn node_label(n: &NodeProto, index: usize) -> String {
    if n.name.is_empty() {
        format!("{} node {index}", n.op_type)
    } else {
        n.name.clone()
    }
}

fn check_opset(m: &ModelProto) -> Result<(), OnnxError> {
    let mut found: Option<i64> = None;
    for o in &m.opset_import {
        if !(o.domain.is_empty() || o.domain == "ai.onnx") {
            continue;
        }
        let v = o.version.ok_or(OnnxError::Opset {
            found: "missing".into(),
        })?;
        if found.is_some_and(|f| f != v) {
            return Err(structure("model imports two different default opsets"));
        }
        found = Some(v);
    }
    match found {
        None => Err(OnnxError::Opset {
            found: "missing".into(),
        }),
        Some(v) if OPSET_RANGE.contains(&v) => Ok(()),
        Some(v) => Err(OnnxError::Opset {
            found: v.to_string(),
        }),
    }
}

/// Value names in definition order. An empty name becomes `v<ordinal>`; a
/// name defined twice is an error that names both places.
#[derive(Default)]
struct Names {
    defined: HashMap<String, String>,
    ordinal: usize,
}

impl Names {
    fn contains(&self, name: &str) -> bool {
        self.defined.contains_key(name)
    }

    fn define(&mut self, name: &str, place: impl FnOnce() -> String) -> Result<String, OnnxError> {
        let name = if name.is_empty() {
            format!("v{}", self.ordinal)
        } else {
            name.to_owned()
        };
        self.ordinal += 1;
        let place = place();
        if let Some(first) = self.defined.get(&name) {
            return Err(OnnxError::Name {
                reason: format!("`{name}` is defined by {first} and by {place}"),
            });
        }
        self.defined.insert(name.clone(), place);
        Ok(name)
    }
}

fn read_input_shape(
    vi: &ValueInfoProto,
    name: &str,
    opts: &ImportOptions,
) -> Result<Vec<usize>, OnnxError> {
    let tt = vi
        .r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .ok_or_else(|| structure(format!("input `{name}` has no tensor type")))?;
    if tt.elem_type != data_type::FLOAT {
        return Err(OnnxError::Dtype {
            what: name.to_owned(),
            dtype: dtype_name(tt.elem_type),
        });
    }
    let shape = tt
        .shape
        .as_ref()
        .ok_or_else(|| structure(format!("input `{name}` has no shape")))?;
    let mut dims = Vec::with_capacity(shape.dim.len());
    for (axis, d) in shape.dim.iter().enumerate() {
        let value = match &d.value {
            Some(Value::DimValue(v)) => *v,
            Some(Value::DimParam(p)) if !p.is_empty() => {
                let bound = opts
                    .dims
                    .get(p)
                    .ok_or_else(|| OnnxError::DimUnbound { name: p.clone() })?;
                i64::try_from(*bound).map_err(|_| structure(format!("--dim {p} is too large")))?
            }
            _ => {
                return Err(structure(format!(
                    "input `{name}` dimension {axis} is unknown; give it a name in the model and pass --dim"
                )));
            }
        };
        if value < 1 {
            return Err(structure(format!(
                "input `{name}` dimension {axis} is {value}; dimensions must be at least 1"
            )));
        }
        dims.push(usize::try_from(value).map_err(|_| structure("dimension too large"))?);
    }
    Ok(dims)
}
