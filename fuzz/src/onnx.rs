//! ONNX importer inputs: raw bytes straight into `import`, and models built
//! from `prost` structs so most inputs get past the decoder.

use arbitrary::{Result as ArbResult, Unstructured};
use oxfuse_core::Graph;
use oxfuse_onnx::{ImportOptions, OnnxError, import, proto::*};
use prost::Message;

fn options() -> ImportOptions {
    ImportOptions {
        limits: crate::small(),
        dims: [("N".to_owned(), 3)].into(),
    }
}

pub fn import_small(bytes: &[u8]) -> Result<Graph, OnnxError> {
    import(bytes, &options())
}

const VALUES: [f32; 8] = [0.0, -0.0, 1.0, -1.5, 1e30, 1e-30, 3.25, -7.0];

fn info(name: &str, dims: &[usize]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TypeTensor {
                elem_type: data_type::FLOAT,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| Dimension {
                            value: Some(dimension::Value::DimValue(*d as i64)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn node(op: &str, inputs: &[&str], out: &str) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![out.into()],
        ..NodeProto::default()
    }
}

fn int_attr(n: &mut NodeProto, name: &str, v: i64) {
    n.attribute.push(AttributeProto {
        name: name.into(),
        i: Some(v),
        r#type: attribute_type::INT,
        ..AttributeProto::default()
    });
}

fn float_attr(n: &mut NodeProto, name: &str, v: f32) {
    n.attribute.push(AttributeProto {
        name: name.into(),
        f: Some(v),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    });
}

/// Widest result a `MatMul` or `Gemm` may produce for `rows` rows: keeps every
/// value at 512 elements or fewer, inside the fuzz limits.
fn cap_cols(rows: usize) -> usize {
    (512 / rows).clamp(1, 8)
}

/// A float32 initializer of `shape` filled from `VALUES`; returns its name.
fn constant(
    inits: &mut Vec<TensorProto>,
    shape: &[usize],
    u: &mut Unstructured,
) -> ArbResult<String> {
    let count: usize = shape.iter().product();
    let mut data = Vec::new();
    for _ in 0..count {
        data.push(*u.choose(&VALUES)?);
    }
    let name = format!("k{}", inits.len());
    inits.push(TensorProto {
        name: name.clone(),
        dims: shape.iter().map(|d| *d as i64).collect(),
        data_type: data_type::FLOAT,
        float_data: data,
        ..TensorProto::default()
    });
    Ok(name)
}

/// A valid model: one rank-2 input and a chain of supported ops over it.
/// Valid by construction, so `import` must accept every one.
pub fn generate(u: &mut Unstructured) -> ArbResult<Vec<u8>> {
    let mut shape = vec![u.int_in_range(1..=8usize)?, u.int_in_range(1..=8usize)?];
    let input_shape = shape.clone();
    let mut nodes = Vec::new();
    let mut inits: Vec<TensorProto> = Vec::new();
    let mut last = "x".to_owned();
    for i in 0..u.int_in_range(1..=6usize)? {
        let out = format!("t{i}");
        let rank2 = shape.len() == 2;
        match u.int_in_range(0..=9u8)? {
            0 => nodes.push(node("Relu", &[&last], &out)),
            1 => nodes.push(node("Sigmoid", &[&last], &out)),
            2 => nodes.push(node("Identity", &[&last], &out)),
            3 => nodes.push(node("Dropout", &[&last], &out)),
            4 => {
                let op = if u.arbitrary()? { "Add" } else { "Mul" };
                let k = match u.int_in_range(0..=2u8)? {
                    0 => constant(&mut inits, &[], u)?,
                    1 => constant(&mut inits, &[*shape.last().unwrap()], u)?,
                    _ => constant(&mut inits, &shape, u)?,
                };
                if u.arbitrary()? {
                    nodes.push(node(op, &[&last, &k], &out));
                } else {
                    nodes.push(node(op, &[&k, &last], &out));
                }
            }
            5 if rank2 => {
                let n = u.int_in_range(1..=cap_cols(shape[0]))?;
                let w = constant(&mut inits, &[shape[1], n], u)?;
                nodes.push(node("MatMul", &[&last, &w], &out));
                shape[1] = n;
            }
            6 if rank2 => {
                let (m, k) = (shape[0], shape[1]);
                let n = u.int_in_range(1..=cap_cols(m))?;
                let trans_b = u.arbitrary::<bool>()?;
                let wshape = if trans_b { [n, k] } else { [k, n] };
                let w = constant(&mut inits, &wshape, u)?;
                let mut args = vec![last.clone(), w];
                if u.arbitrary()? {
                    let c = match u.int_in_range(0..=3u8)? {
                        0 => constant(&mut inits, &[], u)?,
                        1 => constant(&mut inits, &[n], u)?,
                        2 => constant(&mut inits, &[m, 1], u)?,
                        _ => constant(&mut inits, &[m, n], u)?,
                    };
                    args.push(c);
                }
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                let mut g = node("Gemm", &args, &out);
                int_attr(&mut g, "transB", trans_b as i64);
                float_attr(&mut g, "alpha", *u.choose(&[1.0, 0.5, -1.0, 0.0])?);
                float_attr(&mut g, "beta", *u.choose(&[1.0, 0.25, -2.0, 0.0])?);
                nodes.push(g);
                shape = vec![m, n];
            }
            7 => {
                nodes.push(node("Transpose", &[&last], &out));
                shape.reverse();
            }
            8 => {
                let axis = u.int_in_range(0..=shape.len())?;
                let mut f = node("Flatten", &[&last], &out);
                int_attr(&mut f, "axis", axis as i64);
                nodes.push(f);
                shape = vec![
                    shape[..axis].iter().product(),
                    shape[axis..].iter().product(),
                ];
            }
            _ => {
                let total: usize = shape.iter().product();
                let name = format!("s{i}");
                inits.push(TensorProto {
                    name: name.clone(),
                    dims: vec![1],
                    data_type: data_type::INT64,
                    int64_data: vec![-1],
                    ..TensorProto::default()
                });
                nodes.push(node("Reshape", &[&last, &name], &out));
                shape = vec![total];
            }
        }
        last = out;
    }
    let model = ModelProto {
        ir_version: Some(9),
        graph: Some(GraphProto {
            node: nodes,
            initializer: inits,
            input: vec![info("x", &input_shape)],
            output: vec![ValueInfoProto {
                name: last,
                r#type: None,
            }],
            sparse_initializer: vec![],
        }),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: Some(13),
        }],
    };
    Ok(model.encode_to_vec())
}

/// Whether a constant holds a NaN other than the canonical one. Which payload
/// survives when two different NaNs meet is hardware behavior that
/// docs/semantics.md leaves unspecified, and the backends order operands
/// differently, so a raw-byte model carrying such a NaN is imported but not
/// compared bit for bit.
pub fn has_exotic_nan(graph: &Graph) -> bool {
    graph.values().iter().any(|v| match &v.source {
        oxfuse_core::Source::Constant(data) => data
            .iter()
            .any(|x| x.is_nan() && x.to_bits() != f32::NAN.to_bits()),
        _ => false,
    })
}
