//! Builders for hand-made models. Not every test file uses every helper.
#![allow(dead_code)]

use oxfuse_onnx::proto::*;
use prost::Message;

/// One dimension of an input shape.
pub enum D {
    V(i64),
    P(&'static str),
    Unknown,
}

pub fn value_info(name: &str, dims: &[D]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TypeTensor {
                elem_type: data_type::FLOAT,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| Dimension {
                            value: match d {
                                D::V(v) => Some(dimension::Value::DimValue(*v)),
                                D::P(p) => Some(dimension::Value::DimParam((*p).into())),
                                D::Unknown => None,
                            },
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

pub fn input(name: &str, shape: &[i64]) -> ValueInfoProto {
    let dims: Vec<D> = shape.iter().map(|d| D::V(*d)).collect();
    value_info(name, &dims)
}

pub fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: outputs.iter().map(|s| (*s).into()).collect(),
        ..NodeProto::default()
    }
}

pub fn with_int(mut n: NodeProto, name: &str, v: i64) -> NodeProto {
    n.attribute.push(AttributeProto {
        name: name.into(),
        i: Some(v),
        r#type: attribute_type::INT,
        ..AttributeProto::default()
    });
    n
}

pub fn with_float(mut n: NodeProto, name: &str, v: f32) -> NodeProto {
    n.attribute.push(AttributeProto {
        name: name.into(),
        f: Some(v),
        r#type: attribute_type::FLOAT,
        ..AttributeProto::default()
    });
    n
}

pub fn with_ints(mut n: NodeProto, name: &str, v: &[i64]) -> NodeProto {
    n.attribute.push(AttributeProto {
        name: name.into(),
        ints: v.to_vec(),
        r#type: attribute_type::INTS,
        ..AttributeProto::default()
    });
    n
}

pub fn init(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: data_type::FLOAT,
        float_data: data.to_vec(),
        ..TensorProto::default()
    }
}

pub fn init_i64(name: &str, data: &[i64]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: vec![data.len() as i64],
        data_type: data_type::INT64,
        int64_data: data.to_vec(),
        ..TensorProto::default()
    }
}

pub fn model(opset: i64, graph: GraphProto) -> ModelProto {
    ModelProto {
        ir_version: Some(9),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: Some(opset),
        }],
    }
}

pub fn graph(
    inputs: Vec<ValueInfoProto>,
    initializers: Vec<TensorProto>,
    nodes: Vec<NodeProto>,
    outputs: &[&str],
) -> GraphProto {
    GraphProto {
        node: nodes,
        initializer: initializers,
        input: inputs,
        output: outputs
            .iter()
            .map(|n| ValueInfoProto {
                name: (*n).into(),
                r#type: None,
            })
            .collect(),
        sparse_initializer: vec![],
    }
}

pub fn bytes(m: &ModelProto) -> Vec<u8> {
    m.encode_to_vec()
}

use oxfuse_core::{Graph, Source};
use oxfuse_onnx::{ImportOptions, OnnxError, import};
use oxfuse_runtime::{Backend, ReferenceBackend, Tensor};

pub fn load(m: &ModelProto) -> Result<Graph, OnnxError> {
    import(&bytes(m), &ImportOptions::default())
}

pub fn load_ok(m: &ModelProto) -> Graph {
    load(m).unwrap_or_else(|e| panic!("import failed: {e}"))
}

/// Op names of the IR nodes, in order.
pub fn ops(g: &Graph) -> Vec<&'static str> {
    g.values()
        .iter()
        .filter_map(|v| match &v.source {
            Source::Node { op, .. } => Some(op.name()),
            _ => None,
        })
        .collect()
}

/// Run on the reference backend; inputs are `(shape, data)` in input order.
pub fn run(g: &Graph, inputs: &[(&[usize], &[f32])]) -> Vec<Vec<f32>> {
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|(s, d)| Tensor::new(s.to_vec(), d.to_vec()).unwrap())
        .collect();
    ReferenceBackend
        .compile(g)
        .unwrap()
        .run(&tensors)
        .unwrap()
        .iter()
        .map(|t| t.data().to_vec())
        .collect()
}

pub fn shape_of_output(g: &Graph, i: usize) -> Vec<usize> {
    g.value(g.outputs()[i]).ty.shape.clone()
}
