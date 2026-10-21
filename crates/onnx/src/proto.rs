//! The part of `onnx.proto` this importer reads, as `prost` structs. Fields
//! not listed here are skipped by the protobuf wire rules.
//!
//! Source: ONNX `onnx/onnx.proto` (proto2 schema), <https://github.com/onnx/onnx>,
//! Apache-2.0. Field numbers and names follow that file; the structs are
//! hand-written, no `protoc` runs at build time. See `NOTICE`.

#[derive(Clone, PartialEq, prost::Message)]
pub struct ModelProto {
    #[prost(int64, optional, tag = "1")]
    pub ir_version: Option<i64>,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, optional, tag = "2")]
    pub version: Option<i64>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    /// Only its presence matters: sparse initializers are rejected.
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct SparseTensorProto {}

#[derive(Clone, PartialEq, prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// `AttributeProto.AttributeType` values this importer distinguishes.
pub mod attribute_type {
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const STRING: i32 = 3;
    pub const TENSOR: i32 = 4;
    pub const FLOATS: i32 = 6;
    pub const INTS: i32 = 7;
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, optional, tag = "2")]
    pub f: Option<f32>,
    #[prost(int64, optional, tag = "3")]
    pub i: Option<i64>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub s: Option<Vec<u8>>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, boxed, tag = "6")]
    pub g: Option<Box<GraphProto>>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
}

/// `TensorProto.DataType` values this importer accepts.
pub mod data_type {
    pub const FLOAT: i32 = 1;
    pub const INT64: i32 = 7;
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(message, optional, tag = "3")]
    pub segment: Option<Segment>,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    pub data_location: i32,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Segment {
    #[prost(int64, tag = "1")]
    pub begin: i64,
    #[prost(int64, tag = "2")]
    pub end: i64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

/// Only the dense tensor case (`oneof value`, field 1) is read; other cases
/// are skipped and show up as a missing tensor type.
#[derive(Clone, PartialEq, prost::Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeTensor>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TypeTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Dimension {
    #[prost(oneof = "dimension::Value", tags = "1, 2")]
    pub value: Option<dimension::Value>,
}

pub mod dimension {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        #[prost(int64, tag = "1")]
        DimValue(i64),
        #[prost(string, tag = "2")]
        DimParam(String),
    }
}
