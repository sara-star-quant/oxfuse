//! The on-disk JSON formats. These types never leave the CLI crate's loader:
//! the core IR knows nothing about files.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphFile {
    pub format_version: u32,
    #[serde(default)]
    pub inputs: Vec<InputDecl>,
    #[serde(default)]
    pub constants: Vec<ConstantDecl>,
    pub nodes: Vec<NodeDecl>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum FileDType {
    #[serde(rename = "f32")]
    F32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDecl {
    pub name: String,
    #[expect(dead_code, reason = "f32 is the only dtype; parsing rejects others")]
    pub dtype: FileDType,
    pub shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstantDecl {
    pub name: String,
    #[expect(dead_code, reason = "f32 is the only dtype; parsing rejects others")]
    pub dtype: FileDType,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDecl {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub attrs: Attrs,
}

/// Op attributes. Which keys an op accepts is decided by the loader, not here.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attrs {
    /// Signed so a negative entry reaches the loader, which reports it by name
    /// instead of as an opaque parse error.
    pub shape: Option<Vec<i64>>,
    pub perm: Option<Vec<i64>>,
}

/// One tensor in an inputs file (keyed by input name).
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorFile {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// One tensor in the output of `run`.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedTensorFile {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputsFile {
    pub outputs: Vec<NamedTensorFile>,
}

/// The inputs file: input name to tensor. serde's own map deserialization
/// keeps the last of two equal keys; two entries for one input is a mistake
/// worth reporting, not a value to pick silently.
#[derive(Debug)]
pub struct InputsFile(pub BTreeMap<String, TensorFile>);

impl<'de> Deserialize<'de> for InputsFile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputsVisitor;

        impl<'de> Visitor<'de> for InputsVisitor {
            type Value = InputsFile;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "an object mapping input names to {\"shape\": [...], \"values\": [...]}",
                )
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<InputsFile, A::Error> {
                let mut inputs = BTreeMap::new();
                while let Some((name, tensor)) = map.next_entry::<String, TensorFile>()? {
                    if inputs.contains_key(&name) {
                        return Err(de::Error::custom(format!("duplicate input `{name}`")));
                    }
                    inputs.insert(name, tensor);
                }
                Ok(InputsFile(inputs))
            }
        }

        deserializer.deserialize_map(InputsVisitor)
    }
}
