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

/// The entries of an inputs file, keyed by input name. serde's own map deserialization
/// keeps the last of two equal keys; two entries for one input is a mistake worth
/// reporting, not a value to pick silently.
fn unique_entries<'de, A: MapAccess<'de>, T: Deserialize<'de>>(
    mut map: A,
) -> Result<BTreeMap<String, T>, A::Error> {
    let mut inputs = BTreeMap::new();
    while let Some((name, tensor)) = map.next_entry::<String, T>()? {
        if inputs.contains_key(&name) {
            return Err(de::Error::custom(format!("duplicate input `{name}`")));
        }
        inputs.insert(name, tensor);
    }
    Ok(inputs)
}

/// The inputs file: input name to tensor.
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

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<InputsFile, A::Error> {
                unique_entries(map).map(InputsFile)
            }
        }

        deserializer.deserialize_map(InputsVisitor)
    }
}

/// One tensor in an inputs file of `analyze`: a point (`values`) or a box (`lo` and `hi`).
#[derive(Debug)]
pub enum BoxTensorFile {
    Point(TensorFile),
    Box {
        shape: Vec<usize>,
        lo: Vec<f32>,
        hi: Vec<f32>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBoxTensor {
    shape: Vec<usize>,
    values: Option<Vec<f32>>,
    lo: Option<Vec<f32>>,
    hi: Option<Vec<f32>>,
}

impl<'de> Deserialize<'de> for BoxTensorFile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawBoxTensor::deserialize(deserializer)?;
        match (raw.values, raw.lo, raw.hi) {
            (Some(values), None, None) => Ok(BoxTensorFile::Point(TensorFile {
                shape: raw.shape,
                values,
            })),
            (None, Some(lo), Some(hi)) => Ok(BoxTensorFile::Box {
                shape: raw.shape,
                lo,
                hi,
            }),
            (None, Some(_), None) => Err(de::Error::custom("a box needs `hi` next to `lo`")),
            (None, None, Some(_)) => Err(de::Error::custom("a box needs `lo` next to `hi`")),
            (None, None, None) => Err(de::Error::custom(
                "a tensor needs `values`, or `lo` and `hi` for a box",
            )),
            (Some(_), _, _) => Err(de::Error::custom(
                "a tensor has either `values` or a box (`lo` and `hi`), not both",
            )),
        }
    }
}

/// The inputs file of `analyze`: each input is a point or a box.
#[derive(Debug)]
pub struct BoxInputsFile(pub BTreeMap<String, BoxTensorFile>);

impl<'de> Deserialize<'de> for BoxInputsFile {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BoxInputsVisitor;

        impl<'de> Visitor<'de> for BoxInputsVisitor {
            type Value = BoxInputsFile;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "an object mapping input names to {\"shape\": [...], \"values\": [...]} or {\"shape\": [...], \"lo\": [...], \"hi\": [...]}",
                )
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<BoxInputsFile, A::Error> {
                unique_entries(map).map(BoxInputsFile)
            }
        }

        deserializer.deserialize_map(BoxInputsVisitor)
    }
}

/// One output of `analyze`: the values `run` prints, and for each value a bound
/// such that the exact result lies within `value +- bound`. A `null` bound means
/// no certificate: the element is, or came from, a non-finite value.
#[derive(Debug, Serialize)]
pub struct CertifiedTensorFile {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub bounds: Vec<Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct CertificatesFile {
    pub outputs: Vec<CertifiedTensorFile>,
}

/// One output of `analyze` over an input box: every element's exact result and reference
/// float32 result lie in `[lo, hi]` for every input in the box. A `null` end is unbounded.
#[derive(Debug, Serialize)]
pub struct EnclosureTensorFile {
    pub name: String,
    pub shape: Vec<usize>,
    pub lo: Vec<Option<f64>>,
    pub hi: Vec<Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct EnclosuresFile {
    pub outputs: Vec<EnclosureTensorFile>,
}
