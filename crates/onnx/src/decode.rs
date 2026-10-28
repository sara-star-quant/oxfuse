//! Bytes to `ModelProto`, under the size and nesting caps.

use crate::{OnnxError, proto::ModelProto};
use oxfuse_core::Limits;
use prost::Message;

/// Messages allowed per value of `Limits::max_values`. A message of two bytes
/// costs prost about 150 bytes of heap (measured: a 16 MiB file of empty nodes
/// peaks at 1.2 GB), so the byte cap alone does not bound memory.
const MESSAGES_PER_VALUE: usize = 10;

/// The messages the importer reads, by where they sit in the model.
#[derive(Clone, Copy)]
enum Msg {
    Model,
    Graph,
    Node,
    Attribute,
    Value,
    Type,
    TensorType,
    Shape,
    /// Counted but never entered: tensors and the rest hold only scalars,
    /// packed arrays and bytes, or are skipped by the importer.
    Leaf,
}

/// The message type of `field` inside `msg`, from the field numbers in `proto.rs`.
fn child(msg: Msg, field: u64) -> Option<Msg> {
    Some(match (msg, field) {
        (Msg::Model, 7) => Msg::Graph,
        (Msg::Model, 8) => Msg::Leaf,
        (Msg::Graph, 1) => Msg::Node,
        (Msg::Graph, 5 | 15) => Msg::Leaf,
        (Msg::Graph, 11 | 12) => Msg::Value,
        (Msg::Node, 5) => Msg::Attribute,
        (Msg::Attribute, 5) => Msg::Leaf,
        (Msg::Attribute, 6) => Msg::Graph,
        (Msg::Value, 2) => Msg::Type,
        (Msg::Type, 1) => Msg::TensorType,
        (Msg::TensorType, 2) => Msg::Shape,
        (Msg::Shape, 1) => Msg::Leaf,
        _ => return None,
    })
}

fn read_varint(bytes: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let (&b, rest) = bytes.split_first()?;
        *bytes = rest;
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// Counts messages in `bytes` without allocating, stopping as soon as the
/// count passes `budget`. Anything it cannot read is left to prost, which
/// reports it as a decode error.
fn count_messages(mut bytes: &[u8], msg: Msg, depth: usize, count: &mut usize, budget: usize) {
    // prost refuses nesting past 100; stop here and let it say so.
    if depth > 100 {
        return;
    }
    while *count <= budget {
        let Some(tag) = read_varint(&mut bytes) else {
            return;
        };
        let skip = match tag & 7 {
            0 => read_varint(&mut bytes).map(|_| 0),
            1 => Some(8),
            5 => Some(4),
            2 => read_varint(&mut bytes).and_then(|n| usize::try_from(n).ok()),
            _ => return,
        };
        let Some(n) = skip.filter(|&n| n <= bytes.len()) else {
            return;
        };
        let (body, rest) = bytes.split_at(n);
        bytes = rest;
        if tag & 7 == 2
            && let Some(kind) = child(msg, tag >> 3)
        {
            *count += 1;
            if !matches!(kind, Msg::Leaf) {
                count_messages(body, kind, depth + 1, count, budget);
            }
        }
    }
}

/// Decode a serialized `ModelProto`. The size cap and the message count are
/// checked before any decoding; nesting is capped by prost's recursion limit
/// (100 messages).
pub fn decode_model(bytes: &[u8], limits: &Limits) -> Result<ModelProto, OnnxError> {
    if bytes.len() > limits.max_file_bytes {
        return Err(OnnxError::TooLarge {
            found: bytes.len(),
            limit: limits.max_file_bytes,
        });
    }
    let budget = limits.max_values.saturating_mul(MESSAGES_PER_VALUE);
    let mut count = 0;
    count_messages(bytes, Msg::Model, 0, &mut count, budget);
    if count > budget {
        return Err(OnnxError::TooManyMessages { limit: budget });
    }
    ModelProto::decode(bytes).map_err(|e| {
        let reason = e.to_string();
        if reason.contains("recursion limit reached") {
            OnnxError::TooDeep
        } else {
            OnnxError::Decode { reason }
        }
    })
}
