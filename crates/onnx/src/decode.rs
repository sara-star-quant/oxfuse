//! Bytes to `ModelProto`, under the size and nesting caps.

use crate::{OnnxError, proto::ModelProto};
use oxfuse_core::Limits;
use prost::Message;

/// Decode a serialized `ModelProto`. The size cap is checked before any
/// decoding; nesting is capped by prost's recursion limit (100 messages).
pub fn decode_model(bytes: &[u8], limits: &Limits) -> Result<ModelProto, OnnxError> {
    if bytes.len() > limits.max_file_bytes {
        return Err(OnnxError::TooLarge {
            found: bytes.len(),
            limit: limits.max_file_bytes,
        });
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
