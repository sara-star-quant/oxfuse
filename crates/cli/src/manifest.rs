//! The record `oxfuse run --manifest` writes: what ran, on what, and the
//! hashes of what went in and came out. Paths are not recorded.

use crate::RunOptions;
use oxfuse_runtime::Tensor;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const MANIFEST_VERSION: u32 = 1;

pub(crate) struct OutputDigest {
    pub name: String,
    pub shape: Vec<usize>,
    pub sha256: String,
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Over the little-endian bit patterns, so `-0.0` and `0.0` differ.
pub(crate) fn digest(name: &str, tensor: &Tensor) -> OutputDigest {
    let bytes: Vec<u8> = tensor
        .data()
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    OutputDigest {
        name: name.to_owned(),
        shape: tensor.shape().to_vec(),
        sha256: hex(&bytes),
    }
}

pub(crate) fn build(
    options: &RunOptions,
    backend: &str,
    model: &[u8],
    inputs: Option<&[u8]>,
    outputs: &[OutputDigest],
) -> Value {
    let unix_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    json!({
        "manifest_version": MANIFEST_VERSION,
        "oxfuse_version": env!("CARGO_PKG_VERSION"),
        "backend": backend,
        "optimize": options.optimize,
        "disabled_passes": options.disabled,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "simd": oxfuse_runtime::simd_level(),
        "features": {
            "parallel": cfg!(feature = "parallel"),
            "wgpu": cfg!(feature = "wgpu"),
            "onnx": cfg!(feature = "onnx"),
        },
        "model_sha256": hex(model),
        "inputs_sha256": inputs.map(hex),
        "outputs": outputs
            .iter()
            .map(|o| json!({ "name": o.name, "shape": o.shape, "sha256": o.sha256 }))
            .collect::<Vec<_>>(),
        "unix_time": unix_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_of_nothing_is_the_known_empty_hash() {
        assert_eq!(
            hex(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn a_tensor_digest_covers_bits_not_values() {
        let zero = Tensor::new(vec![1], vec![0.0]).unwrap();
        let negative = Tensor::new(vec![1], vec![-0.0]).unwrap();
        assert_ne!(digest("y", &zero).sha256, digest("y", &negative).sha256);
        assert_eq!(digest("y", &zero).sha256, hex(&0u32.to_le_bytes()));
    }
}
