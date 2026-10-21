//! Wire-level behavior of `decode_model`, byte by byte.

use oxfuse_core::Limits;
use oxfuse_onnx::{OnnxError, decode_model};

fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
}

/// A length-delimited field.
fn ld(field: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = varint(field << 3 | 2);
    out.extend(varint(payload.len() as u64));
    out.extend(payload);
    out
}

/// A varint field.
fn vf(field: u64, value: u64) -> Vec<u8> {
    let mut out = varint(field << 3);
    out.extend(varint(value));
    out
}

fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.concat()
}

fn decode(bytes: &[u8]) -> Result<oxfuse_onnx::proto::ModelProto, OnnxError> {
    decode_model(bytes, &Limits::default())
}

fn is_decode_error(r: Result<oxfuse_onnx::proto::ModelProto, OnnxError>) {
    match r {
        Err(OnnxError::Decode { .. }) => {}
        other => panic!("expected a decode error, got {other:?}"),
    }
}

#[test]
fn empty_input_decodes_to_an_empty_model() {
    let m = decode(&[]).unwrap();
    assert!(m.graph.is_none());
    assert!(m.opset_import.is_empty());
}

#[test]
fn one_byte() {
    is_decode_error(decode(&[0x08]));
}

#[test]
fn truncated_mid_field() {
    let full = cat(&[ld(8, &vf(2, 13))]);
    is_decode_error(decode(&full[..full.len() - 1]));
}

#[test]
fn truncated_mid_varint() {
    is_decode_error(decode(&[0x08, 0x80]));
}

#[test]
fn varint_longer_than_ten_bytes() {
    let mut b = vec![0x08];
    b.extend([0x80; 10]);
    b.push(0x01);
    is_decode_error(decode(&b));
}

#[test]
fn length_prefix_exceeds_remaining() {
    is_decode_error(decode(&[0x3a, 0x05, 1, 2]));
}

#[test]
fn length_prefix_2_pow_32() {
    let mut b = vec![0x3a];
    b.extend(varint(1 << 32));
    b.push(1);
    is_decode_error(decode(&b));
}

#[test]
fn length_prefix_2_pow_63() {
    let mut b = vec![0x3a];
    b.extend(varint(1 << 63));
    b.push(1);
    is_decode_error(decode(&b));
}

#[test]
fn wrong_wire_type_for_known_field() {
    // Field 7 (graph) must be length-delimited; send a varint.
    is_decode_error(decode(&vf(7, 1)));
}

#[test]
fn unknown_fields_skipped_at_every_level() {
    let unknown = || cat(&[vf(999, 5), ld(998, b"zz")]);
    let tensor = cat(&[vf(1, 2), vf(2, 1), unknown()]);
    let node = cat(&[ld(4, b"Relu"), unknown()]);
    let graph = cat(&[ld(1, &node), ld(5, &tensor), unknown()]);
    let opset = cat(&[vf(2, 13), unknown()]);
    let model = cat(&[ld(7, &graph), ld(8, &opset), unknown()]);
    let m = decode(&model).unwrap();
    let g = m.graph.unwrap();
    assert_eq!(g.node[0].op_type, "Relu");
    assert_eq!(g.initializer[0].dims, vec![2]);
    assert_eq!(m.opset_import[0].version, Some(13));
}

#[test]
fn repeated_singular_field_last_wins() {
    let opset = cat(&[vf(2, 13), vf(2, 14)]);
    let m = decode(&ld(8, &opset)).unwrap();
    assert_eq!(m.opset_import[0].version, Some(14));
}

#[test]
fn deep_nesting_hits_cap_without_overflow() {
    // graph -> node -> attribute -> graph ..., far deeper than any model.
    let mut graph = Vec::new();
    for _ in 0..2000 {
        let attr = ld(6, &graph);
        let node = ld(5, &attr);
        graph = ld(1, &node);
    }
    let model = ld(7, &graph);
    assert_eq!(decode(&model), Err(OnnxError::TooDeep));
}

#[test]
fn oversize_file_rejected_before_decoding() {
    let limits = Limits {
        max_file_bytes: 4,
        ..Limits::default()
    };
    // Garbage that would also fail decoding: the size error must win.
    let r = decode_model(&[0xff; 5], &limits);
    assert_eq!(r, Err(OnnxError::TooLarge { found: 5, limit: 4 }));
}

#[test]
fn file_at_the_limit_is_decoded() {
    let limits = Limits {
        max_file_bytes: 4,
        ..Limits::default()
    };
    assert!(decode_model(&[], &limits).is_ok());
    // Exactly 4 bytes: tag, then a three-byte varint.
    let at_limit = [0x08, 0x80, 0x80, 0x01];
    assert!(decode_model(&at_limit, &limits).is_ok());
    assert_eq!(
        decode_model(&[0x08, 0x80, 0x80, 0x80, 0x01], &limits),
        Err(OnnxError::TooLarge { found: 5, limit: 4 })
    );
}
