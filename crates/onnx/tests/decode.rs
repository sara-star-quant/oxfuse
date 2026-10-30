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
    let limits = Limits::default().with_max_file_bytes(4);
    // Garbage that would also fail decoding: the size error must win.
    let r = decode_model(&[0xff; 5], &limits);
    assert_eq!(r, Err(OnnxError::TooLarge { found: 5, limit: 4 }));
}

#[test]
fn file_at_the_limit_is_decoded() {
    let limits = Limits::default().with_max_file_bytes(4);
    assert!(decode_model(&[], &limits).is_ok());
    // Exactly 4 bytes: tag, then a three-byte varint.
    let at_limit = [0x08, 0x80, 0x80, 0x01];
    assert!(decode_model(&at_limit, &limits).is_ok());
    assert_eq!(
        decode_model(&[0x08, 0x80, 0x80, 0x80, 0x01], &limits),
        Err(OnnxError::TooLarge { found: 5, limit: 4 })
    );
}

// A model of many tiny messages costs far more memory than its bytes, so the
// message count is capped before prost allocates anything.

fn small_budget() -> Limits {
    // Budget: 10 messages per value, so 100 messages in all.
    Limits::default().with_max_values(10)
}

fn model_with_nodes(n: usize) -> Vec<u8> {
    let graph = ld(1, &[]).repeat(n);
    ld(7, &graph)
}

#[test]
fn the_graph_and_its_nodes_count_and_the_budget_is_inclusive() {
    // The graph message is one, each node another: 99 nodes make 100.
    assert!(decode_model(&model_with_nodes(99), &small_budget()).is_ok());
    assert_eq!(
        decode_model(&model_with_nodes(100), &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
}

#[test]
fn the_budget_scales_with_max_values() {
    let wide = Limits::default().with_max_values(11);
    assert!(decode_model(&model_with_nodes(100), &wide).is_ok());
    assert_eq!(
        decode_model(&model_with_nodes(110), &wide),
        Err(OnnxError::TooManyMessages { limit: 110 })
    );
}

#[test]
fn a_huge_budget_does_not_overflow() {
    let huge = Limits::default().with_max_values(usize::MAX);
    assert!(decode_model(&model_with_nodes(5), &huge).is_ok());
}

#[test]
fn messages_nested_below_a_node_count_too() {
    // One node with 120 empty attributes (field 5 of NodeProto).
    let node = ld(5, &[]).repeat(120);
    let model = ld(7, &ld(1, &node));
    assert_eq!(
        decode_model(&model, &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
}

#[test]
fn every_repeated_message_field_counts() {
    // initializer 5, input 11, output 12, sparse_initializer 15 of GraphProto,
    // opset_import 8 of ModelProto.
    for field in [5, 11, 12, 15] {
        let model = ld(7, &ld(field, &[]).repeat(150));
        assert_eq!(
            decode_model(&model, &small_budget()),
            Err(OnnxError::TooManyMessages { limit: 100 }),
            "graph field {field}"
        );
    }
    let model = ld(8, &[]).repeat(150);
    assert_eq!(
        decode_model(&model, &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
}

#[test]
fn the_type_chain_of_a_value_info_counts() {
    // ValueInfo(11) > type(2) > tensor_type(1) > shape(2) > 150 dims(1).
    let shape = ld(1, &[]).repeat(150);
    let info = ld(2, &ld(1, &ld(2, &shape)));
    let model = ld(7, &ld(11, &info));
    assert_eq!(
        decode_model(&model, &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
}

#[test]
fn bytes_strings_and_unknown_fields_are_not_messages() {
    // A node name, a long raw_data, and an unknown field full of what would
    // look like messages if it were scanned.
    let tensor = ld(9, &vec![0u8; 5000]);
    let unknown = ld(99, &ld(1, &[]).repeat(500));
    let node = cat(&[ld(3, b"name"), unknown]);
    let model = ld(7, &cat(&[ld(1, &node), ld(5, &tensor)]));
    assert!(decode_model(&model, &small_budget()).is_ok());
}

#[test]
fn a_million_tiny_nodes_are_refused_at_the_default_limits() {
    let model = model_with_nodes(1_000_001);
    assert_eq!(
        decode_model(&model, &Limits::default()),
        Err(OnnxError::TooManyMessages { limit: 1_000_000 })
    );
}

#[test]
fn malformed_bytes_are_still_a_decode_error_not_a_count_error() {
    let mut model = model_with_nodes(3);
    model.truncate(model.len() - 1);
    is_decode_error(decode(&model));
    // A length that runs past the end, inside the graph.
    let bad = ld(7, &[0x0a, 0x7f]);
    is_decode_error(decode(&bad));
}

#[test]
fn the_size_cap_is_still_checked_first() {
    let limits = Limits::default().with_max_file_bytes(4);
    assert!(matches!(
        decode_model(&model_with_nodes(10), &limits),
        Err(OnnxError::TooLarge { .. })
    ));
}

fn chain_with_flood(levels: usize, flood: usize) -> Vec<u8> {
    // graph > node > attribute > graph ... `levels` graphs deep, `flood`
    // empty nodes in the innermost graph.
    let mut graph = ld(1, &[]).repeat(flood);
    for _ in 1..levels {
        graph = ld(1, &ld(5, &ld(6, &graph)));
    }
    ld(7, &graph)
}

#[test]
fn a_flood_at_the_deepest_level_prost_accepts_is_counted_and_deeper_is_too_deep() {
    // Each level is graph > node > attribute, so 34 levels put the flood at
    // depth 100, where prost still decodes; one more level is past its limit.
    let limits = Limits::default().with_max_values(1000);
    assert_eq!(
        decode_model(&chain_with_flood(34, 10_001), &limits),
        Err(OnnxError::TooManyMessages { limit: 10_000 })
    );
    assert_eq!(
        decode_model(&chain_with_flood(35, 10_001), &limits),
        Err(OnnxError::TooDeep)
    );
}

#[test]
fn messages_inside_attributes_count() {
    // AttributeProto: tensor (5) and graph (6).
    let tensors = ld(5, &[]).repeat(150);
    let model = ld(7, &ld(1, &ld(5, &tensors)));
    assert_eq!(
        decode_model(&model, &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
    let nodes = ld(1, &[]).repeat(150);
    let model = ld(7, &ld(1, &ld(5, &ld(6, &nodes))));
    assert_eq!(
        decode_model(&model, &small_budget()),
        Err(OnnxError::TooManyMessages { limit: 100 })
    );
}

#[test]
fn scalar_fields_before_a_flood_do_not_hide_it() {
    // A varint, a fixed64 and a fixed32 field in front of 150 nodes, then the
    // same fields between them.
    let fixed64 = [varint(40 << 3 | 1), vec![0; 8]].concat();
    let fixed32 = [varint(41 << 3 | 5), vec![0; 4]].concat();
    let scalars = cat(&[vf(42, 300), fixed64, fixed32]);
    for body in [
        cat(&[scalars.clone(), ld(1, &[]).repeat(150)]),
        cat(&[
            scalars.clone(),
            ld(1, &[]).repeat(75),
            scalars,
            ld(1, &[]).repeat(75),
        ]),
    ] {
        assert_eq!(
            decode_model(&ld(7, &body), &small_budget()),
            Err(OnnxError::TooManyMessages { limit: 100 })
        );
    }
}
