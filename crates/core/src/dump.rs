//! Deterministic text form of the IR, used by `inspect`, `--dump-ir` and the
//! snapshot tests. Same graph, same text, on every platform.

use crate::{DType, Graph, Op, Source, TensorType};
use std::fmt::{self, Display, Formatter};

/// Constants longer than this print as a count, not their data.
const MAX_SHOWN: usize = 8;

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
        }
    }
}

impl Display for TensorType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        write!(f, "{}[{}]", self.dtype, dims.join(", "))
    }
}

/// Plain names print as they are, so ordinary graphs read naturally. Anything
/// that could be mistaken for dump syntax (empty, spaces, control characters,
/// a leading `%`) is quoted and escaped, so a name can never add a line or
/// pass for a value id.
fn shown(name: &str) -> String {
    let plain = !name.is_empty()
        && !name.starts_with('%')
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'));
    if plain {
        name.to_owned()
    } else {
        format!("{name:?}")
    }
}

fn op_with_attributes(op: &Op) -> String {
    match op {
        Op::Reshape { shape } => format!("Reshape[shape={shape:?}]"),
        Op::Transpose { perm } => format!("Transpose[perm={perm:?}]"),
        other => other.name().to_owned(),
    }
}

impl Display for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph {{")?;
        for (index, value) in self.values.iter().enumerate() {
            let name = match &value.name {
                Some(name) => shown(name),
                None => self.label(crate::ValueId(index as u32)),
            };
            let ty = &value.ty;
            match &value.source {
                Source::Input => writeln!(f, "  %{index} {name} = input : {ty}")?,
                Source::Constant(data) if data.len() <= MAX_SHOWN => {
                    writeln!(f, "  %{index} {name} = const : {ty} = {data:?}")?
                }
                Source::Constant(data) => writeln!(
                    f,
                    "  %{index} {name} = const : {ty} = [{} values]",
                    data.len()
                )?,
                Source::Node { op, inputs } => {
                    let args: Vec<String> = inputs.iter().map(|id| format!("%{}", id.0)).collect();
                    writeln!(
                        f,
                        "  %{index} {name} = {}({}) : {ty}",
                        op_with_attributes(op),
                        args.join(", ")
                    )?
                }
            }
        }
        for (name, id) in self.output_names.iter().zip(&self.outputs) {
            writeln!(f, "  output {} = %{}", shown(name), id.0)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use crate::{GraphBuilder, Limits, Op};

    fn builder() -> GraphBuilder {
        GraphBuilder::new(Limits::default())
    }

    #[test]
    fn dump_shows_every_value_and_output() {
        let mut b = builder();
        let x = b.input("x", vec![2, 3]).unwrap();
        let w = b
            .constant("w", vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
        let p = b.node("projection", Op::MatMul, &[x, w]).unwrap();
        let r = b.node("r", Op::Reshape { shape: vec![4] }, &[p]).unwrap();
        let g = b.finish(&[r]).unwrap();
        assert_eq!(
            g.to_string(),
            "graph {\n\
             \x20 %0 x = input : f32[2, 3]\n\
             \x20 %1 w = const : f32[3, 2] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]\n\
             \x20 %2 projection = MatMul(%0, %1) : f32[2, 2]\n\
             \x20 %3 r = Reshape[shape=[4]](%2) : f32[4]\n\
             \x20 output r = %3\n\
             }"
        );
    }

    #[test]
    fn dump_prints_scalars_and_transpose_attributes() {
        let mut b = builder();
        let s = b.constant("s", vec![], vec![-0.0]).unwrap();
        let m = b.input("m", vec![2, 3]).unwrap();
        let t = b
            .node("t", Op::Transpose { perm: vec![1, 0] }, &[m])
            .unwrap();
        let g = b.finish(&[s, t]).unwrap();
        let text = g.to_string();
        assert!(text.contains("%0 s = const : f32[] = [-0.0]"), "{text}");
        assert!(
            text.contains("%2 t = Transpose[perm=[1, 0]](%1) : f32[3, 2]"),
            "{text}"
        );
    }

    #[test]
    fn long_constants_are_summarized() {
        let mut b = builder();
        let c = b.constant("big", vec![9], vec![0.5; 9]).unwrap();
        let g = b.finish(&[c]).unwrap();
        assert!(
            g.to_string()
                .contains("%0 big = const : f32[9] = [9 values]")
        );
    }

    #[test]
    fn output_names_are_printed_not_value_names() {
        let mut b = builder();
        let x = b.input("x", vec![1]).unwrap();
        let g = b
            .finish_named(vec![("first".to_owned(), x), ("second".to_owned(), x)])
            .unwrap();
        let text = g.to_string();
        assert!(text.contains("output first = %0\n"), "{text}");
        assert!(text.contains("output second = %0\n"), "{text}");
    }

    #[test]
    fn dump_is_deterministic() {
        let mut b = builder();
        let x = b.input("x", vec![2]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        let g = b.finish(&[y]).unwrap();
        assert_eq!(g.to_string(), g.clone().to_string());
    }

    #[test]
    fn unusual_names_are_quoted_so_the_dump_stays_unambiguous() {
        let mut b = builder();
        b.input("", vec![1]).unwrap();
        b.input("a\nb", vec![1]).unwrap();
        b.input("%0", vec![1]).unwrap();
        let plain = b.input("plain_name-1.x:y", vec![1]).unwrap();
        let g = b.finish_named(vec![("out put".to_owned(), plain)]).unwrap();
        let text = g.to_string();
        assert!(text.contains("%0 \"\" = input"), "{text}");
        assert!(text.contains("%1 \"a\\nb\" = input"), "{text}");
        assert!(text.contains("%2 \"%0\" = input"), "{text}");
        assert!(text.contains("%3 plain_name-1.x:y = input"), "{text}");
        assert!(text.contains("output \"out put\" = %3"), "{text}");
        assert_eq!(
            text.lines().count(),
            7,
            "a name must never add a line: {text}"
        );
    }
}
