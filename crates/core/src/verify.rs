//! Independent re-check of every IR invariant. The builder cannot produce an
//! invalid graph, so this exists to catch bugs in passes that rebuild graphs.

use crate::shape::{check_shape, checked_num_elements, infer};
use crate::{Error, Graph, Limits, Source, ValueId};
use std::collections::HashSet;

impl Graph {
    pub fn verify(&self) -> Result<(), Error> {
        if self.outputs.is_empty() {
            return Err(Error::NoOutputs);
        }
        if self.output_names.len() != self.outputs.len() {
            return Err(Error::Verify(
                "output names do not match the outputs".to_owned(),
            ));
        }
        let mut names: HashSet<&str> = HashSet::new();
        let mut input_ids = Vec::new();
        for (index, value) in self.values.iter().enumerate() {
            let id = ValueId(index as u32);
            let label = self.label(id);
            if value.ty.shape.contains(&0) {
                return Err(Error::InvalidShape {
                    name: label,
                    shape: value.ty.shape.clone(),
                    reason: "dimensions must be at least 1",
                });
            }
            if let Some(name) = &value.name
                && !names.insert(name.as_str())
            {
                return Err(Error::DuplicateName { name: name.clone() });
            }
            match &value.source {
                Source::Input => input_ids.push(id),
                Source::Constant(data) => {
                    let expected = checked_num_elements(&value.ty.shape).ok_or_else(|| {
                        Error::Verify(format!("`{label}` element count overflows"))
                    })?;
                    if data.len() != expected {
                        return Err(Error::DataLength {
                            name: label,
                            shape: value.ty.shape.clone(),
                            expected,
                            found: data.len(),
                        });
                    }
                }
                Source::Node { op, inputs } => {
                    let mut types = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        if input.index() >= index {
                            return Err(Error::Verify(format!(
                                "node `{label}` uses value %{} defined at or after it",
                                input.0
                            )));
                        }
                        types.push(&self.values[input.index()].ty);
                    }
                    let inferred = infer(&label, op, &types)?;
                    if inferred != value.ty {
                        return Err(Error::Verify(format!(
                            "node `{label}` stored type {:?} differs from inferred {:?}",
                            value.ty, inferred
                        )));
                    }
                }
            }
        }
        if input_ids != self.inputs {
            return Err(Error::Verify(
                "input list does not match the input values".to_owned(),
            ));
        }
        for output in &self.outputs {
            if output.index() >= self.values.len() {
                return Err(Error::Verify(format!(
                    "output %{} is out of range",
                    output.0
                )));
            }
        }
        Ok(())
    }

    /// Re-checks the resource limits the builder enforced. The builder cannot
    /// build a graph that breaks them, so this exists for the pass manager:
    /// a pass must not grow a graph past the limits it started under.
    pub fn verify_limits(&self, limits: &Limits) -> Result<(), Error> {
        if self.values.len() > limits.max_values {
            return Err(Error::LimitExceeded {
                what: "values",
                found: self.values.len(),
                limit: limits.max_values,
            });
        }
        let mut total = 0usize;
        for (index, value) in self.values.iter().enumerate() {
            let name = self.label(ValueId(index as u32));
            total = total.saturating_add(check_shape(&name, &value.ty.shape, limits)?);
            if total > limits.max_total_elements {
                return Err(Error::TensorLimit {
                    name,
                    what: "total tensor elements",
                    found: total,
                    limit: limits.max_total_elements,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Error, Graph, GraphBuilder, Limits, Op, Source, TensorType, ValueId};
    use std::sync::Arc;

    /// x -> y = Relu(x), plus a constant c, output y.
    fn valid() -> Graph {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![2, 3]).unwrap();
        b.constant("c", vec![2], vec![1.0, 2.0]).unwrap();
        let y = b.node("y", Op::Relu, &[x]).unwrap();
        b.finish(&[y]).unwrap()
    }

    #[test]
    fn builder_output_verifies() {
        assert_eq!(valid().verify(), Ok(()));
    }

    #[test]
    fn rejects_forward_reference() {
        let mut g = valid();
        g.values[2].source = Source::Node {
            op: Op::Relu,
            inputs: vec![ValueId(2)],
        };
        assert!(matches!(g.verify(), Err(Error::Verify(m)) if m.contains("defined at or after")));
    }

    #[test]
    fn rejects_stored_type_that_differs_from_inferred() {
        let mut g = valid();
        g.values[2].ty = TensorType::f32(vec![9]);
        assert!(matches!(g.verify(), Err(Error::Verify(m)) if m.contains("differs from inferred")));
    }

    #[test]
    fn rejects_empty_outputs() {
        let mut g = valid();
        g.outputs.clear();
        assert_eq!(g.verify(), Err(Error::NoOutputs));
    }

    #[test]
    fn rejects_output_out_of_range() {
        let mut g = valid();
        g.outputs = vec![ValueId(7)];
        assert!(matches!(g.verify(), Err(Error::Verify(m)) if m.contains("out of range")));
    }

    #[test]
    fn rejects_constant_with_wrong_data_length() {
        let mut g = valid();
        g.values[1].source = Source::Constant(Arc::from(vec![1.0f32]));
        assert!(matches!(
            g.verify(),
            Err(Error::DataLength {
                expected: 2,
                found: 1,
                ..
            })
        ));
    }

    #[test]
    fn rejects_input_list_that_disagrees_with_values() {
        let mut g = valid();
        g.inputs.clear();
        assert!(matches!(g.verify(), Err(Error::Verify(m)) if m.contains("input list")));
    }

    #[test]
    fn rejects_duplicate_names() {
        let mut g = valid();
        g.values[1].name = Some("x".into());
        assert_eq!(g.verify(), Err(Error::DuplicateName { name: "x".into() }));
    }

    #[test]
    fn rejects_output_names_that_do_not_match_outputs() {
        let mut g = valid();
        g.output_names.pop();
        assert!(matches!(g.verify(), Err(Error::Verify(m)) if m.contains("output names")));
    }

    #[test]
    fn rejects_a_zero_dimension_in_a_stored_type() {
        let mut g = valid();
        g.values[0].ty = TensorType::f32(vec![2, 0]);
        assert!(matches!(g.verify(), Err(Error::InvalidShape { .. })));
    }

    #[test]
    fn verify_limits_accepts_a_builder_graph() {
        assert_eq!(valid().verify_limits(&Limits::default()), Ok(()));
    }

    #[test]
    fn verify_limits_rejects_a_graph_built_under_looser_limits() {
        let loose = Limits {
            max_tensor_elements: 1 << 20,
            ..Limits::default()
        };
        let mut b = GraphBuilder::new(loose);
        let x = b.input("x", vec![1000, 1000]).unwrap();
        let g = b.finish(&[x]).unwrap();
        let strict = Limits {
            max_tensor_elements: 1000,
            ..Limits::default()
        };
        assert!(matches!(
            g.verify_limits(&strict),
            Err(Error::TensorLimit {
                what: "tensor elements",
                ..
            })
        ));
    }

    #[test]
    fn verify_limits_checks_the_value_count_and_the_total() {
        let g = valid();
        let few = Limits {
            max_values: 2,
            ..Limits::default()
        };
        assert!(matches!(
            g.verify_limits(&few),
            Err(Error::LimitExceeded { what: "values", .. })
        ));
        let small_total = Limits {
            max_total_elements: 7,
            ..Limits::default()
        };
        assert!(matches!(
            g.verify_limits(&small_total),
            Err(Error::TensorLimit {
                what: "total tensor elements",
                ..
            })
        ));
    }

    #[test]
    fn verify_limits_accepts_a_graph_exactly_at_the_limits() {
        // valid() has 3 values holding 6 + 2 + 6 = 14 elements.
        let exactly = Limits {
            max_values: 3,
            max_total_elements: 14,
            ..Limits::default()
        };
        assert_eq!(valid().verify_limits(&exactly), Ok(()));
        let one_less = Limits {
            max_total_elements: 13,
            ..exactly
        };
        assert!(valid().verify_limits(&one_less).is_err());
    }
}
