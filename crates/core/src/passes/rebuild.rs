//! The one way passes build their result: copy values from the old graph
//! into a fresh builder through an old-to-new id map. A pass decides, per
//! value, whether to copy it, alias it to an earlier value or replace it
//! with a constant. Names and output names carry over, so the graph
//! interface cannot change by accident.

use crate::{Error, Graph, GraphBuilder, Limits, Op, Source, ValueId};
use std::sync::Arc;

pub(crate) struct Rebuilder<'g> {
    graph: &'g Graph,
    builder: GraphBuilder,
    map: Vec<Option<ValueId>>,
}

impl<'g> Rebuilder<'g> {
    pub(crate) fn new(graph: &'g Graph, limits: &Limits) -> Self {
        Self {
            graph,
            builder: GraphBuilder::new(limits.clone()),
            map: vec![None; graph.values().len()],
        }
    }

    /// The new id for an old value. Values are rebuilt in definition order,
    /// so every input of a node is mapped before the node is.
    pub(crate) fn mapped(&self, old: ValueId) -> Result<ValueId, Error> {
        self.map
            .get(old.index())
            .copied()
            .flatten()
            .ok_or_else(|| Error::Verify(format!("value %{} is used before it is rebuilt", old.0)))
    }

    /// `old` is represented by the already-built `new` from now on.
    pub(crate) fn alias(&mut self, old: ValueId, new: ValueId) {
        // `map` has one slot per value of the graph, so `old` is in range; if it were
        // not, `mapped` would report the value as never rebuilt.
        if let Some(slot) = self.map.get_mut(old.index()) {
            *slot = Some(new);
        }
    }

    /// Copies `old` with its inputs remapped.
    pub(crate) fn copy(&mut self, old: ValueId) -> Result<ValueId, Error> {
        let value = self.graph.value(old);
        let name = self.graph.label(old);
        let new = match &value.source {
            Source::Input => self.builder.input(&name, value.ty.shape.clone())?,
            Source::Constant(data) => {
                self.builder
                    .constant_shared(&name, value.ty.shape.clone(), Arc::clone(data))?
            }
            Source::Node { op, inputs } => {
                let inputs = inputs
                    .iter()
                    .map(|&id| self.mapped(id))
                    .collect::<Result<Vec<ValueId>, Error>>()?;
                self.builder.node(&name, op.clone(), &inputs)?
            }
        };
        self.alias(old, new);
        Ok(new)
    }

    /// Replaces `old` (a node) with a different node over already-built inputs.
    pub(crate) fn replace_with_node(
        &mut self,
        old: ValueId,
        op: Op,
        inputs: &[ValueId],
    ) -> Result<ValueId, Error> {
        let name = self.graph.label(old);
        let new = self.builder.node(&name, op, inputs)?;
        self.alias(old, new);
        Ok(new)
    }

    /// Replaces `old` (a node) with a constant of the same name and shape.
    pub(crate) fn replace_with_constant(
        &mut self,
        old: ValueId,
        data: Arc<[f32]>,
    ) -> Result<ValueId, Error> {
        let name = self.graph.label(old);
        let shape = self.graph.value(old).ty.shape.clone();
        let new = self.builder.constant_shared(&name, shape, data)?;
        self.alias(old, new);
        Ok(new)
    }

    /// Finishes with the original output names pointing at the mapped values.
    pub(crate) fn finish(self) -> Result<Graph, Error> {
        let outputs = self
            .graph
            .output_names()
            .iter()
            .zip(self.graph.outputs())
            .map(|(name, &id)| Ok((name.clone(), self.mapped(id)?)))
            .collect::<Result<_, Error>>()?;
        self.builder.finish_named(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_that_was_never_rebuilt_is_an_error_not_a_panic() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![1]).unwrap();
        let graph = b.finish(&[x]).unwrap();
        let rebuild = Rebuilder::new(&graph, &Limits::default());
        assert!(matches!(rebuild.mapped(x), Err(Error::Verify(_))));
        assert!(matches!(rebuild.mapped(ValueId(9)), Err(Error::Verify(_))));
    }

    #[test]
    fn a_rebuilt_value_maps_to_its_new_id() {
        let mut b = GraphBuilder::new(Limits::default());
        let x = b.input("x", vec![1]).unwrap();
        let graph = b.finish(&[x]).unwrap();
        let mut rebuild = Rebuilder::new(&graph, &Limits::default());
        let new = rebuild.copy(x).unwrap();
        assert_eq!(rebuild.mapped(x).unwrap(), new);
    }
}
