//! The one way passes build their result: copy values from the old graph
//! into a fresh builder through an old-to-new id map. A pass decides, per
//! value, whether to copy it, alias it to an earlier value or replace it
//! with a constant. Names and output names carry over, so the graph
//! interface cannot change by accident.

use crate::{Error, Graph, GraphBuilder, Limits, Source, ValueId};
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
    pub(crate) fn mapped(&self, old: ValueId) -> ValueId {
        self.map[old.index()].expect("a value is rebuilt before any value that uses it")
    }

    /// `old` is represented by the already-built `new` from now on.
    pub(crate) fn alias(&mut self, old: ValueId, new: ValueId) {
        self.map[old.index()] = Some(new);
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
                let inputs: Vec<ValueId> = inputs.iter().map(|&id| self.mapped(id)).collect();
                self.builder.node(&name, op.clone(), &inputs)?
            }
        };
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
            .map(|(name, &id)| (name.clone(), self.mapped(id)))
            .collect();
        self.builder.finish_named(outputs)
    }
}
