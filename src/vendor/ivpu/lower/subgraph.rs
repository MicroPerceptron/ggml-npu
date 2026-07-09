//! Fused-subgraph lowering for the Intel NPU: compile the recognized op-DAG into one blob (weights
//! baked as `Constant`s) and dispatch it as a single program — one graph, one recorded execute
//! list, one submit + sync — instead of a dispatch per node. Compilation is cached per subgraph
//! instance (see [`super::super::compile`]).

use alloc::vec::Vec;

use super::super::{compile, session};
use crate::ops::subgraph::Subgraph;
use crate::ops::Operand;

pub fn run(sg: &Subgraph) -> bool {
    let Some(blob) = compile::get_or_compile_subgraph(sg) else {
        return false;
    };
    let inputs: Vec<&Operand> = sg.inputs.iter().collect();
    let outs: Vec<&Operand> = sg.outs.iter().collect();
    session::run(blob, &inputs, &outs)
}
