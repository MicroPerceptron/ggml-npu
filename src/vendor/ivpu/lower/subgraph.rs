//! Fused-subgraph lowering for the Intel NPU: compile the recognized op-DAG into one blob (weights
//! baked as `Constant`s) and dispatch it as a single program — one graph, one recorded execute
//! list, one submit + sync — instead of a dispatch per node. The built graph is cached per subgraph
//! signature in [`super::super::session`]; the [`super::super::compile`] blob it's built from is
//! transient (dropped after the graph is built).

use alloc::vec::Vec;

use super::super::{compile, session};
use crate::ops::subgraph::Subgraph;
use crate::ops::Operand;

pub fn run(sg: &Subgraph) -> bool {
    // Key the program by its structural signature (stable across tokens; shared by weightless glue
    // subgraphs across layers). On a cold miss the closure compiles → builds the graph → drops the
    // transient blob; steady state is a pure graph-cache hit.
    let key = compile::sg_semantic_key(sg);
    let inputs: Vec<&Operand> = sg.inputs.iter().collect();
    let outs: Vec<&Operand> = sg.outs.iter().collect();
    let ok = session::run_subgraph(key, &inputs, &outs, || compile::compile_subgraph_blob(sg));
    if !ok {
        super::super::profile::note(&alloc::format!(
            "[npu] subgraph run FAIL (m={} inputs={} consts={} nodes={} outs={})\n",
            sg.m,
            sg.inputs.len(),
            sg.consts.len(),
            sg.nodes.len(),
            sg.outs.len(),
        ));
    }
    ok
}
