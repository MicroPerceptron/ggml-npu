//! Graph IR v2 lowering for Intel: compile a complete supported scheduler split into one OpenVINO
//! model and execute it as one Level Zero graph.

use alloc::vec::Vec;
use core::ffi::c_void;

use super::super::{compile, session};
use crate::graph_ir::{GraphIr, TensorRole};
use crate::ops::Operand;

pub fn can_lower(graph: &GraphIr) -> bool {
    compile::graph_feasible(graph)
}

pub fn run(graph: &GraphIr) -> bool {
    let inputs: Vec<Operand> = graph
        .inputs
        .iter()
        .filter_map(|&id| {
            let t = graph.values.get(id as usize)?;
            (t.role != TensorRole::Constant).then_some(Operand {
                ptr: t.data as *mut c_void,
                elems: t.dims.iter().try_fold(1usize, |a, d| match d {
                    crate::graph_ir::Dim::Static(x) if *x > 0 => a.checked_mul(*x as usize),
                    crate::graph_ir::Dim::Bounded { max, .. } if *max > 0 => {
                        a.checked_mul(*max as usize)
                    }
                    _ => None,
                })?,
                dtype: t.dtype,
                owned: t.owned,
            })
        })
        .collect();
    let outputs: Vec<Operand> = graph
        .outputs
        .iter()
        .filter_map(|&id| {
            let t = graph.values.get(id as usize)?;
            Some(Operand {
                ptr: t.data as *mut c_void,
                elems: t.dims.iter().try_fold(1usize, |a, d| match d {
                    crate::graph_ir::Dim::Static(x) if *x > 0 => a.checked_mul(*x as usize),
                    crate::graph_ir::Dim::Bounded { max, .. } if *max > 0 => {
                        a.checked_mul(*max as usize)
                    }
                    _ => None,
                })?,
                dtype: t.dtype,
                owned: t.owned,
            })
        })
        .collect();
    if inputs.len()
        != graph
            .inputs
            .iter()
            .filter(|&&id| graph.values[id as usize].role != TensorRole::Constant)
            .count()
        || outputs.len() != graph.outputs.len()
    {
        return false;
    }
    let input_refs: Vec<&Operand> = inputs.iter().collect();
    let output_refs: Vec<&Operand> = outputs.iter().collect();
    let key = graph.semantic_key();
    let result = session::run_subgraph(key, &input_refs, &output_refs, || {
        compile::compile_graph_blob(graph)
    });
    if !result {
        super::super::profile::note(&alloc::format!(
            "[npu ir2] runtime rejected graph: nodes={} inputs={} outputs={}\n",
            graph.nodes.len(),
            input_refs.len(),
            output_refs.len()
        ));
    }
    result
}
