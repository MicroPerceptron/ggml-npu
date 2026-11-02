//! Conservative ggml-to-TOSA lowering. Tensor data is never baked into cache keys.
use crate::ffi::*;
use std::collections::HashMap;
use virtio_accel_tosa::{CapabilityDescriptor, DType, Op};
use virtio_accel_tosa_build::{OperatorKind as K, OwnedGraph, OwnedOperator, OwnedTensor};

pub const MAX_NODES: usize = 128;
pub const MAX_BYTES: usize = 128 * 1024 * 1024;

pub struct Capture {
    pub artifact: Vec<u8>,
    pub inputs: Vec<*mut ggml_tensor>,
    pub outputs: Vec<*mut ggml_tensor>,
}

// ggml tensors are caller-owned and live for the synchronous graph_compute call.
unsafe fn shape(t: *const ggml_tensor) -> Option<Vec<i32>> {
    let t = unsafe { t.as_ref()? };
    if t.type_ != GGML_TYPE_F32
        || t.ne[2..] != [1, 1]
        || t.ne[..2].iter().any(|&n| !(1..=8192).contains(&n))
    {
        return None;
    }
    if t.nb[0] < 4 || t.nb[0] % 4 != 0 || t.nb[1] % 4 != 0 {
        return None;
    }
    let bytes = usize::try_from(t.ne[0])
        .ok()?
        .checked_mul(t.ne[1] as usize)?
        .checked_mul(4)?;
    if bytes > MAX_BYTES {
        return None;
    }
    // Bound pointer arithmetic for a legal host view, including padded or transposed rows.
    let extent = (t.ne[0] as usize - 1)
        .checked_mul(t.nb[0])?
        .checked_add((t.ne[1] as usize - 1).checked_mul(t.nb[1])?)?
        .checked_add(4)?;
    if extent > isize::MAX as usize {
        return None;
    }
    Some(vec![1, t.ne[1] as i32, t.ne[0] as i32])
}

pub unsafe fn byte_len(t: *const ggml_tensor) -> Result<usize, String> {
    let s = unsafe { shape(t) }.ok_or("unsupported tensor layout")?;
    Ok(s.iter().map(|&v| v as usize).product::<usize>() * 4)
}

unsafe fn kind(t: *const ggml_tensor) -> Option<K> {
    let t = unsafe { t.as_ref()? };
    match t.op {
        GGML_OP_MUL_MAT => Some(K::MatMul),
        GGML_OP_ADD => Some(K::Add),
        GGML_OP_MUL => Some(K::Mul),
        GGML_OP_UNARY => match t.op_params[0] as u32 {
            GGML_UNARY_OP_TANH => Some(K::Tanh),
            GGML_UNARY_OP_SILU => Some(K::Sigmoid), // followed by MUL below
            _ => None,
        },
        _ => None,
    }
}

pub fn metadata(op: ggml_op) -> bool {
    matches!(
        op,
        GGML_OP_NONE | GGML_OP_VIEW | GGML_OP_RESHAPE | GGML_OP_PERMUTE | GGML_OP_TRANSPOSE
    )
}

pub unsafe fn supports(t: *const ggml_tensor, cap: CapabilityDescriptor) -> bool {
    unsafe {
        let Some(k) = kind(t) else {
            return false;
        };
        let Some(out) = shape(t) else {
            return false;
        };
        if !(*t).view_src.is_null() {
            return false;
        } // no state writes/in-place outputs
        let Some(a) = shape((*t).src[0]) else {
            return false;
        };
        // Views of computed values must form a scheduler boundary in this first lowering.
        for &src in &(&(*t).src)[..if matches!(k, K::Tanh | K::Sigmoid) {
            1
        } else {
            2
        }] {
            if src.is_null() {
                return false;
            }
            let mut v = src;
            for _ in 0..MAX_NODES {
                if (*v).view_src.is_null() {
                    break;
                }
                v = (*v).view_src;
                if !metadata((*v).op) {
                    return false;
                }
            }
            if !(*v).view_src.is_null() {
                return false;
            }
        }
        let required: &[Op] = match k {
            K::MatMul => {
                let Some(b) = shape((*t).src[1]) else {
                    return false;
                };
                if a[2] != b[2] || out != [1, b[1], a[1]] {
                    return false;
                }
                &[Op::MATMUL, Op::TRANSPOSE, Op::CONST]
            }
            K::Add | K::Mul => {
                let Some(b) = shape((*t).src[1]) else {
                    return false;
                };
                if a != out || b.iter().zip(&out).any(|(&b, &o)| b != 1 && b != o) {
                    return false;
                }
                if k == K::Add {
                    &[Op::ADD]
                } else {
                    &[Op::MUL, Op::CONST]
                }
            }
            K::Tanh => {
                if a != out {
                    return false;
                }
                &[Op::TANH]
            }
            K::Sigmoid => {
                if a != out {
                    return false;
                }
                &[Op::SIGMOID, Op::MUL, Op::CONST]
            }
            _ => return false,
        };
        required.iter().all(|&op| cap.supports_operator(op))
    }
}

fn op(g: &mut OwnedGraph<'static>, k: K, inputs: Vec<String>, output: String) {
    g.push_operator(OwnedOperator::new(k, inputs, vec![output]));
}
fn constant(g: &mut OwnedGraph<'static>, name: &str, dtype: DType, data: Vec<u8>) {
    g.push_tensor(OwnedTensor::constant(name, vec![1], dtype, data));
    op(g, K::Const, vec![], name.into());
}

pub unsafe fn capture(
    cgraph: *mut ggml_cgraph,
    cap: CapabilityDescriptor,
) -> Result<Capture, String> {
    unsafe {
        if cgraph.is_null() {
            return Err("null graph".into());
        }
        let count = ggml_graph_n_nodes(cgraph);
        if count < 0 || count as usize > MAX_NODES {
            return Err("graph exceeds bridge node limit".into());
        }
        let mut g = OwnedGraph::new("ggml");
        let mut values = HashMap::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut total_bytes = 0usize;
        for i in 0..count {
            let t = ggml_graph_node(cgraph, i);
            if t.is_null() {
                return Err("null graph node".into());
            }
            if metadata((*t).op) {
                continue;
            }
            if !supports(t, cap) {
                return Err(format!("unsupported ggml operation {}", (*t).op));
            }
            let k = kind(t).ok_or("unsupported operation")?;
            let nsrc = if matches!(k, K::Tanh | K::Sigmoid) {
                1
            } else {
                2
            };
            let mut args = Vec::new();
            for &src in &(&(*t).src)[..nsrc] {
                if let Some(name) = values.get(&(src as usize)) {
                    args.push(String::clone(name));
                    continue;
                }
                let mut base = (*src).view_src;
                for _ in 0..MAX_NODES {
                    if base.is_null() {
                        break;
                    }
                    if values.contains_key(&(base as usize)) {
                        return Err("view of an intermediate requires a split".into());
                    }
                    base = (*base).view_src;
                }
                if !base.is_null() {
                    return Err("cyclic or excessive view ancestry".into());
                }
                let name = format!("i{}", inputs.len());
                g.push_tensor(OwnedTensor::new(
                    &name,
                    shape(src).ok_or("input shape")?,
                    DType::FP32,
                ));
                g.push_input(&name);
                total_bytes = total_bytes
                    .checked_add(byte_len(src)?)
                    .ok_or("size overflow")?;
                inputs.push(src);
                values.insert(src as usize, name.clone());
                args.push(name);
            }
            let name = format!("o{}", outputs.len());
            let dims = shape(t).ok_or("output shape")?;
            g.push_tensor(OwnedTensor::new(&name, dims.clone(), DType::FP32));
            match k {
                K::MatMul => {
                    let a = shape((*t).src[0]).ok_or("matmul shape")?;
                    let transposed = format!("{name}_wt");
                    g.push_tensor(OwnedTensor::new(
                        &transposed,
                        vec![1, a[2], a[1]],
                        DType::FP32,
                    ));
                    op(
                        &mut g,
                        K::Transpose {
                            perms: [0, 2, 1, 0, 0, 0],
                            rank: 3,
                        },
                        vec![args[0].clone()],
                        transposed.clone(),
                    );
                    let zero = format!("{name}_zero");
                    constant(&mut g, &zero, DType::FP32, 0f32.to_le_bytes().to_vec());
                    op(
                        &mut g,
                        k,
                        vec![args[1].clone(), transposed, zero.clone(), zero],
                        name.clone(),
                    );
                }
                K::Mul | K::Sigmoid => {
                    let shift = format!("{name}_shift");
                    constant(&mut g, &shift, DType::INT8, vec![0]);
                    if k == K::Sigmoid {
                        let sig = format!("{name}_sig");
                        g.push_tensor(OwnedTensor::new(&sig, dims, DType::FP32));
                        op(&mut g, K::Sigmoid, args.clone(), sig.clone());
                        args.push(sig);
                    }
                    args.push(shift);
                    op(&mut g, K::Mul, args, name.clone());
                }
                _ => op(&mut g, k, args, name.clone()),
            }
            g.push_output(&name); // publish every result: ggml may consume intermediate nodes later
            total_bytes = total_bytes
                .checked_add(byte_len(t)?)
                .ok_or("size overflow")?;
            if total_bytes > MAX_BYTES {
                return Err("graph exceeds bridge transfer budget".into());
            }
            values.insert(t as usize, name);
            outputs.push(t);
        }
        let artifact = if outputs.is_empty() {
            Vec::new()
        } else {
            g.build(cap.target).map_err(|e| e.to_string())?
        };
        Ok(Capture {
            artifact,
            inputs,
            outputs,
        })
    }
}

unsafe fn host_tensor(t: *const ggml_tensor) -> Result<(), String> {
    unsafe {
        if t.is_null() || (*t).data.is_null() {
            return Err("unallocated tensor".into());
        }
        if !(*t).buffer.is_null() && !ggml_backend_buft_is_host((*(*t).buffer).buft) {
            return Err("foreign device buffer must be staged by ggml".into());
        }
        Ok(())
    }
}

pub unsafe fn read(t: *const ggml_tensor) -> Result<Vec<u8>, String> {
    unsafe {
        host_tensor(t)?;
        let mut bytes = Vec::with_capacity(byte_len(t)?);
        for y in 0..(*t).ne[1] as usize {
            for x in 0..(*t).ne[0] as usize {
                let p = (*t)
                    .data
                    .cast::<u8>()
                    .add(y * (*t).nb[1] + x * (*t).nb[0])
                    .cast::<f32>();
                bytes.extend_from_slice(&p.read_unaligned().to_le_bytes());
            }
        }
        Ok(bytes)
    }
}

pub unsafe fn write(t: *mut ggml_tensor, bytes: &[u8]) -> Result<(), String> {
    unsafe {
        host_tensor(t)?;
        if bytes.len() != byte_len(t)? {
            return Err("incorrect output length".into());
        }
        for y in 0..(*t).ne[1] as usize {
            for x in 0..(*t).ne[0] as usize {
                let i = (y * (*t).ne[0] as usize + x) * 4;
                let v = f32::from_le_bytes(bytes[i..i + 4].try_into().map_err(|_| "output lane")?);
                (*t).data
                    .cast::<u8>()
                    .add(y * (*t).nb[1] + x * (*t).nb[0])
                    .cast::<f32>()
                    .write_unaligned(v);
            }
        }
        Ok(())
    }
}
