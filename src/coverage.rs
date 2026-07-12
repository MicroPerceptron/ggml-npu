//! Claim ⊆ coverage gate for NPU wire ops and subgraph patterns.
//!
//! The authoritative matrix lives at `tests/coverage_matrix.csv`. Hermetic CI fails when:
//! - a [`WireOp`] is missing from the matrix;
//! - Intel compile claims an op the matrix does not mark `compile_claimed=yes`;
//! - a compile-claimed row lacks a non-empty proof tag;
//! - a proof tag points at a missing script path.
//!
//! Hardware / shim numerical checks remain opt-in via `tests/run_coverage_gate.sh` and env flags.
//! Capability/policy unification for silent fallbacks is owned by issue #8 — this module only
//! audits the existing claim surface against the matrix.

#![cfg(any(test, feature = "test-support"))]

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::WireOp;
use crate::vendor::ivpu::wire_op_compile_supported;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverageRow {
    pub wire_op: WireOp,
    pub compile_claimed: bool,
    pub gate: String,
    pub proof: Vec<String>,
    pub notes: String,
}

const ALLOWED_GATES: &[&str] = &["hermetic", "shim", "opt_in_hw", "none"];

pub fn load_coverage_matrix(csv: &str) -> Result<Vec<CoverageRow>, String> {
    let mut lines = csv.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| String::from("coverage matrix is empty"))?;
    let expected = "wire_op,compile_claimed,gate,proof,notes";
    if header.trim() != expected {
        return Err(format!(
            "unexpected coverage matrix header: {header:?} (expected {expected})"
        ));
    }

    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        let cols = split_csv_line(line);
        if cols.len() != 5 {
            return Err(format!(
                "matrix row {} has {} columns (want 5): {line}",
                index + 2,
                cols.len()
            ));
        }
        let wire_op = WireOp::from_name(&cols[0])
            .ok_or_else(|| format!("unknown wire_op {:?} on matrix row {}", cols[0], index + 2))?;
        let compile_claimed = match cols[1].as_str() {
            "yes" => true,
            "no" => false,
            other => {
                return Err(format!(
                    "row {}: compile_claimed must be yes|no, got {other}",
                    index + 2
                ))
            }
        };
        if !ALLOWED_GATES.contains(&cols[2].as_str()) {
            return Err(format!(
                "row {}: gate must be one of {ALLOWED_GATES:?}, got {}",
                index + 2,
                cols[2]
            ));
        }
        let proof = cols[3]
            .split(';')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(String::from)
            .collect::<Vec<_>>();
        rows.push(CoverageRow {
            wire_op,
            compile_claimed,
            gate: cols[2].clone(),
            proof,
            notes: cols[4].clone(),
        });
    }
    Ok(rows)
}

fn split_csv_line(line: &str) -> Vec<String> {
    let mut cols = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                cols.push(core::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    cols.push(cur);
    cols
}

pub fn validate_claim_coverage(rows: &[CoverageRow], root: &std::path::Path) -> Result<(), String> {
    let mut seen = alloc::collections::BTreeSet::new();
    let mut errors = Vec::new();

    for row in rows {
        if !seen.insert(row.wire_op.as_str()) {
            errors.push(format!("duplicate matrix row for {}", row.wire_op.as_str()));
        }
        let compile_claims = wire_op_compile_supported(row.wire_op);
        if compile_claims != row.compile_claimed {
            errors.push(format!(
                "{}: compile claim {} but matrix compile_claimed={}",
                row.wire_op.as_str(),
                compile_claims,
                if row.compile_claimed { "yes" } else { "no" }
            ));
        }
        if row.compile_claimed {
            if row.gate == "none" {
                errors.push(format!(
                    "{}: compile-claimed op cannot use gate=none",
                    row.wire_op.as_str()
                ));
            }
            if row.proof.is_empty() {
                errors.push(format!(
                    "{}: compile-claimed op lacks proof tags",
                    row.wire_op.as_str()
                ));
            } else if !row.proof.iter().any(|tag| tag.starts_with("contract:")) {
                errors.push(format!(
                    "{}: compile-claimed op must include a contract: proof tag",
                    row.wire_op.as_str()
                ));
            }
        }
        for tag in &row.proof {
            if let Some(rel) = tag.strip_prefix("script:") {
                let path = root.join(rel);
                if !path.is_file() {
                    errors.push(format!(
                        "{}: proof script missing: {rel}",
                        row.wire_op.as_str()
                    ));
                }
            } else if !(tag.starts_with("contract:")
                || tag.starts_with("test:")
                || tag.starts_with("issue:#"))
            {
                errors.push(format!(
                    "{}: unknown proof tag kind in {tag}",
                    row.wire_op.as_str()
                ));
            }
        }
    }

    for op in WireOp::ALL {
        if !seen.contains(op.as_str()) {
            errors.push(format!(
                "WireOp::{} missing from coverage matrix",
                op.as_str()
            ));
        }
        if wire_op_compile_supported(*op) && !seen.contains(op.as_str()) {
            errors.push(format!(
                "compile claims {} but matrix has no row",
                op.as_str()
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn matrix_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/coverage_matrix.csv")
    }

    fn matrix_text() -> String {
        std::fs::read_to_string(matrix_path()).expect("read coverage matrix")
    }

    #[test]
    fn coverage_matrix_covers_every_wire_op_claim() {
        let rows = load_coverage_matrix(&matrix_text()).expect("parse coverage matrix");
        validate_claim_coverage(&rows, PathBuf::from(env!("CARGO_MANIFEST_DIR")).as_path())
            .unwrap_or_else(|err| panic!("coverage gate failed:\n{err}"));
    }

    #[test]
    fn wire_op_all_matches_as_str_round_trip() {
        for op in WireOp::ALL {
            assert_eq!(WireOp::from_name(op.as_str()), Some(*op));
        }
        assert_eq!(WireOp::ALL.len(), 26);
    }

    #[test]
    fn compile_claim_predicate_is_exhaustive_over_wire_op_all() {
        // Forces a compile error if WireOp grows without updating the predicate match arms
        // (via the matrix sync test) and keeps the ALL list honest.
        let claimed = WireOp::ALL
            .iter()
            .copied()
            .filter(|op| wire_op_compile_supported(*op))
            .count();
        assert!(claimed >= 20, "unexpectedly few compile-claimed wire ops: {claimed}");
    }
}
