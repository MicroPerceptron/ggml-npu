---
name: update-project-memory
description: Merges verified durable repository knowledge into .cursor/project-memory.md. Use after work changes or discovers architecture decisions, verified findings, constraints, rejected alternatives, reproducible benchmark baselines, or pending milestones with acceptance criteria.
---

# Update Project Memory

Update `.cursor/project-memory.md` only when direct repository evidence establishes a durable delta.

## Eligible content

Use exactly these categories and key prefixes:

- `ADR/<stable-slug>`: architecture decision and rationale.
- `FINDING/<stable-slug>`: verified finding with bounded scope.
- `CONSTRAINT/<stable-slug>`: durable technical or product constraint.
- `REJECTED/<stable-slug>`: rejected alternative and rationale.
- `BENCH/<stable-slug>`: reproducible benchmark baseline.
- `MILESTONE/<stable-slug>`: pending durable outcome with acceptance criteria.

Reject secrets, tokens, credentials, private keys, personal data, guesses, unsupported conclusions,
transient progress, raw logs, temporary failures, conversational summaries, and absolute local
paths. Do not turn a plan, discussion, or unverified experiment into a durable fact.

## Merge workflow

1. Read `.cursor/project-memory.md` and inspect direct code, test, tracked design-document, or
   benchmark-artifact evidence for every candidate.
2. Reject candidates outside the six eligible categories.
3. Assign a stable category/slug key based on meaning, not task wording or date.
4. Merge by key and meaning:
   - If an existing entry already entails the candidate, do nothing.
   - Consolidate semantic duplicates under one canonical key.
   - Preserve an existing entry when a candidate is uncertain, narrower without value, or older.
   - Replace or explicitly supersede a conflict only when newer direct evidence verifies it.
   - Preserve rationale when a decision or rejected alternative is superseded.
5. Keep the statement concise, scope its claim, and cite repository-relative evidence paths or a
   sanitized reproduction command. Summarize evidence; never paste raw logs.
6. Before writing, reject added text containing local absolute paths such as `/home/...`,
   `/Users/...`, drive-letter paths, UNC paths, or `file://` URLs.
7. Scan candidate text for secret-like assignments or material, including API keys, bearer tokens,
   passwords, credentials, private-key blocks, cookies, and connection strings. Reject rather than
   redact ambiguous source material.
8. For `BENCH`, include metric, workload and configuration, relevant hardware/toolchain, evidence,
   limitations, and a sanitized reproducible command. Never generalize beyond the measured scope.
9. For `MILESTONE`, record a stable outcome and objective acceptance criteria, not daily progress.
   Remove it once completion is directly verified, preserving any resulting decision or finding.
10. Re-read the edited entry against its evidence and the exclusion list. If no eligible delta
    remains, do not edit the file.

## Entry shape

Use the smallest structure that preserves meaning:

```markdown
- `PREFIX/stable-slug` — Durable statement.
  - Rationale|Scope|Acceptance|Limitation: Necessary qualifier.
  - Evidence: `repository/relative/path` or sanitized command.
  - Verified: YYYY-MM-DD
```

Not every qualifier is required. Pending milestones omit `Verified` until complete.

After a successful merge or a no-delta review, do not emit
`<!-- CURSOR_PROJECT_MEMORY_PENDING_7f3b -->`.
