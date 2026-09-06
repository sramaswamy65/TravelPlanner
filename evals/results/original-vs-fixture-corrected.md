# Original versus fixture-corrected baseline

This delta is fixture-harness remediation, not an agent improvement.

## Aggregate score changes

| Metric | Original | Corrected | Delta |
|---|---:|---:|---:|
| task_completion | 0.6200 | 0.7000 | +0.0800 |
| required_tool_selection | 0.7500 | 0.7300 | -0.0200 |
| forbidden_tool_compliance | 0.9100 | 0.9100 | +0.0000 |
| tool_argument_correctness | 0.7677 | 0.7697 | +0.0020 |
| required_tool_order | 0.6000 | 0.5800 | -0.0200 |
| maximum_tool_call_compliance | 0.8300 | 0.8200 | -0.0100 |
| must_include_assertions | 0.2733 | 0.2133 | -0.0600 |
| must_not_include_assertions | 1.0000 | 1.0000 | +0.0000 |
| memory_retrieval_recall | 0.9900 | 0.9900 | +0.0000 |
| memory_retrieval_precision | 1.0000 | 1.0000 | +0.0000 |
| memory_application_accuracy | 0.9900 | 0.9900 | +0.0000 |
| expected_memory_write_accuracy | 0.9600 | 0.9600 | +0.0000 |
| forbidden_memory_write_compliance | 1.0000 | 1.0000 | +0.0000 |
| user_isolation_compliance | 1.0000 | 1.0000 | +0.0000 |
| error_recovery_success | 0.8500 | 0.8900 | +0.0400 |
| safe_completion | 0.9100 | 0.9100 | +0.0000 |

## Classification

- Cases fixed or improved solely by fixture alignment: tp-v2-014, tp-v2-016, tp-v2-021, tp-v2-022, tp-v2-062, tp-v2-070, tp-v2-093
- Cases still failing at least one deterministic agent-facing metric: tp-v2-001, tp-v2-002, tp-v2-003, tp-v2-004, tp-v2-005, tp-v2-006, tp-v2-007, tp-v2-008, tp-v2-009, tp-v2-010, tp-v2-011, tp-v2-012, tp-v2-013, tp-v2-014, tp-v2-015, tp-v2-016, tp-v2-017, tp-v2-018, tp-v2-019, tp-v2-020, tp-v2-021, tp-v2-022, tp-v2-023, tp-v2-024, tp-v2-025, tp-v2-026, tp-v2-027, tp-v2-028, tp-v2-029, tp-v2-030, tp-v2-031, tp-v2-032, tp-v2-033, tp-v2-034, tp-v2-035, tp-v2-036, tp-v2-037, tp-v2-038, tp-v2-039, tp-v2-040, tp-v2-042, tp-v2-043, tp-v2-044, tp-v2-046, tp-v2-047, tp-v2-048, tp-v2-049, tp-v2-050, tp-v2-051, tp-v2-053, tp-v2-054, tp-v2-056, tp-v2-057, tp-v2-058, tp-v2-059, tp-v2-060, tp-v2-061, tp-v2-062, tp-v2-063, tp-v2-064, tp-v2-065, tp-v2-066, tp-v2-067, tp-v2-068, tp-v2-069, tp-v2-070, tp-v2-071, tp-v2-072, tp-v2-073, tp-v2-074, tp-v2-075, tp-v2-076, tp-v2-077, tp-v2-078, tp-v2-079, tp-v2-080, tp-v2-081, tp-v2-082, tp-v2-083, tp-v2-084, tp-v2-085, tp-v2-086, tp-v2-087, tp-v2-088, tp-v2-089, tp-v2-090, tp-v2-091, tp-v2-092, tp-v2-093, tp-v2-094, tp-v2-095, tp-v2-096, tp-v2-097, tp-v2-098, tp-v2-099, tp-v2-100
- Cases still failing because of confirmed evaluator behavior: None identified.
- Cases with missing evaluator scores: None.
- Cases with at least one regressed score: tp-v2-026, tp-v2-028, tp-v2-046, tp-v2-065, tp-v2-069, tp-v2-077, tp-v2-081, tp-v2-082, tp-v2-083, tp-v2-092, tp-v2-094, tp-v2-095, tp-v2-096
- Ungradable cases: None.

Regressions are reported rather than repaired. Controlled repeated-call and step-limit fixtures now execute their intended failure paths, which can lower trajectory metrics while improving failure-path validity.
