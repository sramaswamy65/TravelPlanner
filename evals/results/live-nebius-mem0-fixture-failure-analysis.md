# TravelPlanner baseline failure analysis

## Exact aggregate scores

- `task_completion`: 0.5783
- `tool_trajectory`: 0.8694
- `memory_behavior`: 0.9683
- `grounded_response`: 0.8707
- `safety_compliance`: 1.0
- `error_recovery`: 0.8933
- `safe_completion`: 0.985

## Three largest failure clusters

### task_completion
Frequency: 61/100
Cases: tp-v2-026, tp-v2-027, tp-v2-030, tp-v2-033, tp-v2-034, tp-v2-035, tp-v2-036, tp-v2-037, tp-v2-038, tp-v2-039, tp-v2-040, tp-v2-041, tp-v2-042, tp-v2-043, tp-v2-044, tp-v2-046, tp-v2-047, tp-v2-048, tp-v2-050, tp-v2-051, tp-v2-052, tp-v2-056, tp-v2-057, tp-v2-058, tp-v2-059, tp-v2-060, tp-v2-061, tp-v2-062, tp-v2-063, tp-v2-064, tp-v2-065, tp-v2-066, tp-v2-067, tp-v2-068, tp-v2-069, tp-v2-070, tp-v2-071, tp-v2-072, tp-v2-073, tp-v2-074, tp-v2-075, tp-v2-076, tp-v2-078, tp-v2-079, tp-v2-080, tp-v2-081, tp-v2-083, tp-v2-084, tp-v2-085, tp-v2-086, tp-v2-087, tp-v2-088, tp-v2-089, tp-v2-090, tp-v2-091, tp-v2-092, tp-v2-093, tp-v2-094, tp-v2-095, tp-v2-098, tp-v2-100
Classification: agent behavior unless the per-case record identifies a tool, fixture, dataset, or evaluator error.
Effect: lowers the named deterministic metric.
Recommendation: inspect representative traces before changing the agent or prompt.

### grounded_response
Frequency: 39/100
Cases: tp-v2-026, tp-v2-027, tp-v2-030, tp-v2-041, tp-v2-042, tp-v2-043, tp-v2-044, tp-v2-050, tp-v2-051, tp-v2-057, tp-v2-058, tp-v2-061, tp-v2-062, tp-v2-063, tp-v2-064, tp-v2-065, tp-v2-066, tp-v2-067, tp-v2-071, tp-v2-072, tp-v2-073, tp-v2-074, tp-v2-075, tp-v2-076, tp-v2-080, tp-v2-081, tp-v2-083, tp-v2-084, tp-v2-085, tp-v2-087, tp-v2-088, tp-v2-089, tp-v2-090, tp-v2-091, tp-v2-092, tp-v2-093, tp-v2-094, tp-v2-098, tp-v2-100
Classification: agent behavior unless the per-case record identifies a tool, fixture, dataset, or evaluator error.
Effect: lowers the named deterministic metric.
Recommendation: inspect representative traces before changing the agent or prompt.

### tool_trajectory
Frequency: 37/100
Cases: tp-v2-026, tp-v2-033, tp-v2-034, tp-v2-035, tp-v2-036, tp-v2-037, tp-v2-038, tp-v2-039, tp-v2-040, tp-v2-041, tp-v2-042, tp-v2-043, tp-v2-044, tp-v2-046, tp-v2-047, tp-v2-048, tp-v2-049, tp-v2-052, tp-v2-062, tp-v2-065, tp-v2-067, tp-v2-079, tp-v2-081, tp-v2-083, tp-v2-084, tp-v2-086, tp-v2-087, tp-v2-088, tp-v2-089, tp-v2-090, tp-v2-091, tp-v2-092, tp-v2-093, tp-v2-094, tp-v2-095, tp-v2-096, tp-v2-098
Classification: agent behavior unless the per-case record identifies a tool, fixture, dataset, or evaluator error.
Effect: lowers the named deterministic metric.
Recommendation: inspect representative traces before changing the agent or prompt.
