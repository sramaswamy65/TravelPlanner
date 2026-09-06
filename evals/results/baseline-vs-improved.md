# Baseline vs improved

The same v2 golden labels, deterministic travel fixtures, evaluator code, model configuration, LangSmith trace structure, and sequential concurrency were used.

## Metric comparison

| Metric | Baseline | Improved | Absolute delta | Percentage delta |
|---|---:|---:|---:|---:|
| task_completion | 0.8333 | 1.0000 | +0.1667 | +20.00% |
| required_tool_selection | 0.9167 | 1.0000 | +0.0833 | +9.09% |
| forbidden_tool_compliance | 0.7500 | 0.8333 | +0.0833 | +11.11% |
| tool_argument_correctness | 0.9167 | 1.0000 | +0.0833 | +9.09% |
| required_tool_order | 0.6667 | 0.8333 | +0.1666 | +24.99% |
| maximum_tool_call_compliance | 0.6667 | 0.8333 | +0.1666 | +24.99% |
| must_include_assertions | 0.9167 | 1.0000 | +0.0833 | +9.09% |
| must_not_include_assertions | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| memory_retrieval_recall | 0.5833 | 1.0000 | +0.4167 | +71.44% |
| memory_retrieval_precision | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| memory_application_accuracy | 0.5833 | 1.0000 | +0.4167 | +71.44% |
| expected_memory_write_accuracy | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| forbidden_memory_write_compliance | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| user_isolation_compliance | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| error_recovery_success | 1.0000 | 1.0000 | +0.0000 | +0.00% |
| median_latency_ms | 3423.5600 | 2180.1500 | -1243.4100 | -36.32% |
| median_step_count | 4.0000 | 3.0000 | -1.0000 | -25.00% |

## Latency and tool-call tradeoffs

- Median runtime: 3423.56 ms to 2180.15 ms.
- Median executed/blocked tool records per case: 3.0 to 2.0.
- Total tool records: 26 to 23.
- Required-error cases now stop after the failing weather call instead of making an unnecessary attraction call.

## Case movement

- Improved cases: v2-01, v2-02, v2-04, v2-05, v2-06, v2-07, v2-10, v2-11
- Regressed cases: None
- Cases with remaining failures: v2-08 (forbidden_tool_compliance, maximum_tool_call_compliance, required_tool_order); v2-09 (forbidden_tool_compliance, maximum_tool_call_compliance, required_tool_order)

## Representative LangSmith traces

- **Memory retrieval and application**, `v2-01`: [baseline](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/48cfdebd-c24c-4401-ab55-e10284c5c0b6/r/01a06dda-e4e8-7b80-9597-587d4a3a61e7?poll=true) / [improved](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/de65816f-e742-4444-84d8-22346b6df6b5/r/01a06dec-0658-7582-b78f-6add37aaa5ef?poll=true)
- **Tool selection and trajectory**, `v2-04`: [baseline](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/48cfdebd-c24c-4401-ab55-e10284c5c0b6/r/01a06ddc-4193-7933-8eb6-72ae35d0439b?poll=true) / [improved](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/de65816f-e742-4444-84d8-22346b6df6b5/r/01a06def-12cd-7050-8567-41fb6ea41d2a?poll=true)
- **Response completion and recovery**, `v2-04`: [baseline](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/48cfdebd-c24c-4401-ab55-e10284c5c0b6/r/01a06ddc-4193-7933-8eb6-72ae35d0439b?poll=true) / [improved](https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/de65816f-e742-4444-84d8-22346b6df6b5/r/01a06def-12cd-7050-8567-41fb6ea41d2a?poll=true)

## Findings

### What worked

- Broad user-scoped preference search recovered the seeded durable memories.
- Clear intent ordering removed the accidental write before memory listing.
- Explicit no-memory handling and direct attraction lookup corrected their tool paths.
- Required weather failures now stop without an unnecessary dependent call.

### What did not work

Suppressing prohibited write attempts caused the unchanged evaluator to lose the required `policy_denied` observation. That attempted change was reverted; the existing agent policy still prevents the MCP write, but those blocked attempts remain trajectory failures in `v2-08` and `v2-09`.

### Top remaining failure

The two prohibited-memory cases still record the model's blocked `save_traveler_memory` request, reducing forbidden-tool, order, and call-bound scores.

### What to try next

Represent policy-denied attempted calls separately from executed MCP calls in a future versioned trajectory schema, then version the evaluator and labels together. Do not change the current v2 evaluator retroactively.
