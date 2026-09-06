# TravelPlanner baseline failure analysis

Mode: `deterministic`  
Dataset: `memory-travel-agent-golden-v2`  
Cases: 12

## Aggregate metrics

| Metric | Score |
|---|---:|
| case_count | 12 |
| error_recovery_success | 1.0 |
| expected_memory_write_accuracy | 1.0 |
| forbidden_memory_write_compliance | 1.0 |
| forbidden_tool_compliance | 0.8333 |
| maximum_tool_call_compliance | 0.8333 |
| median_latency_ms | 2180.15 |
| median_step_count | 3.0 |
| memory_application_accuracy | 1.0 |
| memory_retrieval_precision | 1.0 |
| memory_retrieval_recall | 1.0 |
| must_include_assertions | 1.0 |
| must_not_include_assertions | 1.0 |
| required_tool_order | 0.8333 |
| required_tool_selection | 1.0 |
| task_completion | 1.0 |
| tool_argument_correctness | 1.0 |
| user_isolation_compliance | 1.0 |

## Top failure clusters

### 1. Tool selection and trajectory
- Classification: agent failure
- Frequency: 2 of 12 (16.67%)
- Cases: v2-08, v2-09
- Effect: The agent may omit required evidence or request unnecessary/forbidden tools.
- LangSmith traces: https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/de65816f-e742-4444-84d8-22346b6df6b5/r/01a06def-49f6-7f50-8bb5-8180a3324ce3?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/de65816f-e742-4444-84d8-22346b6df6b5/r/01a06dec-6a69-7d23-a481-bc28d93214a8?poll=true

### 2. Memory retrieval and application
- Classification: memory failure
- Frequency: 0 of 12 (0.0%)
- Cases: None
- Effect: Saved preferences may be missed, over-retrieved, or not reflected in the response.
- LangSmith traces: Unavailable

### 3. Response completion and recovery
- Classification: agent failure
- Frequency: 0 of 12 (0.0%)
- Cases: None
- Effect: The final response or write behavior may not satisfy the stated outcome.
- LangSmith traces: Unavailable

## Failure classification

- Agent failures: v2-08, v2-09
- Tool failures or expected tool errors: v2-06, v2-07, v2-08, v2-09
- Memory failures: None
- Evaluator failures: None
- Fixture problems: None detected by the baseline harness.

## Recommended improvements (not implemented)

1. Improve memory search query generation so durable pace, mobility, diet, and attraction preferences are retrieved by intent rather than literal token overlap.
2. Strengthen tool-selection instructions for direct attraction lookup and explicit `do not use memory` requests.
3. Stop planning after a required weather failure unless another tool is explicitly useful for a clearly labeled partial response.
4. Make policy-denied model tool requests visible as blocked attempts while keeping them out of the executed MCP-call count.
