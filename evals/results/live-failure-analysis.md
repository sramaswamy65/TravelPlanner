# TravelPlanner baseline failure analysis

Mode: `live`  
Dataset: `memory-travel-agent-golden-v2`  
Cases: 12

## Aggregate metrics

| Metric | Score |
|---|---:|
| task_completion | 1.0 |
| required_tool_selection | 0.3333 |
| forbidden_tool_compliance | 0.9167 |
| tool_argument_correctness | 0.5278 |
| required_tool_order | 0.25 |
| maximum_tool_call_compliance | 0.6667 |
| must_include_assertions | 0.5833 |
| must_not_include_assertions | 0.9722 |
| memory_retrieval_recall | 0.5 |
| memory_retrieval_precision | 0.9167 |
| memory_application_accuracy | 0.5 |
| expected_memory_write_accuracy | 0.9167 |
| forbidden_memory_write_compliance | 1.0 |
| user_isolation_compliance | 1.0 |
| error_recovery_success | 0.6667 |
| median_latency_ms | 11073.38 |
| median_step_count | 2.5 |
| case_count | 12 |

## Top failure clusters

### 1. Tool selection and trajectory
- Classification: agent failure
- Frequency: 9 of 12 (75.0%)
- Cases: v2-01, v2-02, v2-03, v2-05, v2-06, v2-07, v2-08, v2-10, v2-12
- Effect: The agent may omit required evidence or request unnecessary/forbidden tools.
- LangSmith traces: https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddd-1eaf-79c0-91a7-dd01c841c1b5?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-1358-7ec2-8553-33af3204dc90?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-94ee-7f91-b2b2-492efcc4f0a7?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddf-4362-7780-9931-a6bea99de8f8?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddf-b6f2-7ac2-997f-3781573043d8?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-1b35-73d3-83ad-4cc0d9b9026f?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-75d3-7733-8c02-00dc71efc841?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de1-1646-7f41-9baf-a22ca6a03950?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de1-ca1a-7121-a3c3-c3643527a8b5?poll=true

### 2. Response completion and recovery
- Classification: agent failure
- Frequency: 7 of 12 (58.33%)
- Cases: v2-03, v2-04, v2-06, v2-07, v2-08, v2-09, v2-12
- Effect: The final response or write behavior may not satisfy the stated outcome.
- LangSmith traces: https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-94ee-7f91-b2b2-492efcc4f0a7?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-dd55-7451-b1ac-49767d06e072?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddf-b6f2-7ac2-997f-3781573043d8?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-1b35-73d3-83ad-4cc0d9b9026f?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-75d3-7733-8c02-00dc71efc841?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-cbcc-71d1-815a-19048705657c?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de1-ca1a-7121-a3c3-c3643527a8b5?poll=true

### 3. Memory retrieval and application
- Classification: memory failure
- Frequency: 6 of 12 (50.0%)
- Cases: v2-01, v2-02, v2-04, v2-05, v2-06, v2-07
- Effect: Saved preferences may be missed, over-retrieved, or not reflected in the response.
- LangSmith traces: https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddd-1eaf-79c0-91a7-dd01c841c1b5?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-1358-7ec2-8553-33af3204dc90?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06dde-dd55-7451-b1ac-49767d06e072?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddf-4362-7780-9931-a6bea99de8f8?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06ddf-b6f2-7ac2-997f-3781573043d8?poll=true, https://smith.langchain.com/o/f1a77bdf-664d-47f9-ad92-3a392dd31719/projects/p/1a92ab37-a6cd-450f-a5bd-df59ad54bb64/r/01a06de0-1b35-73d3-83ad-4cc0d9b9026f?poll=true

## Failure classification

- Agent failures: v2-01, v2-02, v2-03, v2-04, v2-05, v2-06, v2-07, v2-08, v2-09, v2-10, v2-12
- Tool failures or expected tool errors: v2-06, v2-07, v2-08
- Memory failures: v2-01, v2-02, v2-04, v2-05, v2-06, v2-07
- Evaluator failures: None
- Fixture problems: None detected by the baseline harness.

## Recommended improvements (not implemented)

1. Improve memory search query generation so durable pace, mobility, diet, and attraction preferences are retrieved by intent rather than literal token overlap.
2. Strengthen tool-selection instructions for direct attraction lookup and explicit `do not use memory` requests.
3. Stop planning after a required weather failure unless another tool is explicitly useful for a clearly labeled partial response.
4. Make policy-denied model tool requests visible as blocked attempts while keeping them out of the executed MCP-call count.
