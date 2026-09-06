# TravelPlanner Loom demonstration script (5-7 minutes)

## 0:00-0:35 - Goal

Open the Streamlit application. Explain that TravelPlanner creates evidence-grounded travel-day plans while demonstrating one agent, MCP tools, durable user-scoped memory, and observable evaluation. It does not book travel or use live travel data; weather and attractions are deterministic fixtures.

## 0:35-1:10 - One-agent architecture and Nebius

Show the README architecture diagram and `src/agent.py`. Point out that exactly one TravelPlanner agent owns the bounded reason-act-observe loop. Show `.env.example`, naming the OpenAI-compatible Nebius endpoint and `NEBIUS_MODEL`; do not reveal `.env` or any key. Explain that the live model selects tools and that deterministic mode substitutes a repeatable model fixture for evaluation.

## 1:10-1:50 - MCP tool call

Submit: `Find museums in Chicago`. Open Agent Activity and show `find_attractions`, its arguments, MCP result status, and fixture-backed museum evidence. Explain that the agent discovers schemas through the official MCP client over stdio and does not import tool functions directly.

## 1:50-2:40 - Mem0 across sessions

In live mode, submit: `Remember that I prefer vegetarian meals`. Show the explicit write request and successful memory result. Start a new chat session with the same user ID and request a Chicago plan. Show `search_traveler_memory` and the disclosed saved preference. Briefly switch to another user ID and show that the memory is absent. Delete the demo memory afterward using the confirmed deletion flow.

## 2:40-3:20 - LangSmith trace

Open one post-improvement trace linked from `evals/results/baseline-vs-improved.md`. Show one parent `travel_planner_run` and its model, MCP, memory, and final-response children. Highlight `case_id`, `dataset_version=v2`, tool-call IDs, and redacted trace fields.

## 3:20-4:00 - Golden dataset

Open `evals/golden_dataset.jsonl` or the downloadable CSV. State the validated composition: 12 unique cases, 6 happy path, 3 edge, 2 known failure, 1 adversarial; 11 memory cases with isolated user IDs and deterministic seeds. Show that the LangSmith dataset is `memory-travel-agent-golden-v2` and was reused rather than duplicated.

## 4:00-4:40 - Baseline

Open `evals/results/baseline-summary.json`. Call out task completion at 83.33%, memory recall/application at 58.33%, tool order and maximum-call compliance at 66.67%, and median runtime at 3,423.56 ms. Show the linked trace for `v2-01`: the generic preference query returned no memory. Then show `v2-04`: a write was attempted before listing memory.

## 4:40-5:35 - Evidence-supported improvements

Show `evals/results/improvement-plan.md`. Summarize four changes: defined broad preference-query semantics with unsafe-result filtering; clarified list/write and no-memory intent; supported direct attraction lookup; and stopped dependent calls after required evidence failure. State that labels, fixtures, evaluators, model configuration, trace structure, and concurrency did not change.

## 5:35-6:20 - Delta and limitation

Open `evals/results/baseline-vs-improved.md`. Report task completion 83.33% to 100%, memory recall/application 58.33% to 100%, tool order/max-call compliance 66.67% to 83.33%, median steps 4 to 3, and median runtime down 36.32%. Eight cases improved and none regressed.

Close with the remaining limitation: `v2-08` and `v2-09` still contain blocked `save_traveler_memory` attempts. The agent prevents the actual MCP write, but the unchanged v2 trajectory evaluator correctly counts the attempted forbidden tool. The next version should distinguish attempted, policy-blocked, and executed calls through a jointly versioned schema and evaluator rather than rewriting the current golden labels.
