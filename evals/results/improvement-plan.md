# Evidence-supported improvement plan

The changes below were selected from `failure-analysis.md` and the linked baseline LangSmith traces. Golden labels, fixture data, evaluators, model identifiers, trace structure, and sequential execution remain unchanged.

| Failure cluster | Proposed change | Predicted metric impact | Possible negative tradeoff |
|---|---|---|---|
| Memory retrieval and application (`v2-01`, `02`, `05`, `06`, `07`) | Define the documented generic `travel preferences` query as a request for all user-scoped durable preferences in the deterministic memory adapter; clarify that query in the MCP description and prompt. | Recall and application from 58.33% toward 100%; precision and isolation unchanged. | Broad retrieval can add irrelevant preferences when a user has many memories. |
| Tool selection: memory intent (`v2-04`, `08`, `09`, `10`) | Distinguish list intent from write intent, suppress tool calls for deterministically prohibited temporary/sensitive writes, and honor explicit `do not use memory`. | Tool order/max-call/forbidden-tool compliance and task completion improve. | Conservative suppression may reject an unusual but safe preference unless the user restates it clearly. |
| Direct attraction lookup (`v2-11`) | Clarify direct lookup behavior and make the deterministic model fixture select `find_attractions` for find/museum requests, then ground its response in the returned category. | Required tool selection, order, arguments, and must-include assertions improve. | Keyword-oriented offline behavior is less flexible than the live model.
| Partial tool failure (`v2-06`, `07`) | Tell the model to stop dependent planning after required weather evidence fails and disclose the limitation. | Recovery should remain 100% while avoiding unnecessary calls and latency. | Independent attractions might sometimes still be useful in a partial response. |
