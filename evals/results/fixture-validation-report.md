# Fixture validation report

Overall valid: **True**

```json
{
  "total_golden_cases": 100,
  "total_unique_fixture_ids": 46,
  "fixtures_implemented": 46,
  "fixtures_missing": [],
  "normal_records_required": 50,
  "normal_records_present": 50,
  "expected_failures_verified": 25,
  "expected_failure_invocations": 25,
  "schema_failures": 0,
  "cleanup_failures": 0,
  "network_call_violations": 0,
  "affected_case_ids": [],
  "fixture_types": {
    "normal data fixture": 16,
    "clarification-only fixture": 1,
    "unsupported-city fixture": 2,
    "invalid-date case": 2,
    "empty-result fixture": 2,
    "ambiguous-city fixture": 1,
    "closed-attraction fixture": 1,
    "unsupported-category fixture": 1,
    "timeout fixture": 2,
    "malformed-data fixture": 2,
    "provider-error fixture": 1,
    "approval-required fixture": 1,
    "unsafe-path fixture": 3,
    "existing-file fixture": 1,
    "missing-memory fixture": 1,
    "memory-provider failure fixture": 2,
    "repeated-call fixture": 1,
    "step-limit fixture": 1,
    "tracing-failure fixture": 1,
    "prompt-injection fixture": 2,
    "cross-user isolation fixture": 1,
    "secret-handling fixture": 1
  },
  "valid": true
}
```

No live providers were invoked; validation resolves local JSON, isolated SQLite memory, and temporary paths only.
