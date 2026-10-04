# Test coverage

```bash
python -m pytest --cov --cov-report=term-missing
```

**94% line coverage, 751 tests**, measured across `app/`, `ml/src/` and
`scripts/`. 40 of 59 measured files are at 100%.

`scripts/demo.py` is omitted: it is an end-to-end smoke test that drives a
running server and exits non-zero on failure. Measuring it under pytest would
report 0% for a script that is verified by being run.

Coverage here is a map of what the tests reach, read by a person deciding where
to look next. It is deliberately **not** a gate: a threshold in CI mostly
produces tests written to satisfy the threshold, and the critical paths below
are covered on purpose rather than incidentally.

## Critical paths

Each of these is covered by tests that assert the *behaviour*, not just that
the line ran.

| Area | Where | Notable |
|---|---|---|
| Anonymous submission | `test_reports_api.py` | identity fields rejected, not ignored; description limits; NUL bytes |
| Case codes | `test_case_codes.py` | 100 bits of entropy asserted arithmetically; `random` unreachable |
| Case tracking | `test_cases_api.py` | malformed and unknown codes byte-identical |
| Authentication | `test_auth_api.py`, `test_tokens.py` | `alg:none`, algorithm confusion, tampered payloads, timing equalisation |
| Authorization | `test_auth_dependency.py` | 12 malformed-header variants; deactivation effective next request |
| Moderation | `test_moderation_*.py` | every forbidden transition, derived from the map rather than listed |
| Concurrency | `test_moderation_concurrency.py` | real threads on real connections; **verified to fail without the row lock** |
| Rate limiting | `test_rate_limiting.py` | per-endpoint isolation, `Retry-After`, salted fingerprints |
| Security | `test_security_headers.py`, `test_error_hardening.py` | header presence on errors; AST scan for dynamic SQL |
| Logging privacy | `test_logging_privacy.py` | real traffic driven, captured output searched for every secret |
| ML failure | `test_triage_integration.py` | 5 failure modes, each proving the report survives |
| Reporter privacy | `test_privacy.py`, `test_triage_integration.py` | structural, not just "the response happened not to contain it" |

## Where coverage is lower, and why

| File | Cover | What is uncovered |
|---|---:|---|
| `scripts/create_moderator.py` | 66% | The interactive `main()` — terminal prompts and engine construction. Its pieces (`validate_username`, `create`, `prompt_for_password`, exit codes) are tested directly; driving a `getpass` loop through a subprocess would test the terminal, not the code. |
| `app/db/session.py` | 57% | The cached engine/sessionmaker singletons. Tests inject their own session by design, so the production factory is exercised by the live server and the Docker image rather than by unit tests. |
| `app/ml/inference.py` | 88% | Defensive branches for artifact shapes that the validated loader already rejects. |
| `ml/src/train_category.py` | 90% | CLI exit paths for a missing or untrainable dataset. |
| `ml/src/text.py` | 75% | One line: the non-string guard in `normalise_text`. |

None of these is a critical path. Each is a deliberate decision, not an
oversight.

## What coverage does not tell you

A line being executed says nothing about whether the assertion around it was
meaningful. Two examples from this repository where that mattered:

- the **concurrency** tests were confirmed by temporarily removing the row lock
  and watching them fail — without that step they would have been 100% covered
  and proved nothing;
- the **`hide_parameters`** logging fix was found by demonstrating the leak
  first, then fixing it, rather than by adding a line to a covered file.
