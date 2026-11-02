# Mizan QA strategy and adversarial test report

## Scope and requirement traceability

The executable implementation and README are the source of truth because no separate SRS was supplied.

| Requirement | Source | Acceptance condition |
|---|---|---|
| RQ-01 | README: HTTP API | Search returns ranked passages with source metadata. |
| RQ-02 | README: generation | Ask produces a grounded answer with numbered citations. |
| RQ-03 | README: streaming | Sources arrive before answer deltas; failures become SSE errors. |
| RQ-04 | README: Arabic | Arabic and mixed-direction queries remain valid UTF-8 and searchable. |
| RQ-05 | README: source filters | A source filter restricts every returned passage. |
| RQ-06 | README: security | HTTP ingestion is disabled unless explicitly enabled behind authentication. |
| RQ-07 | Web implementation | UI exposes loading, empty, error, partial and success states. |
| RQ-08 | PWA implementation | Manifest, icon and offline shell are available without a CDN. |
| RQ-09 | Provider architecture | Provider credentials/models are isolated and failures are actionable. |
| RQ-10 | Index/store architecture | Concurrent reads do not mutate or corrupt ranked results. |

## Strategy

- Unit: Arabic normalization, chunking, citations, provider adapters and retry classification.
- Integration: ingest temporary documents, build a real store with a fake embedder, then retrieve and filter.
- API: validate schemas, status codes, SSE ordering, MIME types, security defaults and malformed bodies.
- End-to-end: submit an Arabic question, observe loading, sources, streamed answer, citations and recovery.
- Regression: offline assets, provider defaults, local environment isolation, article boundaries and evaluation questions.
- Negative/adversarial: whitespace, oversized history, unknown fields, invalid ranges, missing credentials and disabled ingestion.
- Performance/concurrency: warm-start latency, parallel reads, maximum result limits and long streams.

## Important test cases

| ID | Scenario / trace | Preconditions | Steps and input | Expected | Priority |
|---|---|---|---|---|---|
| QA-API-001 | Ranked search / RQ-01 | Two indexed fixtures | POST /search, annual leave, top_k 2 | 200; relevant source first | P0 |
| QA-API-002 | Invalid boundaries / RQ-01 | API running | Send top_k 0, -1, 51, float and string | 422; no retrieval | P0 |
| QA-API-003 | Whitespace query / RQ-01 | API running | Send three spaces | 422 | P1; XFAIL |
| QA-API-004 | Unknown JSON fields | API running | Add admin true | 422; fail closed | P1; XFAIL |
| QA-API-005 | Source isolation / RQ-05 | Multiple sources | Filter to penal | All results match filter | P0 |
| QA-API-006 | Ingestion security / RQ-06 | Default environment | POST /ingest with current directory | 403; no filesystem read | P0 |
| QA-API-007 | Unbounded history / RQ-02 | API running | Send 1,001 turns | 422 before provider call | P0; XFAIL |
| QA-SSE-001 | Event order / RQ-03 | No provider credentials | Ask via stream | sources precedes actionable error | P0 |
| QA-LLM-001 | OpenRouter completion / RQ-09 | Fake transport | Complete system and user messages | Bearer auth and usage parsed | P0 |
| QA-LLM-002 | OpenRouter SSE / RQ-03 | Fake SSE transport | Metadata, deltas, DONE | Exact text; stop at DONE | P0 |
| QA-CON-001 | Concurrent reads / RQ-10 | Warm fixture index | 24 searches, 8 workers | Deterministic; no errors | P1 |
| QA-PWA-001 | Offline assets / RQ-08 | Test client | Fetch manifest, icon, worker | Correct status and metadata | P1 |
| QA-PWA-002 | Worker update safety / RQ-08 | Test client | Fetch service worker | no-cache and root scope | P1 |
| QA-UI-001 | State coverage / RQ-07 | Root page | Inspect shell | Loading, empty, error, success present | P1 |
| QA-E2E-001 | Arabic answer / RQ-02, RQ-04 | Index and provider | Ask annual-leave question | Arabic answer and valid citations | P0 |
| QA-NET-001 | Provider unavailable / RQ-03 | Provider blocked | Ask question | Sources remain; UI unlocks | P0 |
| QA-NET-002 | In-band provider failure / RQ-03 | HTTP 200 SSE error | Stream error then DONE | Actionable RuntimeError | P0; XFAIL |

## Acceptance workflows

1. Search-only: open app, type Arabic query, search, inspect evidence, expand passage.
2. Grounded answer: ask, render sources first, stream answer, click citation, focus matching evidence.
3. No provider: retain sources, show clear error, keep search-only available.
4. No matches: use impossible source filter, show empty state and recovery advice.
5. Offline shell: revisit without network, load branding, clearly report unavailable live API.
6. Security: attempt ingestion under default configuration and deny it.

## Findings and risk order

1. **P0 - request resource exhaustion:** AskRequest history has no count, size or role constraints. A client can create an enormous provider prompt and unexpected spend. Add request-size limits and bounded typed history.
2. **P1 - weak validation:** whitespace-only questions pass validation and unknown JSON properties are ignored. Strip input and configure Pydantic to forbid extras.
3. **P1 - authentication boundary:** read endpoints are public and ingestion uses an environment switch, not authentication. Do not expose publicly; add roles before enabling ingestion.
4. **P1 - startup availability:** startup can block on Hugging Face while the port stays closed. Bind early, expose readiness, use local-only model policy and degrade to BM25.
5. **P1 - test isolation:** local provider/model overrides leaked into contract tests. The new autouse fixture isolates them.
6. **P2 - PWA semantics:** the shell loads offline but legal APIs cannot. Add an explicit offline banner.
7. **P2 - dependency drift:** this Python environment lacks the declared anthropic package, so Claude tests cannot execute.

## Live acceptance run

Run on 2 November 2026 against the local 3,830-passage index and OpenRouter:

| Check | Observed |
|---|---|
| Index state | 12 documents, 3,830 passages |
| Search-only UI | 10 passages in 23 ms |
| Stream source event | First and before all deltas |
| Sources available | 45 ms in direct API measurement |
| First streamed token | 980 ms in direct API measurement |
| Complete browser answer | 3,556 ms |
| Citation navigation | One citation button linked to evidence 9 |
| Provider | openrouter, openai/gpt-4o-mini |

The live question asked who qualifies as a trader under Jordanian commercial law. Retrieval ranked Article 42 first and Article 8 ninth; it did not retrieve the complete definition provision near the top. The model then treated Article 8 as the definition. The citation was mechanically valid but substantively incomplete. Acceptance must therefore measure whether the cited passage entails the answer, not merely whether a citation marker exists.

The browser also retained old evidence while a new request was loading. The question field could display a constitutional question while commercial-law evidence remained visible and both buttons were disabled. Clear or visibly mark prior results as stale at request start.

Fix unbounded request history and startup blocking first because they affect cost exposure and availability. Next, enforce citation entailment/retrieval quality and clear stale evidence before public deployment.
