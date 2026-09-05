# Autonomous Test Orchestration Agent

Give it a URL. It crawls the application, plans what to test, **gates its own
coverage before writing any code**, ranks every flow by business risk, generates
Playwright tests against selectors it proved exist, runs them, decides whether
each failure is a broken script or a real bug, diffs screenshots, files
ticket-ready defects, and hands back a risk-ordered report.

No human step in between. No hand-written test anywhere in the repository.

```bash
curl -X POST localhost:8000/run -H 'Content-Type: application/json' \
     -d '{"url": "https://books.toscrape.com/"}'
```

---

## Contents

1. [What makes this an agent](#what-makes-this-an-agent)
2. [Setup](#setup)
3. [Running it](#running-it)
4. [Architecture](#architecture)
5. [The rubrics](#the-rubrics)
6. [Credentials and security](#credentials-and-security)
7. [Rate limits and model routing](#rate-limits-and-model-routing)
8. [Feature status: implemented / scaffolded / roadmap](#feature-status)
9. [Known limitations](#known-limitations)
10. [Evaluating agent quality over time](#evaluating-agent-quality-over-time)
11. [Repository map](#repository-map)

---

## What makes this an agent

Plenty of tools generate tests. The gap this fills is **coordination**: deciding
when to plan, when to re-plan, when to heal, and when to escalate, without a
human directing each step. Four decisions carry that weight, and all four are
visible in the live decision log:

| Decision | Where | What it does |
|---|---|---|
| **Is this plan worth generating from?** | Coverage gate, *before* the Generator | Applies a six-line rubric. On failure it sends the Planner *specific* feedback ("add a flow that submits the login form with a wrong password and asserts the inline error text") and re-plans. Budget: 2. |
| **Re-plan, proceed, or escalate?** | Orchestrator | Every route carries a confidence and a rationale. When the budget is spent it force-proceeds and stamps `force_proceeded`, rather than looping or giving up. |
| **What matters here?** | Risk ranking, right after the plan is approved | HIGH / MEDIUM / LOW per flow, cited to the rubric. That order then drives generation, execution, healing priority and the report. |
| **Broken test or broken app?** | Healer | Classifies each failure, scores confidence from evidence it gathers itself, and takes a genuinely different branch at 0.60. |

Three things are worth pointing at specifically.

**The coverage gate can actually fail, and the model cannot talk it out of
failing.** A deterministic rubric check runs alongside the LLM judge and can
only make the gate *stricter*, never more permissive. If the routing model
answers "proceed" on a failed gate while re-plan budget remains, that answer is
coerced to "replan" in code ([`agents/orchestrator.py`](agents/orchestrator.py)).
A gate a model can wave through is decorative.

**The low-confidence healer path is a different branch, not a different log
level.** Below 0.60 no patch is applied; the finding is queued for human review
with its evidence, and appears in the UI and the report as such. And no patch,
at any confidence, is allowed to weaken an assertion — `patch_weakens_assertions`
rejects any change that removes an `expect(...)`, downgrades a text assertion to
a visibility check, or comments an assertion out. Making a red test green by
lowering the bar is the one failure mode that would make this system worse than
useless, so it is blocked mechanically rather than by prompting.

**Confidence is not just the model's opinion.** The model's self-reported number
is blended 40/60 with a score computed from evidence this process gathered:
whether the failing locator still resolves when re-probed live, whether the
failure was a timeout or an assertion mismatch, whether a captcha is on screen.
The evidence half carries the larger weight, so a confident-sounding rationale
cannot on its own push a patch into a page that is showing a bot wall.

---

## Setup

**Requirements:** Python 3.11 or 3.12 (Playwright wheels lag on 3.13+), and
about 400 MB for the Chromium bundle.

```bash
git clone <your-fork> && cd <repo>

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium

cp .env.example .env        # Windows: copy .env.example .env
```

Then put a **free** Groq key in `.env`:

```ini
GROQ_API_KEY=gsk_...
```

Get one at [console.groq.com](https://console.groq.com) — no credit card, no
trial period. That is the only credential the agent needs to operate.

Optional extras:

```bash
pip install -r requirements-optional.txt   # ChromaDB, for ENABLE_PRD_GAP_ANALYSIS
```

**No key yet?** `LLM_OFFLINE_MODE=true` runs the whole pipeline with a
deterministic stub instead of a model. It exercises the graph, the browser
stack, the API and the UI end to end — including the re-plan loop — so you can
verify the plumbing. Every report produced this way is stamped
`llm_provider: offline-stub` and carries a loud limitation entry. **It is not
the agent**, and it is not a demo mode; it is a smoke test.

---

## Running it

### API + UI (the demo path)

Two terminals:

```bash
python -m uvicorn api.app:app --reload --port 8000
```

```bash
streamlit run ui/streamlit_app.py
```

Open the Streamlit page, enter a URL, press **Start autonomous run**, and watch
the decision log fill in. It polls `GET /run/{id}/status` every 1.5 s and
renders each event as the agent emits it — stage, summary, confidence, risk,
re-plan flags, and a `NEEDS HUMAN REVIEW` badge where one applies. Lines you
will see:

```
🧠 planner        — Planner found 12 flows (8 happy, 3 edge, 1 error)
🔁 coverage_gate  — Coverage gate: missing auth error-state — sending back to Planner (replan 1/2)
🧠 risk_ranking   — Risk: Complete checkout = HIGH        🔴 HIGH   conf 0.90
🧠 generator      — F003: 7/7 selectors resolved against the live DOM
🧠 healer         — Healer: 0.41 confidence — NOT auto-applied, queued for human review (F007)
🧠 visual_diff    — Visual diff: 12.4% pixels changed on /cart vs baseline
🧠 bug_packager   — Bug packaged: BUG-003 with repro + screenshot
```

### CLI (no servers)

```bash
python cli.py https://books.toscrape.com/
python cli.py https://example.com --intent "focus on checkout and authentication"
python cli.py https://app.example.com --username qa@example.com --password-env QA_PW
python cli.py https://example.com --offline --headed --max-pages 6
```

### Against any URL

Public sites need nothing but the URL. For an application behind a login:

```bash
curl -X POST localhost:8000/run -H 'Content-Type: application/json' -d '{
  "url": "https://app.example.com",
  "intent": "focus on checkout and authentication flows",
  "credentials": {"username": "qa@example.com", "password": "...", "login_url": "https://app.example.com/login"}
}'
```

> ⚠️ **Use a throwaway test account, and point this at staging.** Selector
> validation is *live*: the Generator clicks buttons and fills forms on the real
> application to prove each locator exists before writing it into a test. That is
> what makes the generated tests trustworthy, and it means the agent will create
> whatever a real user would create by clicking around.

### The generated tests are yours

Every run writes a standalone suite to
`reports/runs/<run_id>/generated_tests/`, complete with a `conftest.py` and a
`pytest.ini` the agent also wrote. It runs without the agent, without an API
key, and without this repository:

```bash
cd reports/runs/<run_id>/generated_tests
pip install pytest pytest-asyncio playwright && playwright install chromium
pytest -q
```

Credentials come from `LOGIN_USERNAME` / `LOGIN_PASSWORD` in the environment, or
a `STORAGE_STATE` file. They are never in the test files.

### API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/run` | Start a run. `{url, intent?, prd_text?, credentials?}` → `{run_id}` |
| `GET` | `/run/{id}/status` | Stage, counts, and the **full decision log so far**. Never any credential. |
| `GET` | `/run/{id}/report` | Final JSON report, risk-ordered |
| `GET` | `/run/{id}/report.md` · `/report.html` | Rendered forms |
| `GET` | `/run/{id}/bugs` · `/bugs/{bug_id}` | Packaged defects: ticket, repro script, screenshot |
| `GET` | `/run/{id}/events.jsonl` | Raw event stream |
| `DELETE` | `/run/{id}` | Cancel; wipes credentials immediately |
| `GET` | `/health` · `/runs` | Readiness and run listing |

---

## Architecture

### The pipeline

```mermaid
flowchart TD
    IN(["URL (required)<br/>+ credentials? + PRD? + intent?"]) --> PL

    subgraph AGENTS["Sub-agents"]
        direction TB
        PL["<b>Planner</b><br/>login first, then BFS crawl<br/>→ structured test plan"]
        GEN["<b>Generator</b><br/>live selector validation<br/>→ Playwright test files"]
        HEAL["<b>Healer</b><br/>SCRIPT_ISSUE vs GENUINE_DEFECT<br/>+ confidence"]
    end

    subgraph META["Meta-orchestrator"]
        direction TB
        GATE{"<b>Coverage gate</b><br/>rubric C1–C6<br/><i>runs BEFORE generation</i>"}
        ROUTE{"replan / proceed / escalate<br/><i>confidence + rationale</i>"}
    end

    subgraph DIFF["Differentiation layer"]
        direction TB
        RISK["<b>Risk ranking</b><br/>HIGH / MED / LOW, rubric-cited"]
        VIS["<b>Visual diff</b><br/>Playwright + Pillow vs baseline"]
        BUG["<b>Bug packager</b><br/>repro.py + screenshot + ticket.md"]
    end

    PL --> GATE
    GATE -->|"gaps found"| ROUTE
    ROUTE -->|"replan (budget 1–2/2)<br/>with specific feedback"| PL
    ROUTE -->|"budget spent → force_proceeded"| RISK
    GATE -->|"passed"| RISK

    RISK --> GEN --> RUN["<b>Runner</b><br/>sequential, or parallel behind a flag<br/>screenshot + DOM + console per test"]
    RUN -->|"failures"| HEAL
    RUN -->|"all green"| VIS
    HEAL -->|"SCRIPT_ISSUE ≥ 0.60<br/>patch locator/wait"| RUN
    HEAL -->|"< 0.60 → NOT applied,<br/>queued for human review"| VIS
    HEAL -->|"GENUINE_DEFECT<br/>(assertions never weakened)"| BUG
    VIS --> BUG --> REP["<b>Report</b><br/>ordered by risk"]
    REP --> OUT(["report.json · report.md · report.html<br/>bugs/BUG-00N/ · generated_tests/"])

    PL -.-> LOG
    GATE -.-> LOG
    ROUTE -.-> LOG
    RISK -.-> LOG
    GEN -.-> LOG
    RUN -.-> LOG
    HEAL -.-> LOG
    VIS -.-> LOG
    BUG -.-> LOG
    REP -.-> LOG

    LOG[["<b>decision_log</b> — every node emits on start / decide / finish<br/>→ events.jsonl · GET /run/id/status · live Streamlit view"]]

    style META fill:#fff4e6,stroke:#f59e0b,stroke-width:2px
    style DIFF fill:#eef6ff,stroke:#3b82f6,stroke-width:2px
    style AGENTS fill:#f0fdf4,stroke:#22c55e,stroke-width:2px
    style LOG fill:#fdf2f8,stroke:#ec4899,stroke-width:2px
```

### Why LangGraph, and why these edges

The graph is nodes and **conditional edges**, not a prompt chain, because three
of the transitions are real decisions with real budgets:

```
planner ─▶ coverage_gate ─┬─(gaps, budget left)──▶ planner        # re-plan loop
                          ├─(passed)─────────────▶ risk_ranking
                          └─(budget spent)───────▶ risk_ranking   # force_proceeded

risk_ranking ─▶ generator ─▶ runner ─┬─(failures, first pass)─▶ healer
                                     └─(clean / already healed)▶ visual_diff

healer ─┬─(patch auto-applied)─▶ runner        # re-run once, capped
        └─(nothing to re-run)──▶ visual_diff

visual_diff ─▶ bug_packager ─▶ report ─▶ END
```

Two invariants keep the loops safe: `replan_count` is capped at 2 and
`heal_pass_count` at 1. Both caps are enforced in the edge functions, not left
to a model's judgment.

Every node is wrapped by a decorator that emits `start` / `complete` events,
keeps the live progress snapshot current, and converts an exception into a
recorded error plus a routing hint — so a node failure degrades the run into a
**partial report** instead of killing the process.

### The five design decisions worth defending

**1. Plans describe targets in words; selectors are resolved against the live
DOM.** A plan written against imagined CSS is worthless. The Planner writes
"the Add to Basket button"; `browser/selectors.py` then proposes up to eight
locator strategies — drawn from the element inventory the crawler actually
captured, not from guesses — probes each against the live page, and keeps the
one that resolves to exactly one visible element. A step whose target cannot be
resolved is reported as a coverage gap rather than turned into a test that only
pretends to exercise the flow.

**2. Model-authored code, mechanically gated.** The Generator's output is
Python this process then executes, so it passes an AST audit
(`browser/sandbox.py`) before it is compiled: an import whitelist, a forbidden-name
list, a dunder-attribute block, and a required `async def test_flow(page, ctx)`
signature. Invalid output earns exactly one repair round-trip. If it still
fails, a **deterministic compiler** renders the same validated steps and
locators into source, and the report records `generated_by_model` as the
fallback so nobody mistakes it for model output. The fallback is a compiler over
agent-produced artifacts — the agent's plan, the agent's live-resolved selectors
— not a hand-written test.

**3. Two confidences, blended, with the evidence weighted higher.** Described
above. `differentiation/confidence_scorer.py` is pure and unit-tested, which is
what makes the 0.60 threshold something provable rather than hoped-for.

**4. Credentials never enter the state.** LangGraph state carries
`credentials_present: bool`. The values live in an in-process `SecretBox` keyed
by run id and are wiped in a `finally` block. See below.

**5. The report is ordered by risk.** Primary key risk band, secondary key
outcome severity (failure → review queue → healed → visual regression → pass),
tertiary the original index so two runs over the same data are byte-identical.
A report sorted by flow index tells a manager nothing.

---

## The rubrics

All three live as constants in [`llm/prompts.py`](llm/prompts.py) and are
interpolated into both the prompt and the deterministic fallback logic, so the
judge and the fallback cannot drift apart.

### Coverage (the gate fails if any applicable line is unmet)

| | Requirement |
|---|---|
| **C1** | At least one happy path for each primary discovered area |
| **C2** | At least one edge case (empty input, boundary value, optional field omitted) |
| **C3** | At least one error state (invalid input, failed submit, 404 / empty state) |
| **C4** | If a login UI exists: an auth happy path **and** an invalid-credential error state |
| **C5** | If cart / checkout / payment was discovered: flows covering it |
| **C6** | Every flow states a concrete expected outcome — not a bare sequence of clicks |

C4 and C5 are marked "not applicable" when the crawl found no such surface.

### Risk

| Band | Covers |
|---|---|
| **HIGH** | checkout, payment, cart persistence, authentication, signup, password reset, PII, destructive actions (delete account/order) |
| **MEDIUM** | search, product detail, filters, profile update, any non-payment form submit |
| **LOW** | footer, about, blog, cosmetic nav, static content, theme toggle |
| **→ MEDIUM** | anything unknown or ambiguous |

The rationale must cite the line that drove it.

### Confidence (Healer and Orchestrator)

Start at 0.5, then adjust:

- **Increase** — the original selector still resolves in the captured DOM (+0.18); an unambiguous timeout (+0.12) or assertion mismatch (+0.10); the error names the specific locator (+0.08); the failure reproduced twice (+0.15); the page logged JS errors (+0.06).
- **Decrease** — SPA render race (−0.20); captcha or bot wall (−0.30); flaky network or 5xx (−0.18); ambiguous or templated expected text (−0.15); the failure happened behind a login wall (−0.22).

Output is a number in [0,1] plus a one-sentence rationale. **Strictly below 0.60
is never auto-applied.**

---

## Credentials and security

The rule: a credential exists in memory, for the duration of one run, and
nowhere else.

| Guarantee | How |
|---|---|
| Never in the graph state | State carries `credentials_present: bool`. Values live in `SecretBox`, keyed by run id, wiped in `finally`. |
| Never in a log line | A `logging.Filter` on the root logger redacts the *rendered* message and any exception text, so a module that forgets to redact still cannot leak. |
| Never in an API response | Request and response models are separate classes. `POST /run` is the only endpoint that accepts credentials; every outbound payload passes through `redact_secrets()`. |
| Never in a generated test | Tests call `ctx["secret"]("password")`, resolved at runtime. Rendered files are scanned for credential literals before being written — model output that embedded one is **discarded** and the deterministic compiler is used instead. |
| Never in a screenshot | Password-type and sensitive-named fields are handed to Playwright's native `mask=`. If a sensitive field holds a value and masking is unavailable, the frame is **skipped** rather than taken unmasked. |
| Never in a bug artifact | Repro scripts read `os.environ` and print a message when it is unset. |
| Never in git | `.env`, `.env.*` (except `.env.example`), and `storage_state*.json` are ignored. |
| Never in a URL | `sanitize_url()` strips `user:pass@` userinfo and redacts sensitive query parameters everywhere a URL is displayed. |

`redact_secrets()` walks any JSON-ish structure and replaces (a) registered
secret values, (b) values under sensitive key names, and (c) known credential
patterns — bearer tokens, `gsk_`/`sk-` keys, `https://user:pass@host` — with
`***REDACTED***`. Values shorter than four characters are not registered for
substring matching, because registering a two-character password would punch
holes through unrelated prose; they are still protected by key-name redaction
and by never being serialised.

After login succeeds, the session travels as a Playwright `storage_state` file
so that every later page and every generated test inherits the session **without
ever seeing the password**. That file is written outside the reports tree, is
git-ignored, and is unlinked when the run ends.

Verified end to end: a run submitted with credentials was polled 21 times, and
the values appeared in no status payload, no report (JSON, Markdown or HTML), no
`events.jsonl`, and nowhere on disk under `reports/`.

**Two honest caveats.** Python strings are immutable, so `wipe()` drops the
reference rather than scrubbing the heap — that is documented in the code, not
dressed up as secure erasure. And the AST sandbox is defence in depth, not a
security boundary: run this against applications you are authorised to test, and
use a container if the target is not yours.

---

## Rate limits and model routing

Groq's free tier is generous but finite, so work is split by *kind*, not by
convenience:

| Role | Model | Used for |
|---|---|---|
| **Reasoning** | `openai/gpt-oss-120b` | Coverage evaluation, risk ranking, defect classification, confidence, orchestrator routing, report synthesis |
| **Codegen** | `openai/gpt-oss-20b` | Plan → Playwright translation, ticket prose |

`qwen/qwen3.6-27b` is a drop-in alternative for reasoning via `MODEL_REASONING`.
**Never route codegen to the larger reasoning model** — it is mechanical translation, and burning
the reasoning budget on it is what makes a demo die halfway through.

Handling:

- Exponential backoff with full jitter on 429 and 5xx, honouring `Retry-After`.
- A per-role minimum interval (1.2 s reasoning, 0.35 s codegen) as a floor on spacing.
- `response_format: json_object` where supported, with an automatic retry without it on a 400.
- Auth failures (401/403) are never retried — they are a configuration error, not a transient one.

**Adding a fallback provider is a registration, not a refactor.** Callers ask
for `ModelRole.REASONING` or `ModelRole.CODEGEN`, never a model name. Google's
Gemini free tier speaks the same OpenAI-compatible protocol, so setting
`GEMINI_API_KEY` appends it to the provider chain and it is used automatically
when Groq fails. No caller changes.

A run against a small public site costs roughly 15–25 reasoning calls and one
codegen call per flow. If you hit a limit, lower `CRAWL_MAX_PAGES` and
`MAX_FLOWS_TO_GENERATE` first.

---

## Feature status

Honest accounting. "Scaffolded" means the seam is real and the degraded path
works, but the full version is not built.

| Feature | Status | Notes |
|---|---|---|
| URL as the sole required input | ✅ Implemented | Everything else is optional |
| Planner: login → crawl → structured plan | ✅ Implemented | BFS, depth- and page-capped, same-origin, skips logout links |
| Coverage gate before generation | ✅ Implemented | LLM judge + deterministic rubric that can only tighten it |
| Re-plan loop with specific feedback | ✅ Implemented | Cap 2, then force-proceed with `force_proceeded` recorded |
| Risk ranking carried through the pipeline | ✅ Implemented | Drives generation order, execution order, report order |
| Generator with live selector validation | ✅ Implemented | 8 strategies, live probe, drawn from the crawler's real inventory |
| AST safety audit + one repair round-trip | ✅ Implemented | Whitelist imports, forbidden names, required signature |
| Deterministic compiler fallback | ✅ Implemented | Labelled as such in the report |
| Runner: screenshots, DOM, console, traceback | ✅ Implemented | Per test, on its own browser context |
| Healer: SCRIPT_ISSUE vs GENUINE_DEFECT | ✅ Implemented | With a live locator re-probe as evidence |
| Confidence blend + 0.60 branch | ✅ Implemented | Separate code path, visible in UI and report |
| Assertion-weakening guard | ✅ Implemented | Mechanically blocks "fixes" that lower the bar |
| Visual regression (Pillow pixel diff) | ✅ Implemented | Cross-run baselines, diff images, separate report category |
| Bug packager (repro + screenshot + ticket) | ✅ Implemented | One directory per bug on disk |
| Report: JSON + Markdown + HTML, risk-ordered | ✅ Implemented | HTML is self-contained and theme-aware |
| Live decision log (API + Streamlit) | ✅ Implemented | Emitted on start/decide/finish, written to `events.jsonl` immediately |
| Natural-language intent (`ENABLE_INTENT_BIAS`) | ✅ Implemented | Biases scope without dropping mandatory edge/error coverage |
| Optional PRD informs planning | ✅ Implemented | Excerpt goes into the Planner prompt |
| Parallel execution (`ENABLE_PARALLEL_EXECUTION`) | ✅ Implemented | Semaphore-bounded, isolated contexts, off by default |
| Provider-agnostic fallback LLM | ✅ Implemented | Gemini free tier works by setting one env var |
| PRD gap analysis (`ENABLE_PRD_GAP_ANALYSIS`) | 🟡 Scaffolded | ChromaDB embeddings when installed; degrades to keyword overlap and **says which method ran**. Optional dependency, off by default. |
| Regression radar (`ENABLE_REGRESSION_RADAR`) | 🟡 Scaffolded | Per-target history, atomic writes, "N flows changed since last time". Compares consecutive runs only — no trend analysis. |
| Pattern library | 🟡 Scaffolded | 8 curated flow patterns, regex-triggered, injected as Planner hints. Hand-curated, does not learn. `ROADMAP` in the module is explicit about what is missing. |
| Offline stub mode | 🟡 Scaffolded | Deliberately not the agent. Exercises plumbing without a key; every report is stamped. |
| Learned patterns, domain packs | 🔴 Roadmap | Mine completed runs for flow shapes that repeatedly find defects |
| Pattern-conditioned generation | 🔴 Roadmap | Per-pattern code templates so codegen only fills in locators |
| Semantic visual diff | 🔴 Roadmap | Pixel diff has no notion of importance; layout-tree comparison would |
| Multi-step auth (SSO, MFA, magic link) | 🔴 Roadmap | Only form login and bearer tokens today |
| CI/CD, cross-browser matrix, hosting | ⛔ Out of scope | Explicitly excluded by the brief |

---

## Known limitations

**Read this section before quoting any number from a report.**

1. **Coverage is a sample, not a guarantee.** The crawl is depth- and
   page-capped (12 pages, depth 2 by default). Anything behind a form the agent
   cannot fill, a paywall, or JavaScript that needs a specific interaction
   sequence is simply not seen. A clean report means "nothing broke in what was
   tested", never "the application works".

2. **Generation touches the application.** Live selector validation clicks and
   fills. On a real application that means real state changes. Use staging.

3. **The pixel differ has no semantics.** An anti-aliasing or font-rendering
   difference between two machines can register as a change; a genuinely broken
   layout that happens to occupy the same pixels will not. That is why the
   threshold is configurable, why size mismatches are treated as regressions on
   their own, and why visual findings are reported separately rather than
   failing the run.

4. **Defect classification is a judgment, and it is sometimes wrong.** The
   confidence score exists precisely because it is fallible. Findings below 0.60
   are surfaced for a human rather than acted on; findings above it can still be
   wrong. Treat auto-filed bugs as triaged leads, not verdicts.

5. **Force-proceed leaves real gaps.** After two failed re-plans the pipeline
   proceeds with the best plan it has. The unmet rubric lines are listed in the
   report as untested-flow risk, but they are untested.

6. **Single browser, single viewport.** Chromium at 1280×900. Cross-browser and
   responsive testing are out of scope by the brief.

7. **Authentication is form-login or bearer-token only.** SSO redirects, MFA and
   magic links are not handled. A failed login is reported loudly as
   `AUTH BLOCKED` and the run continues over public pages — it does not silently
   report a green suite for an application it never got into.

8. **The AST sandbox is defence in depth, not a security boundary.**

9. **The smaller codegen model writes weaker tests than the reasoning model would.** That is a deliberate
   rate-limit trade. The deterministic compiler catches the worst of it, and the
   report records which path produced each file.

10. **No test-flakiness detection.** A test is run once (twice if healed). A
    genuinely flaky application will produce noisy results.

---

## Evaluating agent quality over time

Testing an agent that writes tests is circular unless you fix something. Two
things get fixed, and they are the ones that make results comparable across
model changes, prompt changes, and refactors.

### Fix the target: a seeded application

Stand up a small application with **deliberately injected defects** — a known
list. A checkout that drops the last item, a login that accepts an empty
password, a total that rounds wrong, a 404 that renders blank. Then measure per
run:

| Metric | Definition | Why it matters |
|---|---|---|
| **Defect recall** | injected defects found / injected defects | The headline number. Does it find real bugs? |
| **Classification precision** | correct SCRIPT_ISSUE vs GENUINE_DEFECT calls / total | Precision here is what a QA lead's trust is built on |
| **False-bug rate** | bugs filed with no injected defect behind them | The fastest way to lose that trust |
| **Risk agreement** | agreement with a human-labelled risk ranking (Cohen's κ) | Is the ordering actually useful? |
| **Repro validity** | packaged `repro.py` scripts that reproduce when run | An unreproducible ticket is worse than none |
| **Selector durability** | generated tests still passing after a cosmetic-only redeploy | Distinguishes real locators from lucky ones |
| **Coverage-gate yield** | rubric lines unmet at the end vs after the first plan | Does re-planning actually improve coverage? |
| **Cost** | LLM calls and tokens per confirmed defect | The efficiency number that decides adoption |

### Fix the rubric: calibration, not vibes

The rubrics in `llm/prompts.py` are versioned constants precisely so a change to
them is a visible diff. Two checks belong in the harness:

- **Confidence calibration.** Bucket healer actions by confidence and measure
  the actual correctness rate in each bucket. A well-calibrated agent is right
  about 70% of the time when it says 0.7. If the 0.6–0.7 bucket is only 40%
  correct, the threshold is in the wrong place — and that is a measurement, not
  an argument.
- **Gate agreement.** Have a human apply the coverage rubric to a sample of
  plans and compare verdicts. Disagreement tells you whether the rubric is
  ambiguous or the judge is lazy; the deterministic checker already tracks how
  often it has to override the model, which is a free proxy.

### What the repository already supports

- `reports/runs/<id>/events.jsonl` is a complete, timestamped decision trace per
  run — the raw material for all of the above.
- The regression radar stores per-target history, so "which flows changed since
  last time" is already computed.
- Baselines under `reports/baselines/` are cross-run by design.
- The 270 sanity tests pin the deterministic layer (rubric application,
  threshold branching, redaction, report ordering, selector helpers, the AST
  audit) so a regression there fails loudly rather than degrading quality
  quietly.

A run against a fixed target with a fixed rubric is comparable to the run before
it. That is the whole point.

---

## Repository map

```
config.py                   Flags, thresholds, model routing, replan cap = 2, confidence = 0.6
security.py                 SecretBox, redact_secrets, sanitize_url, credential-literal scanner
logging_setup.py            Structured logging with a mandatory redaction filter
cli.py                      Run the pipeline without the API or the UI

agents/
  orchestrator.py           Coverage gate, routing decisions, synthesis, run lifecycle
  planner.py                Login → crawl → structured plan (+ re-plan with feedback)
  generator.py              Live selector walk → model authoring → audit → compiler fallback
  healer.py                 Evidence gathering, classification, confidence, safe patching

graph/
  state.py                  Every typed model + the LangGraph state (no credentials)
  runtime.py                RunContext, EventSink, live progress snapshot
  graph.py                  Nodes, conditional edges, the re-plan and heal loops

llm/
  client.py                 Provider-agnostic, role-routed, retrying, rate-limit aware
  prompts.py                Every prompt + all three rubrics as constants
  json_utils.py             Defensive parse, repair, one retry
  offline_stub.py           Deterministic stub for keyless smoke runs

browser/
  session.py                Playwright lifecycle, storage_state propagation
  login.py                  Credential-safe form login and bearer tokens
  crawler.py                BFS crawl, boundaries, e-commerce detection
  selectors.py              Candidate generation, live probing, ranking, round-trip parsing
  sandbox.py                AST audit of generated code + restricted namespace
  runner.py                 Execution, evidence capture, parallel mode
  screenshots.py            Masked capture, redacted DOM snapshots

differentiation/
  risk_ranking.py           LLM + deterministic rubric, guaranteed back-fill
  confidence_scorer.py      Signal scoring, blending, the 0.60 branch
  visual_diff.py            Pillow pixel diff, cross-run baselines, diff images
  bug_packager.py           repro.py + screenshot.png + ticket.md + bug.json per defect
  prd_gap.py                ChromaDB or keyword overlap, method recorded
  regression_radar.py       Per-target history, "N flows changed since last time"
  pattern_library.py        8 canonical flow patterns + an honest roadmap

api/                        FastAPI: models.py, store.py, app.py
ui/streamlit_app.py         Live decision log, risk table, bugs, review queue
reports/generator.py        Risk ordering, Markdown + HTML rendering, artifact writing
tests/                      270 deterministic sanity tests (no LLM, no browser)
```

### Running the sanity tests

```bash
pytest tests/ -q
```

They cover the JSON parse/retry helpers, risk rubric application, confidence
threshold branching, redaction, report ordering, selector helper fallbacks and
the AST audit. They deliberately assert **nothing** about LLM behaviour — that
is not a testable property, and pretending otherwise would make the suite a
decoration.

---

*Built for the Bessemer Tech Catalyst AI/ML track. Free tools only: Groq's free
tier, Playwright, Pillow, FastAPI, Streamlit, LangGraph. No paid API anywhere in
the pipeline.*
