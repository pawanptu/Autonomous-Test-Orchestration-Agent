# Implement **Agent Memory**: per-agent cross-run learning for Planner, Generator and Healer

## Goal

Today every run against a URL starts from zero. The same site is re-crawled, the
same happy-path flows are re-planned, the same selectors are re-resolved from
scratch in a live browser walk, and the same failure is re-classified and re-filed
as a fresh bug. Nothing the agent learned on run 1 survives into run 2.

Add **agent memory**: a persistent, per-target store with a separate namespace for
each of the three sub-agents, so that a repeat run on the *same URL*:

1. **Fetches everything already completed** — the flows already covered and passing,
   the selectors already proved to resolve, the failures already classified — and
   carries them forward instead of redoing them.
2. **Spends the freed budget going deeper** — the next unsatisfied depth level, pages
   never crawled, flows that previously failed or never executed, and failures whose
   classification is still unresolved.

The visible outcome on a second run against the same URL:

```
Planner   : 18 flow(s) carried forward from run_abc; L1-L2 satisfied; planning L3 edge cases
Generator : 41/58 selectors served from memory (verified live); 2 flows sent straight
            to the deterministic compiler after repeated model-authoring failures
Healer    : F007 matches a known genuine defect first seen 3 runs ago (BUG-002);
            re-file suppressed, marked recurring. F011 matches a known flaky signature.
```

Coverage becomes cumulative across runs instead of resetting.

## Read before writing code

- `differentiation/regression_radar.py` — the existing cross-run store, and the model
  for this work. Copy its disciplines exactly: `_host_slug`, `history_path`,
  `_write_atomic` (tmp file + fsync + `os.replace`), versioned payload, corrupt file
  treated as "no history", and **never raise into the run**. Memory is an enhancement
  layered on a working pipeline; a locked or garbage JSON file must degrade to
  today's behaviour, not fail a run.
- `graph/state.py` — `TestFlow`, `TestStep`, `SelectorValidation`, `GeneratedTest`,
  `TestResult`, `HealerAction`, `PackagedBug`, `FinalReport`, `OrchestrationState`,
  `initial_state`.
- `agents/planner.py` — `generate_plan`, `renumber_flows`, `explore_target`.
- `agents/generator.py` — `resolve_flow_selectors` (the live walk), `generate_test`,
  `compile_steps_to_source` (the deterministic fallback), `render_module`.
- `agents/healer.py` — `derive_signals`, `heal_failure`, `apply_fix`,
  `patch_weakens_assertions`, and `differentiation/confidence_scorer.py`.
- `differentiation/bug_packager.py::next_bug_id` — bug ids are allocated per run
  directory, so the *same* defect gets a different id every run today.
- `graph/graph.py` (`planner_node`, `coverage_gate_node`, `generator_node`,
  `healer_node`), `agents/orchestrator.py::execute_run` (~line 632, beside
  `record_radar`), `llm/prompts.py` (`planner_user`, `generator_user`, `healer_user`).
- `config.py::Settings`, `.env.example`, `api/models.py::RunRequest`, `api/app.py`,
  `cli.py`, `ui/streamlit_app.py`, `reports/generator.py`.

Match the surrounding style: a module docstring that explains *why* and states what
is deliberately **not** stored, typed pydantic models, `from __future__ import
annotations`, `log = get_logger("aivor.memory.<agent>")`, comments only where the
reasoning is non-obvious.

---

## Part 0 — the shared substrate

Create a package `differentiation/memory/` (three agent schemas are too much for one
flat module, and per-agent files mean one corrupt file cannot sink the other two):

```
differentiation/memory/__init__.py          # public API re-exports
differentiation/memory/store.py             # paths, atomic IO, redaction, TTL, fingerprint
differentiation/memory/keys.py              # flow_key, selector_key, failure_signature
differentiation/memory/planner_memory.py
differentiation/memory/generator_memory.py
differentiation/memory/healer_memory.py
```

Layout on disk, one directory per target:

```
reports/baselines/_memory/<host-slug>/meta.json        # version, site fingerprint, run refs
reports/baselines/_memory/<host-slug>/planner.json
reports/baselines/_memory/<host-slug>/generator.json
reports/baselines/_memory/<host-slug>/healer.json
```

`store.py` provides: `memory_dir(target_url)`, `read_json(path) -> dict` (returns
`{}` on any problem, logs at debug), `write_atomic(path, payload)` (returns `bool`,
never raises), `site_fingerprint(site_map)`, `is_stale(entry, ttl_days)`, and a
`MemoryMeta` model. Every namespace payload carries `version`, sanitised `target`,
`updated_at`, and a `runs` list capped at `MEMORY_MAX_RUNS` (default 20).

**What is deliberately not stored — state this in every module docstring.** No DOM,
no screenshots, no raw error text, no step values, no full URLs, no credentials, no
generated source. Every string goes through `security.redact_text`; the target goes
through `security.sanitize_url`. These files are read on every run, so they stay
small, and the less they hold the less there is to leak.

### Stable identity (`keys.py`) — the part that silently breaks everything if wrong

`renumber_flows` reassigns `F001, F002, ...` on every plan, and flow *names* are
model output that drifts between runs. **Flow ids and names are not stable across
runs and must not be used as memory keys.** The regression radar keys on `flow_id`
and documents that as a known limitation; do not inherit it.

```python
def flow_key(flow: TestFlow) -> str:
    """Stable cross-run identity for a flow: sha256 over normalised semantics."""

def selector_key(*, page_path: str, action: str, intent: str) -> str:
    """Stable identity for one selector resolution on one page."""

def failure_signature(*, flow_key: str, signals: ConfidenceSignals,
                      result: TestResult) -> str:
    """Stable identity for a *kind* of failure, not one occurrence."""
```

- `flow_key`: `category` + URL *path* of `flow.url` (sanitised; host and query
  dropped so a session id cannot key the memory) + the ordered step shape, where each
  step contributes `(action, normalised target, value *shape*)`. Normalise by
  lowercasing, collapsing whitespace and stripping id-shaped digits. **Never hash a
  step `value`** — record only its coarse kind (`empty`, `text`, `email`, `number`,
  `long`), so a password change cannot alter a key and a password can never be
  recovered from one. First 16 hex chars.
- `selector_key`: URL path only + `action` + normalised `intent` (the plain-language
  target that `resolve_target` is given).
- `failure_signature`: `flow_key` + `signals.failure_kind` + normalised
  `result.error_type` + `locator_named_in_error` + which marker classes fired
  (`captcha_or_bot_wall`, `auth_wall`, `network_flaky`, `spa_race_suspected`).
  Deliberately excludes the error *message*, so the same defect matches across runs
  even when the message carries a timestamp or an element index.

Each memory entry also stores a redacted `name_hint` so reports read naturally, and
a secondary normalised-name lookup so a small step edit does not orphan an entry —
prefer the fingerprint match, fall back to the name match, and record which matched
in `match_kind`.

### Shared invalidation

- **Site fingerprint.** Hash the structural shape of the crawl (sorted page paths,
  per-page form / input / button counts) into `meta.json`. On load, if the new crawl
  differs beyond a threshold (say >30% of known paths gone, or changed form shapes),
  mark all three namespaces `stale=True`: entries still suppress duplicate work for
  de-duplication purposes, but no longer count as *verified*, and every `high` risk
  flow is re-verified. Emit a `decision` event explaining the demotion.
- **TTL.** `MEMORY_TTL_DAYS` (default 14). Expired entries stop counting toward
  satisfied depth and stop being trusted as selector or classification priors.
- **Never carry a failure forward as coverage.** `failed`, `error` and `not_run`
  always re-plan and re-run.

---

## Part 1 — Planner memory (`planner_memory.py`)

*What the planner already covered, and how deep it has gone.* This is the memory that
delivers the headline behaviour: same URL → fetch what is done → go deeper.

### Schema

`PlannerMemory`: `version`, `target`, `updated_at`, `depth_satisfied: str`,
`known_pages: list[str]` (paths only), `flows: dict[flow_key, FlowMemory]`, `runs`.

`FlowMemory`: `flow_key`, `name_hint`, `category`, `depth_level`, `risk`,
`last_status`, `last_run_id`, `first_seen_at`, `last_seen_at`, `times_run`,
`times_passed`, `consecutive_failures`, `steps_count`.

### The depth ladder

Depth must be explicit, ordered and machine-checkable — "go deeper" cannot be left to
the model's mood:

| Level | Name | Satisfied when |
|-------|------|----------------|
| L1 | `smoke` | The entry page loads and its main navigation works |
| L2 | `happy_path` | Every discovered top-level area has a passing happy-path flow |
| L3 | `edge_case` | Boundary / empty / overlong-input / pagination flows exist per form and list |
| L4 | `error_state` | Invalid input, invalid credentials, 404 / unreachable, blocked-action flows exist |
| L5 | `cross_flow` | Multi-page stateful journeys (browse → add to cart → checkout) exist |
| L6 | `deep_crawl` | Pages beyond the previous crawl frontier are planned and executed |

A level is satisfied only when its flows exist **and** their last status is `passed`
or `healed`. A level holding a `failed`, `error` or `not_run` flow is *unsatisfied*
and those flows are re-planned first — regression before expansion.

`next_depth(memory) -> DepthLevel` returns the lowest unsatisfied level. L6 raises
`crawl_max_depth` / `crawl_max_pages` for that run (bounded: cap at depth 5 / 40
pages, never past the configured ceilings) and steers the planner at pages absent
from `known_pages`.

### Wiring

**`graph/graph.py::planner_node`** — after the crawl, before `generate_plan`: load
planner memory, compute a `PlannerDirective`, emit a `decision` event
(`"Memory: 18 flow(s) carried forward from run_abc; L1-L2 satisfied; planning L3 edge
cases"`, with confidence and target level in `detail`), apply any L6 crawl
escalation, pass the directive into `generate_plan`. Empty or disabled memory must
behave exactly as today and say `first run for this target`.

**`agents/planner.py::generate_plan`** — new keyword-only `memory: PlannerDirective |
None = None`, threaded to `planner_user`. After the model returns, **drop any flow
whose `flow_key` is already covered and not due for re-verification** — the model
re-proposes covered flows no matter how the prompt is worded, so enforce it in code.
Log the drop count; if the model returns *only* duplicates, retry once with a sharper
directive, then fall back to today's unconstrained behaviour rather than emitting an
empty plan.

**`llm/prompts.py::planner_user`** — a memory section before the "Produce between 6
and N flows" line:

```
MEMORY OF PREVIOUS RUNS ON THIS TARGET
Already covered and passing - do NOT re-plan these:
  - [happy_path] Browse catalogue by category
  - [happy_path] Open a product detail page
  ...
Needs re-verification (previously failed or never executed):
  - [edge_case] Search with an empty query
Pages never yet exercised: /basket, /checkout/address
Depth already satisfied: L1 smoke, L2 happy_path
YOUR TARGET THIS RUN: L3 edge_case - boundary values, empty and overlong
input, pagination limits, and unusual-but-legal sequences.
Plan ONLY new flows at this level plus the re-verification flows above.
```

Truncate the covered list (~40 entries, longest-uncovered-area first) so the prompt
cannot blow past `llm_max_tokens`.

**`graph/graph.py::coverage_gate_node` — the trap that will break this feature if
missed.** The coverage rubric demands happy-path + edge + error coverage. On a depth-3
run the plan legitimately contains no happy paths, so the gate rejects it, forces a
replan, and drags the planner straight back to the ground it was told to skip. Pass
the carried-forward flows into `evaluate_coverage` and `deterministic_coverage_check`
and let them satisfy their rubric checks, citing the originating run in
`RubricCheck.evidence` (`"covered by run_abc: Browse catalogue by category"`). The
gate must judge *cumulative* coverage, not this run's slice.

---

## Part 2 — Generator memory (`generator_memory.py`)

*Which selector actually worked, and which flows the model cannot author.* The
generator's expensive work is the live walk in `resolve_flow_selectors`: opening a
browser context and searching the DOM for every step's target. That result is highly
reusable — the same button on the same page resolves the same way next run.

### Schema

`GeneratorMemory`: `version`, `target`, `updated_at`,
`selectors: dict[selector_key, SelectorMemory]`,
`authoring: dict[flow_key, AuthoringMemory]`, `runs`.

`SelectorMemory`: `selector_key`, `page_path`, `action`, `intent_hint` (redacted),
`expression`, `strategy`, `match_count`, `last_verified_at`, `last_run_id`,
`hits`, `misses`, `consecutive_misses`.

`AuthoringMemory`: `flow_key`, `name_hint`, `module_name` (so regenerated test files
keep stable names across runs and diff cleanly), `last_generated_by_model`,
`repair_attempts`, `fallback_used`, `consecutive_model_failures`,
`last_validation_error_kind` (a *kind*, e.g. `syntax`, `banned_import`,
`credential_literal`, `no_assertion` — never the raw text), `last_run_id`.

### Uses

1. **Seed the live walk.** In `resolve_flow_selectors`, before calling
   `resolve_target`, look up `selector_key`. On a hit, verify the remembered
   expression against the current page (the same cheap presence check
   `browser.selectors.reprobe_expression` performs for the healer). If it still
   resolves to exactly one visible node, build the `SelectorValidation` from memory
   with `chosen_strategy` unchanged and `note="served from memory, re-verified live"`,
   and skip the full candidate search. On a miss, fall through to the normal search
   and record the miss.

   **A remembered selector is never trusted blind — it is always re-verified against
   the live page before use.** A stale selector that silently "resolves" from memory
   would generate a test that cannot possibly pass, which is worse than the search it
   replaced. Increment `consecutive_misses` on failure and evict the entry at 2.

2. **Skip the doomed model round-trip.** When `AuthoringMemory.consecutive_model_failures
   >= 2` for a flow, bypass the model author in `generate_test` and go straight to
   `compile_steps_to_source`. Record `generated_by_model` as the fallback compiler
   exactly as today, and add a note that memory routed it there, so nobody reads the
   fallback as a fresh model failure. Reset the counter the moment a model authoring
   attempt succeeds.

3. **Stable module names.** Reuse `module_name` from memory in `render_module` /
   `write_tests` so run 2's generated suite diffs against run 1's instead of
   reshuffling file names.

4. **Prompt hint.** In `llm/prompts.py::generator_user`, mark the resolved steps that
   came from memory (`"source": "memory (re-verified)"` inside the existing
   `SELECTOR RESOLUTION` block). Do not add a new free-text section; the generator
   prompt is deliberately mechanical.

### Safety

Store a selector expression **only after** it has passed the same credential-literal
scan the generator already applies to model output — a resolved selector can quote
page text, and page text can contain an email or a token. Never store a step value or
generated source. Drop every selector entry for a page whose structural fingerprint
changed.

---

## Part 3 — Healer memory (`healer_memory.py`)

*What this failure turned out to be last time, and whether the fix worked.* The
healer is where cross-run memory changes the answer most: today, a genuine defect
that survives three runs is classified from scratch three times and filed as three
different bug ids.

### Schema

`HealerMemory`: `version`, `target`, `updated_at`,
`failures: dict[failure_signature, FailureMemory]`, `runs`.

`FailureMemory`: `signature`, `flow_key`, `name_hint`, `failure_kind`,
`classification` (`DefectClass`), `blended_confidence`, `times_seen`,
`first_seen_at`, `first_seen_run_id`, `last_seen_run_id`,
`patch_kind_tried` (`locator_substitution` | `wait_adjustment` | `none`),
`patch_worked: bool | None`, `rerun_status`, `needs_human_review`,
`packaged_bug_id`, `resolved_at` (set when the flow later passes clean),
`flaky_score: float`.

### Uses

1. **Recurring defect, one bug.** When a failure's signature was previously
   classified `GENUINE_DEFECT` and packaged, pass the stored `packaged_bug_id` into
   the bug packager: do not allocate a new id via `next_bug_id`, mark the bug
   `recurring=True`, and record `first_seen_run_id` and `times_seen` on it. The
   report then says *"BUG-002, first seen 3 runs ago, still present"* instead of
   presenting a four-run-old defect as new. Conversely, when a signature with a
   `packaged_bug_id` stops appearing and the flow passes, set `resolved_at` and let
   the report note the fix.

2. **Known flaky.** A signature that has failed and then passed on re-run across
   multiple runs earns a rising `flaky_score`. Above a threshold, surface it as flaky
   in the report and bias toward `SCRIPT_ISSUE` / `ENVIRONMENT` rather than filing a
   defect on a coin-flip.

3. **Do not repeat a failed patch.** If `patch_kind_tried` is recorded with
   `patch_worked=False`, `apply_fix` must not re-apply that same patch kind for that
   signature. Try the other kind if it is applicable; otherwise skip straight to
   human review with the memory cited as the reason. Re-applying a fix that is known
   not to work burns a rerun and produces the same red.

4. **Prior in the prompt.** Add a `PRIOR RUNS` block to
   `llm/prompts.py::healer_user`, clearly labelled as historical evidence rather than
   current observation:

   ```
   PRIOR RUNS - this failure signature has been seen before:
     seen 3 time(s), first in run_abc
     previously classified GENUINE_DEFECT at confidence 0.72
     a locator substitution was applied once and did NOT fix it
   Treat this as one more piece of evidence, not as a verdict. The live
   probe and the DOM snapshot above are the current facts.
   ```

### The bounds on healer memory — implement these explicitly

The healer's decision is the one the entire pipeline exists to automate, so memory
must inform it without hijacking it:

- Memory may shift the **blended** confidence by at most ±0.10, and that clamp is a
  named constant (`MEMORY_CONFIDENCE_INFLUENCE = 0.10`) with a test. Without a clamp,
  one wrong classification reinforces itself forever: it raises the confidence, which
  gets stored, which raises it again.
- Memory may **never** move a decision across the 0.6 auto-apply threshold on its own.
  If the live evidence lands below the threshold and only memory pushes it above,
  the decision stays below and the run records why.
- Memory **never** touches `patch_weakens_assertions`. That guard is mechanical and
  absolute regardless of how many prior runs a patch appeared to work in.
- A stale or TTL-expired entry is shown to the model as history but contributes
  **zero** to the numeric blend.

---

## Cross-cutting wiring

**`config.py` + `.env.example`** — `enable_agent_memory: bool = True`
(`ENABLE_AGENT_MEMORY`) as the master switch, plus per-agent switches
`ENABLE_PLANNER_MEMORY`, `ENABLE_GENERATOR_MEMORY`, `ENABLE_HEALER_MEMORY` (all
default `true`, all gated behind the master), `memory_max_runs: int = 20`,
`memory_ttl_days: int = 14`. Include them in `from_env` and the reported flags dict.

**`graph/state.py`** — add to `OrchestrationState`: `agent_memory: dict[str, Any]`
(the three loaded namespaces plus meta), `memory_directive: dict[str, Any]`,
`carried_flows: list[CarriedFlow]`, `memory_stats: dict[str, Any]`. Set them in
`initial_state`. Add a `CarriedFlow` model and a `memory` block on `FinalReport`
holding all three agents' contributions.

Load all three namespaces **once** at run start (in `execute_run`, before the graph
is invoked) and put them in state, rather than re-reading files inside each node.

**`agents/orchestrator.py::execute_run`** — beside the `record_radar` call, record
all three namespaces. Same shape as the radar: gated on the flag, skipped when
already recorded, never raises, and each namespace reports its own `persisted` bool
so one unwritable file does not hide the other two.

**`reports/generator.py`** — an "Agent memory" section in JSON, Markdown and HTML,
with one sub-block per agent: planner (flows carried forward with their originating
run, new this run, depth reached, levels outstanding), generator (selectors served
from memory vs re-resolved, flows routed to the fallback compiler by memory), healer
(recurring defects with their original bug ids and age, newly resolved defects, flows
flagged flaky). Include the cumulative sentence: *"Cumulative coverage across 3 runs:
31 flows, 27 passing, L1-L3 satisfied, L4 error_state next."*

**`api/models.py::RunRequest`** — note `extra="forbid"`, so every field must be
declared: `use_memory: bool = True`, `reset_memory: bool = False`,
`memory_agents: list[Literal["planner","generator","healer"]] | None = None` (None
means all), `depth: Literal["auto","L1","L2","L3","L4","L5","L6"] = "auto"`. Surface
the memory block in status and report responses. Add `GET /memory/{host}` and
`DELETE /memory/{host}` (optional `?agent=` filter) — validate the host slug against
the same safe-charset rule as `_host_slug` and reject anything containing `/`, `\` or
`..` **before** touching the filesystem.

**`cli.py`** — `--no-memory`, `--reset-memory`, `--memory-agent`, `--depth`.

**`ui/streamlit_app.py`** — near the regression-radar expander (~line 661), an "Agent
memory" expander with a tab or sub-section per agent, plus a "Reset memory for this
target" control.

---

## Tests (`tests/`, matching the existing style)

**Keys and store**
1. `flow_key` is stable when `renumber_flows` changes ids and when a name is re-cased
   or re-spaced; it differs for a genuinely different step sequence.
2. `flow_key` and `selector_key` never embed a step value — change only a
   password-shaped value and assert the digest is unchanged.
3. `failure_signature` matches across runs when only the error message text differs,
   and differs when `failure_kind` differs.
4. Corrupt / empty / unreadable namespace file → load returns empty, no exception,
   and the other two namespaces still load.
5. `record` never raises when the directory is unwritable and reports
   `persisted: False`.
6. Redaction: a payload built from flows and selectors carrying secret-shaped strings
   contains none of them.

**Planner**
7. `next_depth` returns the lowest unsatisfied level, and a level containing one
   failed flow is unsatisfied even when every other flow in it passed.
8. Carried-forward flows satisfy the coverage rubric checks (guards the gate trap).
9. `generate_plan` drops model-proposed duplicates of covered flows, and does not
   return an empty plan when the model proposes only duplicates.
10. TTL expiry stops entries counting toward `depth_satisfied` while still suppressing
    duplicate planning.

**Generator**
11. A remembered selector that no longer resolves is not used, is counted as a miss,
    and is evicted after two consecutive misses.
12. A remembered selector that still resolves produces a `SelectorValidation`
    equivalent to a fresh search, with the memory note set.
13. Two consecutive model-authoring failures route the third attempt straight to
    `compile_steps_to_source`, and one success resets the counter.
14. `module_name` is reused across runs for the same `flow_key`.

**Healer**
15. A recurring genuine defect reuses its original `packaged_bug_id` instead of
    allocating a new one, and carries `times_seen` and `first_seen_run_id`.
16. A signature whose stored patch kind failed is not re-patched with that kind.
17. Memory shifts the blended confidence by at most `MEMORY_CONFIDENCE_INFLUENCE`,
    and cannot by itself carry a decision across the 0.6 auto-apply threshold.
18. `patch_weakens_assertions` still rejects a weakening patch for a signature whose
    memory says that patch previously "worked".
19. A stale or expired entry contributes zero to the blend.

**Site change**
20. A changed site fingerprint marks all three namespaces stale, forces high-risk
    re-verification, and drops selectors for changed pages.

## Acceptance

- Two consecutive runs against `https://books.toscrape.com/` produce **disjoint**
  new-flow sets, and run 2's report shows run 1's flows carried forward with run 1's
  id.
- Run 2 resolves a materially smaller share of selectors from scratch than run 1, and
  the report states how many were served from memory.
- A defect present in two consecutive runs appears under **one** bug id, marked
  recurring, with its first-seen run.
- Run 2's decision log contains one memory event per agent.
- `ENABLE_AGENT_MEMORY=false` reproduces today's behaviour byte-for-byte in the parts
  of the report that existed before; each per-agent flag can be turned off
  independently with the other two still working.
- Deleting `reports/baselines/_memory/` mid-flight breaks nothing.
- The existing regression-radar files and behaviour are untouched.
- `python -m pytest` passes.

## Non-goals

Do not build a shared server-side memory, cross-target learning, or a database. Do
not change the radar's file format. Do not cache generated test *source* for replay —
generator memory stores selector resolutions and authoring outcomes, not artefacts.
Do not let memory become a substitute for live verification anywhere: every
remembered selector is re-probed before use, and every remembered classification is
evidence weighed against fresh evidence, never a verdict. Say all of this in the
module docstrings so the next reader does not assume otherwise.
