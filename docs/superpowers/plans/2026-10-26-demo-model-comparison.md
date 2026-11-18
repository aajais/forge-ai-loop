# Demo model comparison implementation plan

> Execute inline with test-driven development and verification-before-completion.

**Goal:** Limit Studio choices to base Qwen, one-epoch SFT Qwen, evaluated RL Qwen, and Sol; compare all four concurrently in a vertical view.

**Architecture:** A shared browser model catalog filters the existing status response without removing experimental backends. A comparison page uses four independent existing generation requests, with fresh histories, fixed-order result slots, individual errors and elapsed times. Completed comparisons are retained in browser storage; ordinary generations remain in the server manifest.

**Spec:** User requested a demo-focused model list and simultaneous vertically stacked execution, and selected RL as a fourth model.

**Constraints:** Preserve historical labels and all backend configurations. Pin RL's default to the evaluated v6 checkpoint. No automated paid inference for testing. Preserve current typography/colors; use full-width vertically separated outputs, readable model identities and clear running/error states. No Git checkout exists; originals backed up under /private/tmp/bufo-demo-ui-before.

## Tasks

- [x] Add behavioral tests for curated selection and historical identity; implement shared demo catalog and RL default.
- [x] Add tests proving concurrent submission, fixed output order, per-model failure isolation, duplicate-submit guard, fresh context, and saved-result restoration. Implement comparison page/controller and links from Studio.
- [ ] Run Node UI tests and relevant Python API/model tests; inspect page through Chrome, without launching paid generations. Review changed files and document usage.

## Review focus

- One unavailable model must not hide other outputs or silently substitute a model.
- Out-of-order completion must not reorder model sections.
- Failed/non-JSON responses must be visible without discarding successes.
- Reload must preserve completed results and mark interrupted requests honestly; never auto-resubmit.
- Comparison requests must use independent fresh histories and preserve exact checkpoint identity.

## Verification ledger

- Node chat/comparison tests: 12 passed, including actual PNG rendering paths, concurrency, fixed ordering, isolated failures and reload behavior.
- Python UI/model/inference-policy/benchmark-runtime checks: 48 passed (75 existing Weave/Pydantic deprecation warnings).
- Ruff checks passed for changed Python tests.
- Independent read-only review: no blocking findings. Image-rendering test gap addressed.
- Browser visual verification remains incomplete: user browser activity interrupted inspection. Attempting to start the server from the managed terminal failed with `Operation not permitted` when binding localhost. No paid model calls were made.
- Ruling: keep comparison orchestration in a dedicated page sharing the existing API, rather than expand the chat controller; this keeps normal conversation behavior unchanged and all comparison outputs on one page.
- Ruling: pending-checkpoint regression test now sets its own explicit PENDING fixture because the default RL checkpoint is intentionally configured.
