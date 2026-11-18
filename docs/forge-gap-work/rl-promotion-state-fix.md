# Preserve completed RL training when Registry promotion fails

October 15, 2026. Scope: `src/bufogen/train_rl.py` and its orchestration tests.

The actual direct-routing proof completed its optimizer update, but optional
Registry linking raised `CommError`. Previously the shared exception handler
overwrote the experiment's status with `failed`, obscuring the distinction
between completed training and a failed Registry operation.

The trainer now persists the service's returned step, checkpoint reference, and
completed update count immediately after `backend.train` returns, before metrics
upload. After the last requested update, it resolves the returned step alias to
an immutable W&B version using `artifact.version`; `artifact.name` can retain
the queried alias. It looks up that immutable URI again and verifies matching
artifact ID/digest, entity/project, actual logging run, and available base-model
metadata.

Optional Registry linking has its own result:

- Successful linking: `status=completed`, `training_status=completed`,
  `promotion_status=completed`.
- Linking exception: completed training and immutable checkpoint are preserved,
  `promotion_status=failed` and `promotion_failure.error_type` are recorded, an
  explicit warning is emitted, and the verified checkpoint URI is returned.
- Metrics upload or artifact-verification failure after completed training:
  `status=post_training_failed` is recorded and the exception is raised. The
  returned checkpoint and actual completed update count remain available for
  read-only recovery. No automatic training retry is introduced.

`loop_report.py` already includes the full RL record; it needs no schema change
to display these distinctions. No original attempt, benchmark-fingerprinted
agent/model source, or production route was changed.

Verification: **52 focused RL tests passed; Ruff passed.** Orchestration
regressions cover normal promotion, actual `wandb.errors.CommError` during
promotion, metrics failure after optimizer return, and foreign artifact
provenance. Each confirms one optimizer call, durable update evidence, and the
appropriate promotion/verification outcome.

Artifact existence and successful training do not establish serving success or
quality improvement. The root's separate inference test remains necessary; its
first v2 serving attempt returned HTTP 500 and must not be described as a
successful end-to-end run.
