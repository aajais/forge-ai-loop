# Forge AI Loop preflight

Verified 2026-10-15. This research performed authenticated GET requests only; no projects, providers, datasets, jobs, artifacts, or routes were created. Credentials were loaded through `bufogen.settings` and were not printed.

## Live access and eligibility

The current environment resolves to **wandb-smle/bufo-gen**.

| Read-only check | Actual result |
| --- | --- |
| `GET https://forge.coreweave.com/api/distillation/v1/projects` | HTTP 200, four existing projects; none owns `bufo-gen` |
| `GET /api/distillation/v1/wandb-projects` | HTTP 200; existing `bufo-gen` project is accessible |
| `GET /api/distillation/v1/projects/availability?alias=bufo-gen&project=bufo-gen&mode=import` | HTTP 200, `{"available":true}` |
| `GET /api/training/v1/preview/base-models` | HTTP 200; includes exact `google/gemma-4-26B-A4B-it` |

The deployment model catalog includes the requested Gemma model. This verifies advertised architecture eligibility, not project-specific job acceptance, available quota, or a completed training job. Public documentation still labels Gemma experimental and available on request. Do not substitute another model silently. [Supported models](https://docs.coreweave.com/model-distillation/supported-models), [live catalog endpoint](https://docs.coreweave.com/products/post-training/serverless-training/api-reference/models/list-base-models).

## Management interfaces

Base URL: `https://forge.coreweave.com/api/distillation/v1`. Headers: `Authorization: Bearer <WANDB_API_KEY>`, `Wandb-Entity: wandb-smle`, and `Content-Type: application/json` for JSON writes. `Wandb-Entity` is a header, not a query parameter. [Authentication](https://docs.coreweave.com/model-distillation/reference/authentication).

To attach the existing W&B project, the documented operation is `PUT /projects/bufo-gen`:

```json
{
  "project": "bufo-gen",
  "project_mode": "import",
  "targets": [{"model_ref": "<registered-provider>/<exact-serving-model>", "weight": 1}]
}
```

This **publishes initial routing**, so use the actual existing teacher/provider configuration. A project requires at least one target; the proxy alias and W&B project name may differ. Response HTTP 201 includes `id`, `entity`, `project`, `alias`, `proxy_backend`, timestamps, and `versions`. [Project creation schema](https://docs.coreweave.com/model-distillation/reference/management/projects/create-a-model-distillation-project).

`GET /providers` is documented to lazily provision the built-in `wandb-inference` provider, so it was deliberately not called during this read-only task. Teacher provider/model readiness remains unchecked. [Provider listing](https://docs.coreweave.com/model-distillation/reference/management/providers/list-providers).

## Curated dataset and complete tool trajectories

`POST /projects/{alias}/datasets` with `{"name":"bufo-sol-curated"}` creates an immediately ready empty dataset (HTTP 201). `POST /projects/{alias}/datasets/{id}/entries?mode=append` accepts a **JSON array**, or NDJSON with `Content-Type: application/x-ndjson`; maximum 5,000 rows/request. Add `dry_run=true` for transactional validation without committing rows. [Upload workflow](https://docs.coreweave.com/model-distillation/studio/datasets-upload).

Exact allowed row fields are `messages` (required), `tools`, `tool_choice`, `response_format`, `row_id`, `group_id`, `split`, `metadata`, and `provenance`; other top-level keys are rejected. A representative application row is:

```json
{
  "row_id": "teacher-trace-123",
  "group_id": "scenario-123",
  "split": "train",
  "messages": [
    {"role": "system", "content": "<actual drawing instructions>"},
    {"role": "user", "content": "<actual user prompt>"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_render", "type": "function", "function": {"name": "render_svg", "arguments": "<original JSON argument string>"}}]},
    {"role": "tool", "tool_call_id": "call_render", "content": "<actual render result>"},
    {"role": "assistant", "content": "<actual final response>"}
  ],
  "tools": [{"type": "function", "function": {"name": "render_svg", "description": "<actual description>", "parameters": {"type": "object", "properties": {}}}}],
  "metadata": {"teacher_model": "<actual Sol model>", "curation": "<actual selection evidence>"},
  "provenance": {"source": "weave", "trace_id": "<actual trace ID>"}
}
```

Preserve the complete real tool schema and message history; the placeholder schema above is illustrative. `messages` permits 2–1,000 messages, including `system`, `developer`, `user`, `assistant`, and `tool`; **the final message must be assistant**. `tools` permits up to 128 function tools. `split` is exactly `train` or `val`; one group cannot span both. `row_id` and `group_id` are at most 512 characters. Metadata and provenance are each limited to 64 KiB and ten levels. Response fields: `dry_run`, `created`, `updated`, `revision`, `entry_counts: {total,train,val}`, `group_conflicts`. [Entry schema](https://docs.coreweave.com/model-distillation/reference/management/datasets/add-or-replace-dataset-entries).

This is a direct supported route for curated Sol tool-use examples. Existing self-logged Weave calls can also be sourced by setting dataset `params.query.source` to `{"entity":"wandb-smle","project":"bufo-gen"}` rather than `"task"`, but only compatible reconstructed Chat Completions qualify. Explicit row upload gives control over teacher selection and group splits. [Dataset sources](https://docs.coreweave.com/model-distillation/studio/datasets).

## Distillation job and deployed artifact

`POST /projects/{alias}/finetunes` requires all three fields below (the endpoint schema requires `model_name` even though the UI can generate it):

```json
{
  "dataset_id": "<ready-dataset-UUID>",
  "model_name": "bufo-gemma4-sol-sft",
  "base_model": "google/gemma-4-26B-A4B-it",
  "model_config": {
    "config": {"batch_size": 2, "epochs": 1, "assistant_turns": "all"}
  }
}
```

Omit `model_config` to accept automatic configuration. Optional `relabel_run_id` selects completed replacement outputs; omit it when uploaded rows already contain Sol outputs. Additional supported options: `default_temperature` 0–2; `model_config.max_examples` positive integer; batch size `"auto"` or 1–32; positive integer epochs; positive scalar learning rate or nonnegative per-step list; assistant turns `all|last`; `metric_logging` and `experimental_config` objects. Train all assistant turns to learn intermediate tool decisions. [Create fine-tune schema](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/create-a-fine-tune).

No minimum training-row count was specified in the public fine-tune schema inspected; it requires a ready dataset with training rows. The effective service minimum remains unknown until validation/submission. The documented loss selection is `assistant_turns: all|last`; no explicit per-message weight mask or custom tool/chat-template field is modeled by the fine-tune request. Arbitrary `experimental_config` does not establish that an undocumented option works. ART documents masking non-assistant tokens, but exact preservation of uploaded assistant `weight` fields through Model Distillation was not verified. [ART SFT data format](https://art.openpipe.ai/fundamentals/sft-training/).

Creation returns HTTP 202. Poll `GET /projects/{alias}/finetunes/{id}` or the list endpoint. Status is `queued|training|deployed|failed`; `training_progress` is 0–100. Preserve `id`, `task_id`, `dataset_id`, `model_name`, `base_model`, `dataset_artifact_ref`, and **`model_artifact_ref`**. Only a deployed job with its actual artifact is a tuned model. Failure type `RemoteTrainingFailed` cannot be resumed by retrying the distillation job. [Fine-tuning behavior](https://docs.coreweave.com/model-distillation/studio/fine-tuning).

## SFT initialization for Serverless RL

Serverless training base: `https://forge.coreweave.com/api/training/v1`. Model creation is `POST /preview/models` with `{project,name,base_model,entity,return_existing}` and returns `{id,entity,project,name,base_model,run_id}`. There is **no documented initial-artifact field** in this request. RL creation is `POST /preview/training-jobs` with `{model_id,trajectory_groups,experimental_config?}`. [Create model](https://docs.coreweave.com/products/post-training/serverless-training/api-reference/models/create-model), [RL job schema](https://docs.coreweave.com/products/post-training/serverless-training/api-reference/training-jobs/create-rl-training-job).

The documented warmup path continues SFT and RL on the **same training model / LoRA**. ART `ServerlessBackend.register` passes `return_existing=True` using entity/project/storage name. Therefore resume the actual distillation training identity with the original Gemma `base_model`; verify its checkpoint step and artifact before the first RL rollout. Distillation's `task_id` is not proven to be the serverless `model_id`; recover the real underlying identity instead of assuming the IDs match. Preserve the immutable SFT artifact for comparison before RL advances the model. [ART warmup](https://art.openpipe.ai/fundamentals/sft-training/), [ServerlessBackend implementation](https://github.com/OpenPipe/ART/blob/main/src/art/serverless/backend.py).

ART 0.5.20 also implements `await backend._experimental_fork_checkpoint(model, from_model=..., from_project=..., not_after_step=...)`. It selects W&B `lora` artifacts by `stepN` alias, downloads one, and uploads a copy under the destination collection. Its source explicitly cautions that serverless checkpoint tracking may not reflect the copied artifact until the next training step. **A successfully served copied adapter does not establish that RL training initialized from it.** Prefer verified same-model continuation unless fork initialization can be established independently. [Checkpoint forking](https://art.openpipe.ai/features/checkpoint-forking/).

## Runtime issues found before any job launch

- Current `.venv` has no `art` module. The repository declares optional `openpipe-art>=0.5`.
- The latest release inspected was `openpipe-art==0.5.20`. Its wheel no longer contains `art.auto_trajectory`; the current `train_rl.py` import fails with that release. Capture now uses `with art.Trajectory() as trajectory:` and `trajectory.exchanges`; checking only legacy `messages_and_choices` is insufficient. [Release metadata](https://pypi.org/project/openpipe-art/0.5.20/).
- New capture observes both Chat Completions and Responses requests. `WebContext.search_web` calls Sol's Responses API inside `agent.generate`, so an enclosing unfiltered capture can collect teacher research requests. `from art.trajectories import no_capture` provides `with no_capture():` to exclude nested research while retaining its returned text as student tool context. Keep judges outside capture and verify every captured student request's model. [Capture scope implementation](https://github.com/OpenPipe/ART/blob/main/src/art/trajectories/_scope.py).
- No distillation/SFT/RL jobs were submitted in preflight. No trained checkpoint, measured post-training improvement, or effective model-create entitlement has yet been proven by this task.

The raw documentation OpenAPI URLs redirected to a docs access-code page. Endpoint pages ending in `.md` include their full public OpenAPI fragments and supplied the schemas above.

## Showcase links and proxy contract

The exact **documented Model Distillation UI entrypoint** is `https://forge.coreweave.com/post-train/distillation`. Open it, choose `wandb-smle`, and select the imported `bufo-gen` project. The official project/import documentation does **not** specify a per-project browser URL template. Until the actual imported project's browser URL is observed, use this documented entrypoint for the showcase link; do not derive a UI path from management API paths. [Quick start, sign-in step](https://docs.coreweave.com/model-distillation/quickstart).

For project-attributed baseline requests, the proxy uses this exact contract after routing is applied:

```python
client = AsyncOpenAI(
    base_url="https://proxy.training.coreweave.com/v1",
    api_key=required("WANDB_API_KEY"),
)
response = await client.chat.completions.create(
    model="bufo-gen",  # Actual imported proxy alias; no entity/project prefix.
    messages=complete_chat_history,
    tools=actual_function_tools,
    extra_body={"metadata": {
        "wandb.entity": "wandb-smle",
        "wandb.thread_id": stable_conversation_id,
    }},
)
```

This is `POST https://proxy.training.coreweave.com/v1/chat/completions` with a W&B bearer key. It is a **Chat Completions** interface, so a backend using `OpenAIResponsesModel` must not simply change its base URL to the proxy. Pass the complete message history every turn. The proxy forwards normal Chat Completions fields and replaces the alias with the chosen provider model. [Connect your application](https://docs.coreweave.com/model-distillation/get-started/connect-your-app).

An unversioned alias resolves **version 1**, not the newest version. `bufo-gen@v2` pins version 2 exactly. The first slash denotes a direct provider reference, such as `openai/<exact-model>`; direct provider calls bypass project routing and lack project attribution unless `wandb.project` metadata is supplied. `wandb.thread_id` keeps a conversation on one target within a routing revision. Team selection for proxy calls belongs in body metadata; management requests instead use the `Wandb-Entity` header. [Model routing](https://docs.coreweave.com/model-distillation/proxy/model-routing).

## Managed SFT identity followup

After the main implementation submitted the pilot, read-only inspection of `GET /projects/bufo-gen/finetunes/100202f8-7034-4bc2-bd2f-2375dd43fbba` observed `status: training`, ten training examples, four validation examples, and the additional field `wandb_run_id: bufo-gemma-sol-pilot-20261015`. Its `task_id` is the Model Distillation **project UUID**, `dd5ddbe2-b31c-43e1-ab39-b5488c78f3d8`, not a Serverless Training model UUID. The response had a dataset artifact but no model artifact at inspection time. The W&B run with this run ID existed with empty configuration and summary; these did not expose a model UUID. The informational GET deletion-plan also exposed no training model identity.

No public GET/list endpoint mapping a serverless model name to `model_id` was found in the official endpoint inventory or ART 0.5.20 model resource. `POST /preview/models` with `return_existing: true` is the documented name-resolution path, but it creates a model when absent; it is **not a read-only existence check**. Once deployed, inspect the exact artifact's collection and `logged_by()` provenance, and verify model/checkpoint equality before continuing. Artifact provenance may supply more identity evidence, but no such UUID mapping was established in this inspection. [Serverless endpoint inventory](https://docs.coreweave.com/products/post-training/serverless-training/api-reference), [create/return-existing contract](https://docs.coreweave.com/products/post-training/serverless-training/api-reference/models/create-model), [fine-tune GET](https://docs.coreweave.com/model-distillation/reference/management/fine-tunes/get-a-fine-tune).
