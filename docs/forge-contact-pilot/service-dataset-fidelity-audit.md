# Service dataset fidelity: contact SFT pilot

The actual Forge dataset artifact preserves all 24 uploaded `render_scene`
training exchanges, including the full system instructions, tool schema,
scene argument strings, tool result strings, and final assistant text.
This verifies the service-side JSON transformation, not the hosted trainer's
tokenization, loss masks, or retained sequence lengths.

## Evidence and comparison

Audited fine-tune `61b53c05-d136-4e13-9375-56517ebd647b`, whose recorded
`dataset_artifact_ref` is
`wandb-artifact:///wandb-smle/bufo-gen/distillation-dataset-61b53c05-d136-4e13-9375-56517ebd647b:v0`.
The root agent downloaded only the three manifest-listed files. Manifest
observation time: `2026-10-20T16:37:38.727270+00:00`.

| Identity | Value |
|---|---|
| Artifact ID | `QXJ0aWZhY3Q6NDA4Nzk1Mjc3OA==` |
| Artifact digest | `e7343b19c72dbe790eab43fd8fe940f0` |
| Frozen upload SHA256 | `365d75eda892534ce5c04c9451709b0bf9ef0d94fd182431e94814718c969a52` |
| Downloaded `train.jsonl` SHA256 | `5bbc32e54123d355a1b1174d827648b5e9554dd3dd1b9f33544cfe4273dea9f9` |
| System instruction UTF-8 SHA256 | `f0c4f0de2b6825605138f2a3eb1b0fa7c56231288da1b75d94317ef6f47aefe2` |
| Canonical JSON `tools` SHA256 | `4a8690e8f1ff01a1f78a40fa4485cfcb4875e056614fd9b9af34da4a01ffd480` |

All three files match the manifest's byte lengths and base64 MD5 digests:
`train.jsonl` 267,265 bytes, `val.jsonl` zero bytes, and `metadata.json`
17 bytes. The metadata contains `{"total_len":24}`. Canonical JSON hashes
use sorted keys, compact separators, Python's default ASCII escaping, and
`allow_nan=False`.

The service changed row order, so comparison joined on the exact unique user
message: 24 distinct local prompts matched 24 distinct remote prompts, with
none missing or added. A recursive comparison of each matched row's
`messages` and `tools` found exactly two differing fields:

- `messages[2].tool_calls[0].id`: original provider ID became
  `distillation_call_2_0`.
- `messages[3].tool_call_id`: changed to that same ID.

All 24 call/result links remain valid. Reusing this ID across independent
examples is not a within-conversation collision. No other nested field
differed. Each example retains the five-role sequence
`system → user → assistant(tool call) → tool → assistant(text)`.
All 24 argument strings and all 24 tool-result strings parse as JSON objects.
The complete schema and instruction hashes are uniform across the corpus.

Each service training line contains only `messages`, `tools`, and an added
`reward: 1.0`. Upload row/group IDs, split, metadata, and provenance are absent
from these lines; their audit trail remains in the frozen upload and its
source receipt. The added reward is a service serialization field, not a
new visual quality assessment or evidence of reinforcement learning.

## Training configuration and limits

The saved fine-tune response independently echoes `assistant_turns: "all"`,
six epochs, batch size two, learning rate `0.0001`, 24 training examples and
zero validation examples. Turn selection is a job setting, not a field in
the downloaded JSONL; preserving messages does not prove how the remote
trainer masks them.

The prior [tool-loss audit](../forge-gap-work/tool-loss-audit.md) measured
the pinned local ART preprocessing for the earlier SVG corpus. Those token
counts and mask results do not automatically apply to this contact corpus
or certify the hosted implementation. No local mask measurement was
repeated here, and no inference or training calls were made.

Reproduction inputs:

- `runs/forge-contact-pilot/learnability/training/upload-entries.json`
- `runs/forge-contact-pilot/learnability/training/candidate/job-observation.json`
- `runs/forge-contact-pilot/learnability/training/dataset-audit/artifact-manifest.json`
- `runs/forge-contact-pilot/learnability/training/dataset-audit/artifact/{train.jsonl,val.jsonl,metadata.json}`

The saved job's `status: training`, zero progress and null start timestamp
do not become evidence of executed optimizer updates merely because the
dataset artifact is intact. This audit rules out dropped scene payloads
in this exported dataset; it does not diagnose the current scheduling delay.
