# SVG tool-call supervision audit

The local data and pinned ART preprocessing include the SVG geometry in the
training target. This audit found no evidence that the upload accidentally
trained only the short final assistant reply. It cannot certify the hosted
Forge trainer's implementation or effective sequence limit.

## Actual uploaded data

Both curriculum SFT jobs use the direct Forge REST path in
`src/bufogen/forge.py`, rather than invoking the installed ART client's
`train_sft` method. `reviewed_entries` deep-copies complete `messages` and
`tools`; `upload_entries` submits those entries unchanged. The actual job
receipts request `assistant_turns: all`, batch size 2, and respectively one and
three epochs.

Reconstructed the upload using the existing `_source` implementation and
verified both hashes against the saved e1 state:

- Source: `runs/forge-gap/training/reviewed-examples.jsonl`
- Source SHA256: `e8f2f4b23c30ae73bd8fd394ac340e0b6d299f3c277307b70ce1d2d3dc746984`
- Upload SHA256: `87bff51d3698176f09ed16c494e08bdfc7412687380af83351dcf72d5e693273`
- 147 accepted examples: 124 train and 23 validation.
- 151 assistant `render_svg` tool calls, each retaining its SVG argument.
- Saved Forge upload acknowledgements report all 147 entries created, revision
  2, with the same split counts. This is upload evidence, not a downloaded
  verification of the service's final tokenized training artifact.

## Local mask measurement

Executed the pinned ART response-mask function and normalization helpers on all
147 actual upload entries, using the already cached Gemma4 tokenizer and an
isolated Transformers installation. No package was installed, no source was
changed, no model weights were loaded, and PyTorch was not imported.

For each entry the audit:

1. Normalized JSON tool-call argument strings to mappings using ART's helper.
2. Rendered the complete messages and tool schemas with Gemma4's template and
   ART defaults `preserve_thinking=True, enable_thinking=False`.
3. Verified offset-bearing text tokenization exactly matched the template's
   tokenized output.
4. Applied the installed `response_only_labels` with the auto-detected markers
   `<|turn>user\n` and `<|turn>model\n`.
5. Found every SVG payload verbatim inside its rendered `render_svg` call and
   checked all overlapping token positions for non-ignored labels.

**Every measured SVG target token was supervised: 325,790 included, zero
masked, with zero missing SVG payloads.** These SVG positions comprise about
95.5% of all supervised positions in this local calculation. This count
includes tokens crossing SVG string boundaries; it is a positional coverage
audit, not a claim about per-character gradient weights.

| Tokens per complete example | Minimum | Median | P95 | Maximum | Total |
|---|---:|---:|---:|---:|---:|
| Full sequence | 3,799 | 4,836 | 5,633 | 6,563 | 714,172 |
| Supervised positions | 1,242 | 2,298 | 3,085 | 4,034 | 340,977 |
| `render_svg` call regions | 1,174 | 2,228 | 3,011 | 3,908 | 329,879 |
| SVG argument regions | 1,147 | 2,203 | 2,983 | 3,852 | 325,790 |
| Final assistant text, encoded separately | 14 | 21 | 25 | 30 | 3,071 |

P95 uses the sorted observation at index `floor(.95 * (n - 1))`.

The pinned local `assistant_turns=all` path masks by response/user markers,
rather than by OpenAI message role alone. Gemma4 embeds tool responses within
the model turn, so those tool response regions also remain supervised until
the next user marker. This does not exclude SVG tokens. The remote service
could use different masking, and this audit did not inspect remote labels.

## Sequence limits and update-count constraint

Pinned ART's `tokenize_sft_batch` drops entire examples exceeding
`max_seq_length`; it does not truncate their SVG tails. Its local backend forms
batches first, removes oversized examples inside tokenization, and then skips
batches with no supervised tokens. Packing preserves labels without
truncation.

| Hypothetical full-sequence limit | Examples above limit, all splits | Training examples above limit |
|---|---:|---:|
| 4,096 | 139 / 147 | 116 / 124 |
| 6,144 | 3 / 147 | 3 / 124 |
| 8,192 | 0 / 147 | 0 / 124 |
| 16,384 | 0 / 147 | 0 / 124 |
| 32,768 | 0 / 147 | 0 / 124 |

The cached model config advertises 262,144 positions. Neither that model limit
nor these hypothetical thresholds establishes the hosted trainer's limit.
The submitted job configuration contains no explicit maximum sequence length.
Saved Forge state/latest receipts expose no token counts, dropped-example
counts, or effective sequence limit.

Actual saved W&B histories contain **62 optimizer updates for e1 and 186 for
e3**, matching 124 training examples at batch size 2 across one or three
epochs. Under the inspected local ART dropping implementation, a 4,096-token
limit would retain only eight training examples and yield at most eight
nonempty batches per epoch, even if they were spread across separate original
batches. Consequently, **62 real updates per epoch is inconsistent with that
specific 4,096-token whole-example-dropping hypothesis**. It does not rule out
a different remote truncation policy or partial drops: removing three examples
at 6,144 could still leave all 62 original batches nonempty.

The saved histories expose gradient step, training loss and learning rate,
with epoch aggregate loss rows. They do not expose retained token counts.
The root agent subsequently completed supported read-only Forge and W&B API
queries, recorded at `2026-10-15T19:31:35.889532+00:00` in
`runs/forge-gap/training-runtime-field-audit.json`. Both Forge job responses
explicitly report `training_example_count: 124` and
`validation_example_count: 23`; both return the requested all-turn training
configuration. Neither response exposes maximum sequence length, dropped
examples, token counts, trainer version, or a training-job ID. W&B configs
contain our model display/source metadata and epoch count; config and summary
contain no additional sequence/drop/token fields. The example counts support
dataset coverage but are not a documented per-token retention certificate.

Initial direct reads from this subagent failed transport and its W&B MCP
query returned `AuthenticationError`; the root's successful read supersedes
that access limitation. No guessed endpoint or mutation was attempted. The
installed ART SDK has no model runtime-config retrieval method. Its training
event reader requires a real training-job ID; the Forge `task_id` must not be
treated as that ID without an explicit API contract.

## Supported conclusion and remaining verification

The upload contains the geometry, `assistant_turns=all` was requested, and the
locally inspected pinned implementation supervises all its SVG tokens. Loss
improvement therefore should not be explained as merely learning “Here you
go” without contrary remote evidence. Lower token loss alone still does not
measure hand contact, connected anatomy, or prompt satisfaction.

To certify the hosted objective, obtain the service's effective tokenizer and
template revision, masking policy, maximum sequence length, retained/dropped
example counts, and supervised token counts—or a redacted token/label sample
for one of these actual generation IDs. Preserve the distinction between
these requested remote facts and the measured local behavior above.

## Implementation references and reproduction identity

- ART source: `45a17cbd2a85cc0333172e652fc1211c6ae3f3da`.
- [SFT tokenization and over-length handling](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/preprocessing/tokenize.py#L1002).
- [Response-only labels](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/preprocessing/response_masking.py).
- [Local backend batch handling](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/local/backend.py#L1895).
- [Gemma4 marker detection](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/utils/model_config.py).
- [ART serverless SFT serialization, a separate client path](https://github.com/OpenPipe/ART/blob/45a17cbd2a85cc0333172e652fc1211c6ae3f3da/src/art/serverless/backend.py#L535).
- Cached tokenizer location: `/private/tmp/bufo-gemma4-tokenizer`.
- Cached template SHA256: `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`.
- Tokenizer JSON SHA256: `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
- Installed response-mask source SHA256: `c8b16a8591c085083a014ee057b5d269d99bd325d2378d4dfc61c367bdb4f571`.
- Actual service evidence: `runs/forge-gap/training/e1/forge-state.json`,
  `runs/forge-gap/training/e3/forge-state.json`, and their respective
  `wandb-metrics.json` histories, plus the live read in
  `runs/forge-gap/training-runtime-field-audit.json`.

These implementation references were inspected from the installed pinned
source. They are not evidence of which commit Forge deployed remotely.
