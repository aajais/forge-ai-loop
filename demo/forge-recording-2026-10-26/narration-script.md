# Bufo · CoreWeave Forge demo narration

Synthesized voice script; recording and audio generation are not complete.

## CoreWeave Forge · The Bufo AI loop

This is Bufo, an agent that creates new scenes around a recognizable frog character. The demo follows the AI loop in CoreWeave Forge: run, observe, curate, improve, and evaluate. We will use recorded validation conversations and real training results to show what improved, and what still needs work.

## Run + Observe · Inspect the complete agent turn

Start with this baseline conversation. The request includes two orange signal batons, an airplane behind Bufo, and a smaller observer on the right. The trace preserves the prompt, planning and rendering tools, final response, and generated image. The image renders successfully, but the saved visual judge finds that the airplane is not clearly established. A successful tool call is not the same as a successful task.

## Compare behavior · Same prompt, fine-tuned Qwen

Now open the same prompt from the one epoch fine tuned Qwen checkpoint. Its saved verdict passes. We can inspect the actual image and tool sequence, rather than relying on a friendly final sentence. This pair makes the behavior change concrete. It is an illustrative example; the full validation cohort is what measures the improvement.

## Curate · Preserve teacher trajectories and lineage

The shared training dataset contains nine hundred and eleven reviewed teacher trajectories from GPT six point one Sol. Each example retains the user request, planning tool, drawing tool arguments, tool results, and final response. Training uses all assistant turns, so the model learns the behavior that produces the drawing, not just its caption. Human annotation queues provide a separate review step; the blinded confirmation queue remains reserved for evaluation.

## Improve · Managed distillation and an immutable checkpoint

In Forge Post Training, the deployed student is Qwen three point eight, twenty seven B. This job uses one epoch, batch size two, and a learning rate of zero point zero zero zero one. The page connects the training dataset, Weights and Biases run, and immutable version one model artifact. A separate three epoch experiment lets us test whether more training actually helps.

## Evaluate · Measure image quality on 48 validation prompts

Here is the saved image evaluation comparison. On the same forty eight validation prompts, base Qwen passes seventeen, while the one epoch student passes thirty: thirty five point four percent to sixty two point five percent. The three epoch student passes twenty seven, so additional epochs do not win this comparison. We inspect requirement coverage, character similarity, render validity, and latency together. These are validation results, not a final release claim. The historical Sol reference used different runtime settings, so this does not establish Sol parity.

## Close the loop · Configure serving and continue observing

Routing connects the candidate back to serving. The Qwen target now uses temperature zero point seven, twenty thousand tokens, and thinking enabled, matching the recorded generation settings. Conversation identity keeps related requests together, and trace sampling is set to one hundred percent. The traffic weights are preserved. A fresh proxy smoke test, verified pricing, and final human confirmation are still pending. The result is a connected improvement workflow, with evidence guiding the next iteration.
