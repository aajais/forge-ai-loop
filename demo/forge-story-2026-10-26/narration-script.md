# Bufo: improving an agent with Forge

Synthesized narration. Recorded app examples and validation results; no claim of live generation.

## Meet Bufo | A character, any scene

Meet Bufo. Describe a scene, and this agent plans a drawing, renders it, and returns an image you can use as an emoji. You can change models or continue the conversation to refine the result. The challenge is consistency: Bufo should stay recognizable, and every requested detail should appear. A convincing response does not tell us whether the image actually meets those requirements.

## Why Forge | From a visual failure to evidence

Here is a concrete failure from our validation set. We asked for an airport marshaller holding two orange batons, an airplane behind Bufo, and a smaller observer on the right. The baseline rendered an image, but the airplane is not clearly established. To improve this app, we need to connect that failure to its execution, turn useful examples into training data, and measure whether a new model fixes the behavior. That is the Forge loop.

## Observe | Understand what the agent did

Agent Lens gives us the complete conversation and trace. We can follow the request through planning and rendering to the final PNG, and inspect which model produced it. This separates execution problems from visual quality problems. Our saved image evaluations check character similarity and requested details; a successful render alone is not enough. The trace gives us the context we need to investigate and curate examples.

## Curate | Teach the behavior behind the image

For this iteration, we collected and reviewed teacher examples from GPT six point one Sol. This dataset contains nine hundred and eleven trajectories, including the request, planning, drawing tool arguments, tool results, and final response. Keeping the trajectory matters: the student must learn how to construct the scene, not just write a caption. These reviews include automated and assistant review. The separate blinded human confirmation queue has not yet been completed.

## Improve | Distill, train, and track the candidate

Forge Post Training connects that dataset to managed distillation. Our student is Qwen three point eight, twenty seven B. The selected run trains all assistant turns for one epoch, with batch size two and learning rate zero point zero zero zero one. The dataset, training run, and versioned checkpoint stay linked. We also tested three epochs and a reinforcement learning update. Neither beat the selected checkpoint on aggregate validation passes. Each experiment has to earn its place in the app.

## Evaluate | Did the change actually help?

On forty eight validation prompts, base Qwen passed seventeen. The one epoch student passed thirty: an increase from thirty five point four to sixty two point five percent. Three epochs passed twenty seven; reinforcement learning matched thirty. We compare requirement coverage, character similarity, render validity, and latency. This is a measured improvement, with failures still remaining. It does not establish Sol parity: that historical reference used different runtime settings. Human confirmation remains the next check.

## Inspect the result | Same prompt, better completion

Return to the airport example. The fine tuned checkpoint now produces a recognizable airplane behind Bufo, along with the batons and smaller observer. Its saved verdict passes. This example makes the improvement visible, while the full evaluation tells us how broadly it holds. Both the drawing and the trace remain available, so we can challenge the score instead of taking it on faith.

## Back to Bufo | The next iteration starts here

Back in Bufo, the fine tuned model is available alongside the baseline, using its exact model name. We can try scenes and keep refining the conversation. New failures give us the next examples to investigate. That is how this project uses Forge: run the app, observe its behavior, curate useful trajectories, improve the model, and evaluate the result before advancing it. The loop stays grounded in the image the user actually receives.
