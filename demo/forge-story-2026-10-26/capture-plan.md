# Capture plan — app-first Forge story

Use normal Terminal for macOS screen capture and local synthesized narration. Keep Chrome visible on the recorded display. No microphone audio is recorded; no uploads.

1. Bufo app: open http://127.0.0.1:8765/ and show the existing cloud-rider baseline conversation. Show the prompt, PNG, model name, and composer. Capture 25–35 seconds.
2. Show model choices briefly; keep exact model names visible. No generation wait is needed: these are recorded conversations.
3. Agent Lens: inspect the airport-marshaller baseline trace. Expand planning/rendering tools if possible; retain the final image and the prompt. Capture 25–35 seconds to replace the reused Observe shot.
4. Return to Bufo and open the existing cloud-rider ft-Qwen/Qwen3.8-27B conversation. Show the model/checkpoint identity and image, then the composer to establish continued iteration. Capture 25–35 seconds.
5. Stop the recorder with q. Use receipt/progress timing to choose clean ranges in shots.json. Verify every range visually before marking verified=true.

Previously verified footage supplies Curate, Improve, Evaluate, and the matched airport result. New footage provides the opening and closing app scenes. All app outputs are existing conversations, not presented as live generation.

Run assemble.py only after audio and all ranges are ready. It refuses missing or unverified scenes. Decode the result, inspect chapter frames, and verify audio before delivery.
