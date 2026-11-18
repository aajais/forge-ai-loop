"""Persist real image outputs alongside native agent conversations for review."""

import weave
from PIL import Image

from bufogen.agents.prompting import style_references


class SavedGenerationScore(weave.Scorer):
    """Replay explicit saved evidence on its source, with no inference fallback.

    Source calls precede reference selection and scoring, so their original outputs
    cannot satisfy a reference judge's inference-cache checks. The saved payload retains
    the original rubric and reference provenance; this scorer is clearly named Saved*.
    """

    @weave.op()
    def score(self, output: dict, saved_score: dict | None = None) -> dict:
        if not saved_score:
            raise ValueError("An explicit saved score is required")
        source = output.get("generation", output)
        if not source.get("id") or saved_score.get("generation_id") != source["id"]:
            raise ValueError("Saved score does not match source generation")
        return saved_score


def loaded_image(path):
    with Image.open(path) as image:
        return image.copy()


@weave.op()
def visual_generation(prompt: str, generation: dict) -> dict:
    candidate, image_error = None, None
    if generation.get("png_path"):
        try:
            candidate = loaded_image(generation["png_path"])
        except (OSError, ValueError) as exc:
            image_error = type(exc).__name__
    return {
        "image": candidate,
        "image_error": image_error,
        "references": [loaded_image(path) for path in style_references()],
        "generation": generation,
    }


def record_generation(row):
    with weave.attributes(
        {
            "conversation_id": row.group_id,
            "prompt_id": row.prompt_id,
            "backend": row.backend,
            "prompt_version": row.prompt_version,
        }
    ):
        _, call = visual_generation.call(
            prompt=row.prompt, generation=row.to_dict(), __should_raise=True
        )
    if call is not None:
        row.weave_call_id = call.id
    return row


def create_scene_review_queue(rows, *, name="Bufo scene calibration v1") -> dict:
    """Create a distinct scene queue, preserving existing identity queues and annotations."""
    return create_review_queue(rows, name=name, scene=True)


def create_review_queue(rows, *, name="Bufo identity calibration v1", scene=False) -> dict:
    from weave.trace_server.interface.builtin_object_classes.annotation_spec import AnnotationSpec

    from bufogen.integrations.telemetry import init_weave
    from bufogen.settings import project_path

    client = init_weave()
    fields = {
        "bufo_present": {"type": "boolean"},
        "bufo_similarity": {"type": "integer", "minimum": 0, "maximum": 4},
        "prompt_alignment": {"type": "integer", "minimum": 0, "maximum": 4},
        "failure_reason": {"type": "string"},
    }
    if scene:
        fields.update(
            {
                "anatomy": {"type": "integer", "minimum": 0, "maximum": 4},
                "composition": {"type": "integer", "minimum": 0, "maximum": 4},
                "readability": {"type": "integer", "minimum": 0, "maximum": 4},
                "requirements_satisfied": {"type": "boolean"},
                "acceptance": {"type": "string", "enum": ["accept", "reject", "needs_review"]},
            }
        )
    refs = [
        weave.publish(
            AnnotationSpec(name="bufo-review-" + field, field_schema=schema),
            name="bufo-review-" + field,
        ).uri()
        for field, schema in fields.items()
    ]
    existing = [q for q in client.list_annotation_queues(name=name) if q.name == name]
    queue_id = (
        existing[0].id
        if existing
        else client.create_annotation_queue(
            name=name,
            scorer_refs=refs,
            description=(
                "Compare the candidate image with the original bufo references. Presence means the "
                "specific bufo character, not any green frog. Similarity: 0 unrelated, 1 generic frog, "
                "2 partly resembles bufo, 3 recognizable bufo, 4 faithful identity. Judge facial "
                "proportions, raised eyes, wide muzzle, jaw and stout body; allow new poses and scenes. "
                "Prompt alignment: 0 unrelated through 4 satisfies the request. Record concrete failures. "
                "For scene review, anatomy, composition and readability use 0 unusable, 1 major "
                "errors, 2 mixed, 3 minor errors, 4 clear and coherent. Compare every explicit "
                "requirement against pixels; text labels do not prove actions. Acceptance is a "
                "review decision, not an automatic threshold. Automatic scene scores are uncalibrated."
            ),
        )
    )
    call_ids = []
    for row in rows:
        if not row.weave_call_id:
            record_generation(row)
        if row.weave_call_id:
            call_ids.append(row.weave_call_id)
    if call_ids:
        client.add_calls_to_annotation_queue(
            queue_id,
            call_ids=call_ids,
            display_fields=[
                "inputs.prompt",
                "output.generation.benchmark_metadata",
                "output.image",
                "output.references",
                "output.generation.conversation_url",
            ],
        )
    return {
        "queue_id": queue_id,
        "call_ids": call_ids,
        "url": f"https://wandb.ai/{project_path()}/weave/annotation-queues/{queue_id}",
    }
