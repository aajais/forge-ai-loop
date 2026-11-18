# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo"]
# ///
"""Inspect existing curation evidence; never starts inference or training."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    return Path, json, mo


@app.cell
def _(mo):
    mo.md("""
    # Bufo: from reviewed examples to a trained checkpoint

    Inspect who reviewed each example and follow its original agent conversation.
    These are **judge-reviewed distillation examples**. They are not human approvals
    or a claim that production failures have already been used for training.

    Locally, run `python -m bufogen.workflows.demo_handoff` first. In Forge Notebooks, upload
    the resulting `runs/forge-demo/curation-lineage.json` below. This notebook only
    reads evidence; it makes no model calls and changes no reviews or routing.
    """)
    return


@app.cell
def _(mo):
    lineage_file = mo.ui.file(filetypes=[".json"], label="Curation lineage JSON")
    lineage_file
    return (lineage_file,)


@app.cell
def _(Path, json, lineage_file, mo):
    local_lineage = Path("runs/forge-demo/curation-lineage.json")
    file_bytes = lineage_file.contents()
    mo.stop(
        file_bytes is None and not local_lineage.is_file(),
        mo.md("Upload the generated curation-lineage.json to inspect the reviewed rows."),
    )
    lineage = json.loads(file_bytes if file_bytes is not None else local_lineage.read_bytes())
    families = sorted({row["family"] for row in lineage})
    reviewers = sorted({row["reviewer_kind"] for row in lineage})
    return families, lineage, reviewers


@app.cell
def _(families, mo, reviewers):
    family_filter = mo.ui.dropdown(["All", *families], value="All", label="Prompt family")
    reviewer_filter = mo.ui.dropdown(["All", *reviewers], value="All", label="Reviewer kind")
    mo.hstack([family_filter, reviewer_filter], justify="start")
    return family_filter, reviewer_filter


@app.cell
def _(family_filter, lineage, mo, reviewer_filter):
    filtered_rows = [
        row
        for row in lineage
        if (family_filter.value == "All" or row["family"] == family_filter.value)
        and (reviewer_filter.value == "All" or row["reviewer_kind"] == reviewer_filter.value)
    ]
    review_table = mo.ui.table(
        filtered_rows,
        selection="single",
        pagination=True,
        page_size=10,
        label=f"{len(filtered_rows)} reviewed examples; select a row to inspect its provenance",
    )
    review_table
    return (review_table,)


@app.cell
def _(mo, review_table):
    chosen = review_table.value
    mo.stop(
        not chosen, mo.md("Select an example above to see its review and original conversation.")
    )
    selected = chosen[0]
    mo.vstack(
        [
            mo.md("### Review evidence"),
            mo.plain_text(selected["prompt"]),
            mo.plain_text(selected.get("review_evidence") or "No review explanation recorded."),
            mo.plain_text(f"Reviewer: {selected['reviewer_kind']} · {selected.get('reviewer_id')}"),
            mo.md(f"[Open original agent conversation]({selected['source_conversation_url']})"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complete the next production-data iteration

    1. Open a production conversation with a visible, verified image failure in Agent Lens.
    2. Preserve its prompt, image, source ID and failure requirements in a review queue.
    3. Review a corrected authentic agent trajectory; retain the real reviewer identity.
    4. Export explicit accepted **train** rows to a new dataset version. Keep validation
       and held-out families out of training.
    5. Link the dataset version to its new training job and evaluate that checkpoint.

    The existing confirmation queue remains a blinded human task. This notebook does
    not fill in that queue or turn automatic scores into human reviews.
    """)
    return


if __name__ == "__main__":
    app.run()
