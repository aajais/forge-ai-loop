"""Build config/distill_prompts.yaml: the forge-gap set plus 24 extra validation prompts.

Usage: uv run python scripts/make_distill_prompts.py

Deterministic. Refuses to overwrite an existing output whose content differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config" / "forge_gap_prompts.yaml"
OUTPUT = ROOT / "config" / "distill_prompts.yaml"

# family -> variant -> (scene description, requirements)
NEW = {
    "airport_marshaller": {
        "twohands": (
            "Draw Bufo on an airport apron, wearing a yellow hard hat and "
            "crossing two orange signal batons in front of his chest, one "
            "held in each hand, with an airplane parked behind him.",
            [
                "Each hand grips one orange baton",
                "The two batons cross in an X shape",
                "Bufo wears a yellow hard hat",
                "An airplane is visible behind Bufo",
            ],
        ),
        "nightscene": (
            "Draw Bufo on a night-time airport apron, raising two "
            "orange signal batons, one in each hand, under a dark sky "
            "with a yellow crescent moon.",
            [
                "Each raised hand grips one baton",
                "Both batons are orange",
                "The sky is dark",
                "A yellow crescent moon is visible",
            ],
        ),
        "caption": (
            "Draw Bufo on an airport apron, waving one orange baton beside "
            'a small yellow board that reads "GATE 4".',
            [
                "One hand grips an orange baton",
                "A yellow board reads exactly GATE 4",
                "The board is to the viewer right of Bufo",
                "An airplane is visible behind Bufo",
            ],
        ),
        "closeup": (
            "Draw a close-up of Bufo's head and shoulders on an airport "
            "apron, one orange baton held high in the hand on the viewer's "
            "right with the baton tip well inside the frame; no feet are "
            "shown.",
            [
                "One hand grips an orange baton",
                "The baton is held in the hand on the viewer's right",
                "The baton tip is inside the frame",
                "Only the head and shoulders are visible, with no feet",
            ],
        ),
        "pair": (
            "Draw Bufo on an airport apron, raising one orange baton, with a "
            "second smaller Bufo in a yellow vest standing on the viewer left "
            "holding a red flag.",
            [
                "Two separate Bufo characters appear in the scene",
                "The smaller Bufo is to the viewer left",
                "The smaller Bufo wears a yellow vest",
                "The main Bufo grips an orange baton",
            ],
        ),
        "weather": (
            "Draw Bufo on a rainy airport apron, wearing a yellow rain hat "
            "and holding two orange signal batons raised while rain falls "
            "from a grey cloud.",
            [
                "Each raised hand grips one baton",
                "A grey cloud is above Bufo",
                "Blue raindrops are visible",
                "Bufo wears a yellow rain hat",
            ],
        ),
    },
    "florist_arranging": {
        "twohands": (
            "Draw Bufo in a flower shop, holding a yellow flower in each "
            "hand with the flower heads pointing up above a green vase on "
            "a table, with a shelf of pots behind him.",
            [
                "Each hand grips one yellow flower stem",
                "The flower heads point upward",
                "A green vase is on the table below the flowers",
                "A shelf of pots is visible behind Bufo",
            ],
        ),
        "nightscene": (
            "Draw Bufo in a flower shop at night, placing a pink flower "
            "into a vase beside a lit yellow lamp and a dark window with "
            "stars.",
            [
                "One hand grips the pink flower stem",
                "A lit yellow lamp is visible",
                "Stars are visible through the window",
                "The stem enters the vase",
            ],
        ),
        "caption": (
            "Draw Bufo in a flower shop, placing a red flower into a blue "
            'vase below a small hanging card that reads "FRESH".',
            [
                "One hand grips the red flower stem",
                "The stem enters the blue vase",
                "A hanging card reads exactly FRESH",
                "The card is above Bufo's head",
            ],
        ),
        "closeup": (
            "Draw a close-up of Bufo's hands and face in a flower shop, one "
            "hand holding a white flower stem inside the rim of a blue "
            "vase.",
            [
                "One hand grips a white flower stem",
                "The stem enters the blue vase",
                "Bufo's face is visible above the vase",
                "The vase and hand are inside the frame",
            ],
        ),
        "pair": (
            "Draw Bufo in a flower shop, handing a red flower to a second "
            "smaller Bufo standing on the viewer right who holds out a basket.",
            [
                "Two separate Bufo characters appear in the scene",
                "The smaller Bufo is to the viewer right",
                "The main Bufo's hand holds the red flower",
                "The smaller Bufo holds a basket",
            ],
        ),
        "weather": (
            "Draw Bufo at an open-air flower stall on a sunny day under a "
            "striped awning, with three orange flowers standing in a vase "
            "and one more orange flower held in his hand.",
            [
                "One hand grips an orange flower stem",
                "Three orange flowers stand in the vase",
                "A yellow sun is visible in the sky",
                "A striped awning is above the stall",
            ],
        ),
    },
    "ice_cream_cart": {
        "twohands": (
            "Draw Bufo beside an ice-cream cart, holding one cone in each "
            "hand, a pink scoop on one cone and a green scoop on the other.",
            [
                "Each hand grips one cone",
                "One scoop is pink and the other is green",
                "Each cone has one scoop on top",
                "An ice-cream cart is visible",
            ],
        ),
        "nightscene": (
            "Draw Bufo beside an ice-cream cart at night, holding a cone "
            "with a pink scoop under a glowing string of yellow lights and "
            "a dark sky.",
            [
                "One hand grips the cone",
                "A pink scoop sits on the cone",
                "A string of yellow lights glows above the cart",
                "The sky is dark",
            ],
        ),
        "caption": (
            "Draw Bufo beside an ice-cream cart, holding a green scoop on a "
            'cone, with a small cart banner that reads "ICE".',
            [
                "One hand grips the cone",
                "A green scoop sits on the cone",
                "The cart banner reads exactly ICE",
                "An ice-cream cart is visible",
            ],
        ),
        "closeup": (
            "Draw a close-up of Bufo's face with closed eyes and one hand "
            "holding a cone with a large pink scoop, the scoop touching Bufo's "
            "open mouth.",
            [
                "One hand grips the cone",
                "The pink scoop touches Bufo's open mouth",
                "Bufo's eyes are closed",
                "The cone and hand are inside the frame",
            ],
        ),
        "pair": (
            "Draw Bufo behind an ice-cream cart, reaching over the cart counter "
            "to hand a cone to a second smaller Bufo standing on the viewer left.",
            [
                "Two separate Bufo characters appear in the scene",
                "The smaller Bufo is to the viewer left",
                "The main Bufo's hand holds the cone over the counter",
                "An ice-cream cart is between the two Bufos",
            ],
        ),
        "weather": (
            "Draw Bufo beside an ice-cream cart on a hot day, holding a cone "
            "with a blue scoop while a yellow sun shines and the scoop drips.",
            [
                "One hand grips the cone",
                "A blue scoop sits on the cone",
                "A yellow sun is visible in the sky",
                "A drip falls from the scoop",
            ],
        ),
    },
    "locksmith": {
        "twohands": (
            "Draw Bufo at a locksmith bench, holding a gold key in one hand and a "
            "silver key in the other, both raised.",
            [
                "One hand grips a gold key",
                "The other hand grips a silver key",
                "Exactly two keys are visible",
                "A wooden workbench is visible",
            ],
        ),
        "nightscene": (
            "Draw Bufo at a locksmith bench at night, turning a gold key in a "
            "blue padlock held in his other hand, lit by a yellow desk lamp "
            "beside a window showing a crescent moon.",
            [
                "A hand grips a gold key",
                "The key tip enters the padlock keyhole",
                "The other hand holds the blue padlock",
                "A yellow desk lamp is lit",
                "A crescent moon is visible in the window",
            ],
        ),
        "caption": (
            "Draw Bufo at a locksmith bench, holding up a gold key beside a small "
            'red tag that reads "OPEN".',
            [
                "A hand grips a gold key",
                "A red tag reads exactly OPEN",
                "The tag is to the viewer right of the key",
                "A padlock is visible on the bench",
            ],
        ),
        "closeup": (
            "Draw a close-up of Bufo's hand pushing a gold key into a blue padlock, "
            "with Bufo's face visible behind the padlock.",
            [
                "A hand grips a gold key",
                "The key tip enters the padlock keyhole",
                "The padlock is blue",
                "Bufo's face is visible behind the padlock",
            ],
        ),
        "pair": (
            "Draw Bufo at a locksmith bench, handing a gold key to a second smaller "
            "Bufo standing on the viewer right who holds a blue padlock.",
            [
                "Two separate Bufo characters appear in the scene",
                "The smaller Bufo is to the viewer right",
                "The main Bufo's hand holds the gold key",
                "The smaller Bufo holds a blue padlock",
            ],
        ),
        "weather": (
            "Draw Bufo outside a locksmith shop on a snowy day, inserting a gold "
            "key into a blue padlock hanging from a metal hasp on the door while "
            "white snowflakes fall and snow lies on the ground.",
            [
                "A hand grips a gold key",
                "The padlock hangs from a metal hasp on the door",
                "White snowflakes are visible",
                "Snow lies on the ground",
            ],
        ),
    },
}


def build() -> list[dict]:
    rows = yaml.safe_load(SOURCE.read_text())
    for family, variants in NEW.items():
        for variant, (scene, requirements) in variants.items():
            rows.append(
                {
                    "id": f"gap-validation-{family}-{variant}",
                    "family": family,
                    "split": "validation",
                    "prompt": f"{scene} Visible details: {'; '.join(requirements)}.",
                    "requirements": list(requirements),
                }
            )
    return rows


def main() -> int:
    text = yaml.safe_dump(build(), sort_keys=False, allow_unicode=True, width=100)
    if OUTPUT.exists():
        if OUTPUT.read_text() == text:
            print(f"{OUTPUT} already up to date")
            return 0
        print(f"Refusing to overwrite differing {OUTPUT}", file=sys.stderr)
        return 1
    OUTPUT.write_text(text)
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
