"""Freeze development and confirmation prompts before generating targets.

This creates prompts/requirements only, never model outputs or review labels.
"""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Scene, visible action, explicit independently checkable constraints.
TRAIN = {
    "concert_stage": ("on a concert stage", "singing into a blue microphone held at his mouth", ["A hand grips the microphone", "The microphone touches or is immediately in front of the mouth", "The microphone is blue", "Stage spotlights are visible"]),
    "f1_presentation": ("giving an F1 presentation", "holding a pointer whose tip touches a red race car on a screen", ["A hand grips a pointer", "The pointer tip touches the red car on the screen", "A presentation screen and red race car are visible"]),
    "gpu_garden": ("in a GPU garden", "watering the middle of exactly three GPU seedlings with a blue watering can", ["Exactly three GPU seedlings are visible", "A hand grips the blue watering can", "Water connects the can spout to the middle seedling"]),
    "office_coffee": ("at an office desk", "holding a blue mug by its handle beside an open laptop", ["A hand contacts the mug handle", "The mug is blue", "An open laptop sits on the desk"]),
    "skate_park": ("at a skate park", "balancing with both feet on one blue skateboard above a ramp", ["Both feet contact the same skateboard", "The skateboard is blue", "A ramp is visible below the board"]),
    "pizza_kitchen": ("in a pizza kitchen", "lifting one pizza slice with a cheese strand connecting it to the rest of the pizza", ["A hand grips one pizza slice", "A visible cheese strand joins the slice to the remaining pizza", "A kitchen counter or oven is visible"]),
    "basketball_court": ("on a basketball court", "dribbling an orange basketball with a hand contacting its top below waist height", ["A hand contacts the top of the orange ball", "The ball is below Bufo's waist", "Basketball court lines are visible"]),
    "workshop_repair": ("in a robot workshop", "using a wrench to tighten a visible bolt on a blue robot", ["A hand grips the wrench", "The wrench contacts a bolt on the robot", "The robot is blue"]),
    "rainy_commute": ("on a rainy city street", "holding the shaft of a blue umbrella above his head", ["A hand grips the umbrella shaft", "The blue canopy is above the head", "Rain and city buildings are visible"]),
    "video_call": ("at a video-call desk", "waving one open hand beside an open laptop labeled HELLO", ["One open hand is raised in a wave", "An open laptop is visible", "The laptop visibly reads HELLO"]),
    "film_set": ("on a film set", "holding a black-and-white clapperboard with both hands, one on each side", ["Both hands touch opposite sides of the clapperboard", "The clapperboard is recognizable", "A film camera on a tripod is visible"]),
    "ping_pong": ("at a table-tennis table", "holding a red paddle directly behind a small white ball above the table", ["A hand grips the red paddle handle", "A white ball sits immediately in front of the paddle", "A table and central net are visible"]),
    "archery_range": ("at an archery range", "holding a bow in one hand and pulling its string with the other, with an arrow aimed at a target", ["One hand grips the bow", "The other hand touches the drawn string", "An arrow lies on the bow aimed toward a visible target"]),
    "laundry_room": ("in a laundry room", "lifting a blue shirt out of a red laundry basket with both hands", ["Both hands grip the blue shirt", "The shirt rises above a red laundry basket", "A washing machine is visible"]),
    "bakery": ("in a bakery", "using both hands to press a rolling pin onto a round piece of dough", ["Both hands grip the rolling pin", "The rolling pin contacts the dough", "A round piece of dough rests on a work surface"]),
    "tennis_court": ("on a tennis court", "holding a blue racket with its strings touching a yellow tennis ball", ["A hand grips the racket handle", "The strings contact a yellow ball", "A tennis net or court lines are visible"]),
}
VALIDATION = {
    "airport_marshaller": ("on an airport apron", "holding two orange signal batons, one in each raised hand", ["Exactly two orange batons are visible", "Each raised hand grips one baton", "An airplane is visible behind Bufo"]),
    "florist_arranging": ("in a flower shop", "inserting a red flower stem into a blue vase while holding the vase with the other hand", ["One hand grips the red flower stem", "The stem enters the blue vase", "The other hand touches the vase"]),
    "ice_cream_cart": ("beside an ice-cream cart", "holding a cone in one hand while using the other to place a pink scoop on top", ["One hand grips the cone", "The other hand holds a scoop utensil", "A pink scoop touches the top of the cone", "An ice-cream cart is visible"]),
    "locksmith": ("at a locksmith bench", "inserting a gold key into a large blue padlock held in the other hand", ["A hand grips a gold key", "The key tip enters the padlock keyhole", "The other hand holds the blue padlock"]),
}
CONFIRMATION = {
    "sushi_bar": ("at a sushi counter", "using chopsticks to lift one piece of sushi above a blue plate", ["A hand grips the chopsticks", "Both chopstick tips contact one sushi piece", "The sushi is above a blue plate"]),
    "soccer_pitch": ("on a soccer pitch", "stopping a black-and-white soccer ball under one foot", ["One foot contacts the top of the soccer ball", "The other foot is planted on the ground", "Soccer pitch markings are visible"]),
    "woodcarving": ("at a woodcarving bench", "holding a chisel against a wooden block and a mallet above the chisel handle", ["One hand grips a chisel contacting the wood", "The other hand grips a mallet", "The mallet is aligned above the chisel handle"]),
    "birdwatching": ("in a birdwatching hide", "holding binoculars against his eyes with both hands", ["Both hands touch the binoculars", "The binocular eyepieces contact the eye area", "A bird sits visibly on a nearby branch"]),
    "snowman_building": ("in a snowy yard", "pressing a carrot into the middle snowball of a three-ball snowman", ["A hand contacts the carrot", "The carrot enters the middle snowball", "Exactly three stacked snowballs form the snowman"]),
    "car_wash": ("at a car wash", "pressing a yellow sponge onto the side of a blue car", ["A hand grips a yellow sponge", "The sponge contacts the blue car", "Soap bubbles surround the contact area"]),
    "bowling_alley": ("in a bowling alley", "holding a red bowling ball against his belly with both hands", ["Both hands contact the red bowling ball", "The ball is against the belly", "Bowling pins stand at the far end of a lane"]),
    "spacewalk": ("outside a space station", "gripping a handrail with one hand while tethered to the station", ["A hand grips the station handrail", "A tether visibly connects Bufo to the station", "Stars and a planet are visible"]),
    "pancake_breakfast": ("at a breakfast stove", "holding a spatula under a pancake lifted above a frying pan", ["A hand grips the spatula", "The spatula supports the pancake", "The pancake is above a frying pan"]),
    "golf_green": ("on a golf green", "holding a putter with both hands and touching a white golf ball with its head", ["Both hands grip the putter shaft", "The putter head contacts a white ball", "A flagged hole is visible"]),
    "drum_lesson": ("in a music practice room", "holding a drumstick in each hand with one tip touching a blue drum", ["Each hand grips a drumstick", "One drumstick tip touches the blue drumhead", "The second stick is raised"]),
    "science_fair": ("at a science fair", "pouring red liquid from a tilted beaker into a blue funnel held over a flask", ["One hand grips a tilted beaker", "A continuous red stream enters a blue funnel", "The funnel leads into a flask"]),
    "sailing_dinghy": ("in a small sailing dinghy", "pulling a rope attached to the sail with both hands", ["Both hands grip the rope", "The rope visibly connects to the sail", "A small boat floats on water"]),
    "kite_flying": ("in a grassy field", "holding a spool with a taut string connected to a red kite overhead", ["A hand grips the spool", "An unbroken string connects spool to kite", "A red kite is above Bufo"]),
    "chalk_artist": ("on a city sidewalk", "kneeling and holding blue chalk against a partly drawn yellow sun", ["Bufo is kneeling", "A hand holds blue chalk touching the pavement drawing", "A yellow sun is partly drawn on the pavement"]),
    "bicycle_tuneup": ("beside a bicycle repair stand", "holding a pump handle with both hands while its hose connects to a blue bicycle tire valve", ["Both hands grip the pump handle", "A hose connects pump to tire valve", "The bicycle frame is blue"]),
}

VARIANTS = [
    ("base", "Keep the main action large and clear.", []),
    ("fullbody", "Show the whole body with both feet inside the frame.", ["The whole body and both feet are visible inside the frame"]),
    ("clothing", "Add a fitted blue vest attached to Bufo's torso; keep the head and acting hands inside the frame.", ["A blue vest is attached to the torso", "Head and acting hands are inside the frame"]),
    ("observer", "Add one smaller Bufo watching on the viewer right; only the main Bufo performs the action.", ["Exactly two Bufos are visible", "The smaller observer is to the viewer right", "Only the main Bufo performs the action"]),
    ("sign", "Add a sign at the top that reads exactly READY.", ["A top sign reads exactly READY"]),
    ("rail", "Put a low foreground rail across the lower legs; keep the face, hands and main action unobscured.", ["A foreground rail occludes part of the lower legs", "Face, hands and main action are visible"]),
    ("small", "Place Bufo on the viewer left at less than one third of the image height, keeping the action readable.", ["Bufo is on the viewer left", "Bufo is less than one third of image height"]),
    ("minimal", "Use simple flat colors and a pale backdrop, retaining every required scene element. Keep the whole figure connected.", ["The scene uses flat colors", "Bufo's head, torso and limbs form one connected figure"]),
]

def rows(scenes, split, variants):
    for family, (scene, action, checks) in scenes.items():
        for variant, extra, more in variants:
            # The lower-leg rail would contradict the foot/ball contact requirement.
            # Keep it out of confirmation; training examples with feet actions use a sign instead.
            if variant == "rail" and family == "skate_park":
                extra = "Put a low rail behind the ramp, with the board and both feet fully visible."
                more = ["A low rail is behind the ramp", "Board and both feet remain visible"]
            requirements = [*checks, *more]
            yield {"id":f"gap-{split}-{family}-{variant}", "family":family, "split":split,
                   "prompt":f"Draw Bufo {scene}, {action}. {extra} Visible details: " + "; ".join(requirements) + ".",
                   "requirements":requirements}

def main():
    original = yaml.safe_load((ROOT / "config/forge_prompts.yaml").read_text())
    old_test = {p["family"] for p in original if p["split"] == "heldout"}
    assert not set(TRAIN) & (set(VALIDATION) | set(CONFIRMATION) | old_test)
    assert not set(VALIDATION) & (set(CONFIRMATION) | old_test)
    assert not set(CONFIRMATION) & old_test
    data = [*rows(TRAIN,"train",VARIANTS), *rows(VALIDATION,"validation",VARIANTS[:6]),
            *rows(CONFIRMATION,"heldout",VARIANTS[:5])]
    assert len(data) == 232 and len({p["id"] for p in data}) == len(data)
    encoded = yaml.safe_dump(data,sort_keys=False,allow_unicode=True,width=100)
    path = ROOT / "config/forge_gap_prompts.yaml"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("Refusing to overwrite a frozen curriculum")
    path.write_text(encoded)
    output = ROOT / "runs/forge-gap"
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"schema_version":1,"source":str(path.relative_to(ROOT)),
                "sha256":hashlib.sha256(encoded.encode()).hexdigest(),
                "counts":{"train":128,"validation":24,"heldout":80},
                "families":{"train":sorted(TRAIN),"validation":sorted(VALIDATION),
                            "confirmation":sorted(CONFIRMATION),"old_test_excluded":sorted(old_test)},
                "confirmation_outputs_generated":False,
                "selection_rule":"Select on validation strict pass count; break ties by requirement coverage then fewer generation errors. Open confirmation only after selection."}
    (output / "curriculum-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest["counts"]))

if __name__ == "__main__":
    main()
