"""Append 128 augmented train prompts (8 per train family) to config/distill_prompts.yaml.

Usage: uv run python scripts/augment_prompts.py --out config/distill_prompts.yaml --seed 20261025

Deterministic, template based, no model calls. Each augmented prompt is derived from the family's
`gap-train-<family>-base` prompt with one changed element (a prop colour, a count, an exact sign
text, a pose or a setting detail). Existing entries are never touched: the generated block is only
appended, and a second run with the same seed is a no-op. Also writes the provenance table.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = "stage-b-augment-2026-10"
FRAMINGS = (
    "Keep the main action large and clear.",
    "Keep the main action large and easy to read.",
    "Keep the action large and clear.",
)


def _t(kind, scene, replace=None, add=()):
    return {"kind": kind, "scene": scene, "replace": replace or {}, "add": list(add)}


# family -> (base scene prefix words are inside each scene) -> eight single-change variants.
TEMPLATES = {
    "concert_stage": [
        _t(
            "colour",
            "Draw Bufo on a concert stage, singing into a red microphone held at his mouth.",
            {"The microphone is blue": "The microphone is red"},
        ),
        _t(
            "colour",
            "Draw Bufo on a concert stage, singing into a green microphone held at his mouth.",
            {"The microphone is blue": "The microphone is green"},
        ),
        _t(
            "count",
            "Draw Bufo on a concert stage, singing into a blue microphone held at his mouth, "
            "under exactly three stage spotlights.",
            {"Stage spotlights are visible": "Exactly three stage spotlights are visible"},
        ),
        _t(
            "sign",
            "Draw Bufo on a concert stage, singing into a blue microphone held at his mouth. "
            "Add a banner above the stage that reads exactly ENCORE.",
            add=["A banner above the stage reads exactly ENCORE"],
        ),
        _t(
            "pose",
            "Draw Bufo on a concert stage, singing into a blue microphone held at his mouth "
            "while his other arm is raised high.",
            add=["The free arm is raised above shoulder height"],
        ),
        _t(
            "setting",
            "Draw Bufo on a concert stage with a drum kit behind him, singing into a blue "
            "microphone held at his mouth.",
            add=["A drum kit is visible behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo on a concert stage in front of a red curtain, singing into a blue "
            "microphone held at his mouth.",
            add=["A red curtain is visible behind Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo on a concert stage, singing into a blue microphone held at his mouth. "
            "Add a banner above the stage that reads exactly LIVE.",
            add=["A banner above the stage reads exactly LIVE"],
        ),
    ],
    "f1_presentation": [
        _t(
            "colour",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a blue "
            "race car on a screen.",
            {
                "The pointer tip touches the red car on the screen": "The pointer tip touches the blue car on the screen",
                "A presentation screen and red race car are visible": "A presentation screen and blue race car are visible",
            },
        ),
        _t(
            "colour",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a yellow "
            "race car on a screen.",
            {
                "The pointer tip touches the red car on the screen": "The pointer tip touches the yellow car on the screen",
                "A presentation screen and red race car are visible": "A presentation screen and yellow race car are visible",
            },
        ),
        _t(
            "count",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches the left "
            "one of exactly two red race cars on a screen.",
            {
                "The pointer tip touches the red car on the screen": "The pointer tip touches the left car on the screen",
                "A presentation screen and red race car are visible": "A presentation screen shows exactly two red race cars",
            },
        ),
        _t(
            "sign",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a red "
            "race car on a screen, with a title above the screen that reads exactly POLE.",
            add=["A title above the screen reads exactly POLE"],
        ),
        _t(
            "pose",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a red "
            "race car on a screen while his other hand rests on his hip.",
            add=["The free hand rests on the hip"],
        ),
        _t(
            "setting",
            "Draw Bufo giving an F1 presentation beside a podium, holding a pointer whose tip "
            "touches a red race car on a screen.",
            add=["A podium is visible beside Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a red "
            "race car on a screen, with a checkered flag hanging beside the screen.",
            add=["A checkered flag is visible beside the screen"],
        ),
        _t(
            "sign",
            "Draw Bufo giving an F1 presentation, holding a pointer whose tip touches a red "
            "race car on a screen, with a title above the screen that reads exactly LAP 1.",
            add=["A title above the screen reads exactly LAP 1"],
        ),
    ],
    "gpu_garden": [
        _t(
            "colour",
            "Draw Bufo in a GPU garden, watering the middle of exactly three GPU seedlings "
            "with a red watering can.",
            {"A hand grips the blue watering can": "A hand grips the red watering can"},
        ),
        _t(
            "colour",
            "Draw Bufo in a GPU garden, watering the middle of exactly three GPU seedlings "
            "with a green watering can.",
            {"A hand grips the blue watering can": "A hand grips the green watering can"},
        ),
        _t(
            "count",
            "Draw Bufo in a GPU garden, watering the middle of exactly five GPU seedlings "
            "with a blue watering can.",
            {"Exactly three GPU seedlings are visible": "Exactly five GPU seedlings are visible"},
        ),
        _t(
            "sign",
            "Draw Bufo in a GPU garden, watering the middle of exactly three GPU seedlings "
            "with a blue watering can. Plant a wooden sign in the soil that reads exactly ROW 7.",
            add=["A wooden sign in the soil reads exactly ROW 7"],
        ),
        _t(
            "pose",
            "Draw Bufo in a GPU garden, kneeling to water the middle of exactly three GPU "
            "seedlings with a blue watering can.",
            add=["Bufo kneels beside the seedlings"],
        ),
        _t(
            "setting",
            "Draw Bufo in a GPU garden with a wooden fence behind him, watering the middle of "
            "exactly three GPU seedlings with a blue watering can.",
            add=["A wooden fence is visible behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo in a GPU garden under a bright sun, watering the middle of exactly "
            "three GPU seedlings with a blue watering can.",
            add=["The sun shines above the garden"],
        ),
        _t(
            "sign",
            "Draw Bufo in a GPU garden, watering the middle of exactly three GPU seedlings "
            "with a blue watering can. Plant a wooden sign in the soil that reads exactly TENDED.",
            add=["A wooden sign in the soil reads exactly TENDED"],
        ),
    ],
    "office_coffee": [
        _t(
            "colour",
            "Draw Bufo at an office desk, holding a red mug by its handle beside an open laptop.",
            {"The mug is blue": "The mug is red"},
        ),
        _t(
            "colour",
            "Draw Bufo at an office desk, holding a green mug by its handle beside an open laptop.",
            {"The mug is blue": "The mug is green"},
        ),
        _t(
            "count",
            "Draw Bufo at an office desk, holding a blue mug by its handle beside an open "
            "laptop and a stack of exactly two books.",
            add=["Exactly two books are stacked on the desk"],
        ),
        _t(
            "sign",
            "Draw Bufo at an office desk, holding a blue mug by its handle beside an open "
            "laptop with a sticky note on it that reads exactly MONDAY.",
            add=["A sticky note on the laptop reads exactly MONDAY"],
        ),
        _t(
            "pose",
            "Draw Bufo at an office desk, leaning back in his chair and holding a blue mug by "
            "its handle beside an open laptop.",
            add=["Bufo leans back in a chair"],
        ),
        _t(
            "setting",
            "Draw Bufo at an office desk with a potted plant on it, holding a blue mug by its "
            "handle beside an open laptop.",
            add=["A potted plant sits on the desk"],
        ),
        _t(
            "setting",
            "Draw Bufo at an office desk with a desk lamp, holding a blue mug by its handle "
            "beside an open laptop.",
            add=["A desk lamp is visible on the desk"],
        ),
        _t(
            "sign",
            "Draw Bufo at an office desk, holding a blue mug by its handle beside an open "
            "laptop with a sticky note on it that reads exactly FRIDAY.",
            add=["A sticky note on the laptop reads exactly FRIDAY"],
        ),
    ],
    "skate_park": [
        _t(
            "colour",
            "Draw Bufo at a skate park, balancing with both feet on one red skateboard above a ramp.",
            {"The skateboard is blue": "The skateboard is red"},
        ),
        _t(
            "colour",
            "Draw Bufo at a skate park, balancing with both feet on one green skateboard above a ramp.",
            {"The skateboard is blue": "The skateboard is green"},
        ),
        _t(
            "count",
            "Draw Bufo at a skate park, balancing with both feet on one blue skateboard above "
            "a ramp with exactly two orange cones beside it.",
            add=["Exactly two orange cones are visible"],
        ),
        _t(
            "sign",
            "Draw Bufo at a skate park, balancing with both feet on one blue skateboard above "
            "a ramp, with a banner behind him that reads exactly DROP IN.",
            add=["A banner behind Bufo reads exactly DROP IN"],
        ),
        _t(
            "pose",
            "Draw Bufo at a skate park, balancing with both feet on one blue skateboard above "
            "a ramp with both arms stretched out.",
            add=["Both arms are stretched out to the sides"],
        ),
        _t(
            "setting",
            "Draw Bufo at a skate park with a chain-link fence behind him, balancing with both "
            "feet on one blue skateboard above a ramp.",
            add=["A chain-link fence is visible behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo at a skate park in front of a graffiti wall, balancing with both feet "
            "on one blue skateboard above a ramp.",
            add=["A wall with colourful graffiti is visible behind Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo at a skate park, balancing with both feet on one blue skateboard above "
            "a ramp, with a banner behind him that reads exactly SKATE.",
            add=["A banner behind Bufo reads exactly SKATE"],
        ),
    ],
    "pizza_kitchen": [
        _t(
            "colour",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice from a pizza on a red plate "
            "with a cheese strand connecting it to the rest of the pizza.",
            add=["The pizza sits on a red plate"],
        ),
        _t(
            "colour",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice from a pizza on a green "
            "tray with a cheese strand connecting it to the rest of the pizza.",
            add=["The pizza sits on a green tray"],
        ),
        _t(
            "count",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice topped with exactly three "
            "olives, with a cheese strand connecting it to the rest of the pizza.",
            add=["Exactly three olives sit on the lifted slice"],
        ),
        _t(
            "sign",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice with a cheese strand "
            "connecting it to the rest of the pizza, with a chalkboard menu that reads "
            "exactly MARGHERITA.",
            add=["A chalkboard menu reads exactly MARGHERITA"],
        ),
        _t(
            "pose",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice high above his head with a "
            "cheese strand connecting it to the rest of the pizza.",
            add=["The slice is held above head height"],
        ),
        _t(
            "setting",
            "Draw Bufo in a pizza kitchen in front of a brick oven with visible flames, "
            "lifting one pizza slice with a cheese strand connecting it to the rest of the pizza.",
            add=["A brick oven with visible flames is behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo in a pizza kitchen with a rolling pin lying on the counter, lifting one "
            "pizza slice with a cheese strand connecting it to the rest of the pizza.",
            add=["A rolling pin lies on the counter"],
        ),
        _t(
            "sign",
            "Draw Bufo in a pizza kitchen, lifting one pizza slice with a cheese strand "
            "connecting it to the rest of the pizza, with a chalkboard menu that reads exactly HOT.",
            add=["A chalkboard menu reads exactly HOT"],
        ),
    ],
    "basketball_court": [
        _t(
            "colour",
            "Draw Bufo on a basketball court, dribbling a purple basketball with a hand "
            "contacting its top below waist height.",
            {
                "A hand contacts the top of the orange ball": "A hand contacts the top of the purple ball"
            },
        ),
        _t(
            "colour",
            "Draw Bufo on a basketball court, dribbling a white basketball with a hand "
            "contacting its top below waist height.",
            {
                "A hand contacts the top of the orange ball": "A hand contacts the top of the white ball"
            },
        ),
        _t(
            "count",
            "Draw Bufo on a basketball court, dribbling an orange basketball with a hand "
            "contacting its top below waist height, with a second orange ball resting on the floor.",
            add=["Exactly two basketballs are visible"],
        ),
        _t(
            "sign",
            "Draw Bufo on a basketball court, dribbling an orange basketball with a hand "
            "contacting its top below waist height, under a scoreboard that reads exactly 21.",
            add=["A scoreboard above the court reads exactly 21"],
        ),
        _t(
            "pose",
            "Draw Bufo on a basketball court, crouching low while dribbling an orange "
            "basketball with a hand contacting its top below waist height.",
            add=["Bufo crouches with bent knees"],
        ),
        _t(
            "setting",
            "Draw Bufo on a basketball court with a hoop and net behind him, dribbling an "
            "orange basketball with a hand contacting its top below waist height.",
            add=["A basketball hoop with a net is visible behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo on a basketball court in front of wooden bleachers, dribbling an orange "
            "basketball with a hand contacting its top below waist height.",
            add=["Wooden bleachers are visible behind Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo on a basketball court, dribbling an orange basketball with a hand "
            "contacting its top below waist height, under a scoreboard that reads exactly HOME.",
            add=["A scoreboard above the court reads exactly HOME"],
        ),
    ],
    "workshop_repair": [
        _t(
            "colour",
            "Draw Bufo in a robot workshop, using a wrench to tighten a visible bolt on a red robot.",
            {"The robot is blue": "The robot is red"},
        ),
        _t(
            "colour",
            "Draw Bufo in a robot workshop, using a wrench to tighten a visible bolt on a green robot.",
            {"The robot is blue": "The robot is green"},
        ),
        _t(
            "count",
            "Draw Bufo in a robot workshop, using a wrench to tighten a visible bolt on a blue "
            "robot that has exactly two antennas.",
            add=["The robot has exactly two antennas"],
        ),
        _t(
            "sign",
            "Draw Bufo in a robot workshop, using a wrench to tighten a visible bolt on a blue "
            "robot, with a sign above the workbench that reads exactly REPAIRS.",
            add=["A sign above the workbench reads exactly REPAIRS"],
        ),
        _t(
            "pose",
            "Draw Bufo in a robot workshop, kneeling beside a blue robot and using a wrench to "
            "tighten a visible bolt on it.",
            add=["Bufo is kneeling beside the robot"],
        ),
        _t(
            "setting",
            "Draw Bufo in a robot workshop with a pegboard of tools on the wall, using a "
            "wrench to tighten a visible bolt on a blue robot.",
            add=["A pegboard of tools is visible on the wall"],
        ),
        _t(
            "setting",
            "Draw Bufo in a robot workshop with a workbench beside him, using a wrench to "
            "tighten a visible bolt on a blue robot.",
            add=["A workbench is visible beside Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo in a robot workshop, using a wrench to tighten a visible bolt on a blue "
            "robot, with a sign above the workbench that reads exactly OPEN.",
            add=["A sign above the workbench reads exactly OPEN"],
        ),
    ],
    "rainy_commute": [
        _t(
            "colour",
            "Draw Bufo on a rainy city street, holding the shaft of a red umbrella above his head.",
            {"The blue canopy is above the head": "The red canopy is above the head"},
        ),
        _t(
            "colour",
            "Draw Bufo on a rainy city street, holding the shaft of a yellow umbrella above his head.",
            {"The blue canopy is above the head": "The yellow canopy is above the head"},
        ),
        _t(
            "count",
            "Draw Bufo on a rainy city street with exactly three puddles, holding the shaft "
            "of a blue umbrella above his head.",
            add=["Exactly three puddles are visible on the street"],
        ),
        _t(
            "sign",
            "Draw Bufo on a rainy city street beside a bus stop sign that reads exactly "
            "ROUTE 9, holding the shaft of a blue umbrella above his head.",
            add=["A bus stop sign reads exactly ROUTE 9"],
        ),
        _t(
            "pose",
            "Draw Bufo on a rainy city street, walking mid-stride and holding the shaft of a "
            "blue umbrella above his head.",
            add=["Bufo is mid-stride with one foot lifted"],
        ),
        _t(
            "setting",
            "Draw Bufo on a rainy city street with a yellow taxi passing, holding the shaft "
            "of a blue umbrella above his head.",
            add=["A yellow taxi is visible on the street"],
        ),
        _t(
            "setting",
            "Draw Bufo on a rainy city street beside a glowing street lamp, holding the shaft "
            "of a blue umbrella above his head.",
            add=["A glowing street lamp is visible beside Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo on a rainy city street beside a bus stop sign that reads exactly "
            "ROUTE 42, holding the shaft of a blue umbrella above his head.",
            add=["A bus stop sign reads exactly ROUTE 42"],
        ),
    ],
    "video_call": [
        _t(
            "sign",
            "Draw Bufo at a video-call desk, waving one open hand beside an open laptop labeled HI.",
            {"The laptop visibly reads HELLO": "The laptop visibly reads HI"},
        ),
        _t(
            "sign",
            "Draw Bufo at a video-call desk, waving one open hand beside an open laptop labeled BYE.",
            {"The laptop visibly reads HELLO": "The laptop visibly reads BYE"},
        ),
        _t(
            "colour",
            "Draw Bufo at a video-call desk wearing a red headset, waving one open hand beside "
            "an open laptop labeled HELLO.",
            add=["Bufo wears a red headset"],
        ),
        _t(
            "count",
            "Draw Bufo at a video-call desk with exactly two potted plants, waving one open "
            "hand beside an open laptop labeled HELLO.",
            add=["Exactly two potted plants are visible"],
        ),
        _t(
            "pose",
            "Draw Bufo at a video-call desk, waving both open hands beside an open laptop "
            "labeled HELLO.",
            {"One open hand is raised in a wave": "Both open hands are raised in a wave"},
        ),
        _t(
            "setting",
            "Draw Bufo at a video-call desk with a desk lamp, waving one open hand beside an "
            "open laptop labeled HELLO.",
            add=["A desk lamp is visible on the desk"],
        ),
        _t(
            "setting",
            "Draw Bufo at a video-call desk with a coffee mug on it, waving one open hand "
            "beside an open laptop labeled HELLO.",
            add=["A mug sits on the desk"],
        ),
        _t(
            "colour",
            "Draw Bufo at a video-call desk, waving one open hand beside an open yellow laptop "
            "labeled HELLO.",
            {"An open laptop is visible": "An open yellow laptop is visible"},
        ),
    ],
    "film_set": [
        _t(
            "colour",
            "Draw Bufo on a film set, holding a black-and-white clapperboard with both hands, "
            "one on each side, beside a red film camera on a tripod.",
            {"A film camera on a tripod is visible": "A red film camera on a tripod is visible"},
        ),
        _t(
            "colour",
            "Draw Bufo on a film set, holding a black-and-white clapperboard with both hands, "
            "one on each side, beside a blue film camera on a tripod.",
            {"A film camera on a tripod is visible": "A blue film camera on a tripod is visible"},
        ),
        _t(
            "count",
            "Draw Bufo on a film set lit by exactly two studio lights, holding a "
            "black-and-white clapperboard with both hands, one on each side.",
            add=["Exactly two studio lights are visible"],
        ),
        _t(
            "sign",
            "Draw Bufo on a film set, holding a black-and-white clapperboard with both hands, "
            "one on each side, under a sign that reads exactly QUIET.",
            add=["A sign above the set reads exactly QUIET"],
        ),
        _t(
            "pose",
            "Draw Bufo on a film set, raising a black-and-white clapperboard above his head "
            "with both hands, one on each side.",
            add=["The clapperboard is held above head height"],
        ),
        _t(
            "setting",
            "Draw Bufo on a film set with a director's chair beside him, holding a "
            "black-and-white clapperboard with both hands, one on each side.",
            add=["A director's chair is visible beside Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo on a film set with a boom microphone overhead, holding a "
            "black-and-white clapperboard with both hands, one on each side.",
            add=["A boom microphone is visible above Bufo"],
        ),
        _t(
            "sign",
            "Draw Bufo on a film set, holding a black-and-white clapperboard with both hands, "
            "one on each side, under a sign that reads exactly ACTION.",
            add=["A sign above the set reads exactly ACTION"],
        ),
    ],
    "ping_pong": [
        _t(
            "colour",
            "Draw Bufo at a table-tennis table, holding a blue paddle directly behind a small "
            "white ball above the table.",
            {"A hand grips the red paddle handle": "A hand grips the blue paddle handle"},
        ),
        _t(
            "colour",
            "Draw Bufo at a table-tennis table, holding a green paddle directly behind a small "
            "white ball above the table.",
            {"A hand grips the red paddle handle": "A hand grips the green paddle handle"},
        ),
        _t(
            "count",
            "Draw Bufo at a table-tennis table, holding a red paddle directly behind a small "
            "white ball above the table, with a second white ball resting on the table.",
            add=["Exactly two white balls are visible"],
        ),
        _t(
            "sign",
            "Draw Bufo at a table-tennis table, holding a red paddle directly behind a small "
            "white ball above the table, under a scoreboard that reads exactly 11.",
            add=["A scoreboard above the table reads exactly 11"],
        ),
        _t(
            "pose",
            "Draw Bufo at a table-tennis table, lunging forward and holding a red paddle "
            "directly behind a small white ball above the table.",
            add=["Bufo leans forward in a lunge"],
        ),
        _t(
            "setting",
            "Draw Bufo at a table-tennis table with a water bottle beside it, holding a red "
            "paddle directly behind a small white ball above the table.",
            add=["A water bottle stands beside the table"],
        ),
        _t(
            "setting",
            "Draw Bufo at a table-tennis table in front of a low barrier, holding a red paddle "
            "directly behind a small white ball above the table.",
            add=["A low barrier is visible behind the table"],
        ),
        _t(
            "sign",
            "Draw Bufo at a table-tennis table, holding a red paddle directly behind a small "
            "white ball above the table, under a scoreboard that reads exactly SERVE.",
            add=["A scoreboard above the table reads exactly SERVE"],
        ),
    ],
    "archery_range": [
        _t(
            "colour",
            "Draw Bufo at an archery range, holding a red bow in one hand and pulling its "
            "string with the other, with an arrow aimed at a target.",
            add=["The bow is red"],
        ),
        _t(
            "colour",
            "Draw Bufo at an archery range, holding a green bow in one hand and pulling its "
            "string with the other, with an arrow aimed at a target.",
            add=["The bow is green"],
        ),
        _t(
            "count",
            "Draw Bufo at an archery range, holding a bow in one hand and pulling its string "
            "with the other, with an arrow aimed at a target that already has exactly two "
            "arrows stuck in it.",
            add=["Exactly two arrows are stuck in the target"],
        ),
        _t(
            "sign",
            "Draw Bufo at an archery range, holding a bow in one hand and pulling its string "
            "with the other, with an arrow aimed at a target beside a sign that reads exactly 20 M.",
            add=["A sign beside the target reads exactly 20 M"],
        ),
        _t(
            "pose",
            "Draw Bufo at an archery range, kneeling on one knee, holding a bow in one hand "
            "and pulling its string with the other, with an arrow aimed at a target.",
            add=["Bufo kneels on one knee"],
        ),
        _t(
            "setting",
            "Draw Bufo at an archery range, holding a bow in one hand and pulling its string "
            "with the other, with an arrow aimed at a target mounted on a hay bale.",
            add=["The target is mounted on a hay bale"],
        ),
        _t(
            "setting",
            "Draw Bufo at an archery range with a flag flying at its edge, holding a bow in "
            "one hand and pulling its string with the other, with an arrow aimed at a target.",
            add=["A flag is visible at the edge of the range"],
        ),
        _t(
            "sign",
            "Draw Bufo at an archery range, holding a bow in one hand and pulling its string "
            "with the other, with an arrow aimed at a target beside a sign that reads exactly WIND.",
            add=["A sign beside the target reads exactly WIND"],
        ),
    ],
    "laundry_room": [
        _t(
            "colour",
            "Draw Bufo in a laundry room, lifting a green shirt out of a red laundry basket "
            "with both hands.",
            {"Both hands grip the blue shirt": "Both hands grip the green shirt"},
        ),
        _t(
            "colour",
            "Draw Bufo in a laundry room, lifting a yellow shirt out of a red laundry basket "
            "with both hands.",
            {"Both hands grip the blue shirt": "Both hands grip the yellow shirt"},
        ),
        _t(
            "count",
            "Draw Bufo in a laundry room with exactly two socks on the floor, lifting a blue "
            "shirt out of a red laundry basket with both hands.",
            add=["Exactly two socks lie on the floor"],
        ),
        _t(
            "sign",
            "Draw Bufo in a laundry room, lifting a blue shirt out of a red laundry basket "
            "with both hands, beside a wall sign that reads exactly WASH.",
            add=["A wall sign reads exactly WASH"],
        ),
        _t(
            "pose",
            "Draw Bufo in a laundry room, lifting a blue shirt high overhead out of a red "
            "laundry basket with both hands.",
            add=["The shirt is held above head height"],
        ),
        _t(
            "setting",
            "Draw Bufo in a laundry room with a shelf of detergent bottles on the wall, "
            "lifting a blue shirt out of a red laundry basket with both hands.",
            add=["A shelf of detergent bottles is visible on the wall"],
        ),
        _t(
            "setting",
            "Draw Bufo in a laundry room with a dryer next to the washing machine, lifting a "
            "blue shirt out of a red laundry basket with both hands.",
            add=["A dryer is visible beside the washing machine"],
        ),
        _t(
            "sign",
            "Draw Bufo in a laundry room, lifting a blue shirt out of a red laundry basket "
            "with both hands, beside a wall sign that reads exactly FOLD.",
            add=["A wall sign reads exactly FOLD"],
        ),
    ],
    "bakery": [
        _t(
            "colour",
            "Draw Bufo in a bakery, using both hands to press a red rolling pin onto a round "
            "piece of dough.",
            add=["The rolling pin is red"],
        ),
        _t(
            "colour",
            "Draw Bufo in a bakery, using both hands to press a blue rolling pin onto a round "
            "piece of dough.",
            add=["The rolling pin is blue"],
        ),
        _t(
            "count",
            "Draw Bufo in a bakery with exactly three loaves of bread on a shelf, using both "
            "hands to press a rolling pin onto a round piece of dough.",
            add=["Exactly three loaves of bread are visible on a shelf"],
        ),
        _t(
            "sign",
            "Draw Bufo in a bakery, using both hands to press a rolling pin onto a round "
            "piece of dough, beside a chalkboard that reads exactly FRESH.",
            add=["A chalkboard reads exactly FRESH"],
        ),
        _t(
            "pose",
            "Draw Bufo in a bakery, leaning forward over the table and using both hands to "
            "press a rolling pin onto a round piece of dough.",
            add=["Bufo leans forward over the work surface"],
        ),
        _t(
            "setting",
            "Draw Bufo in a bakery with an oven behind him, using both hands to press a "
            "rolling pin onto a round piece of dough.",
            add=["An oven is visible behind Bufo"],
        ),
        _t(
            "setting",
            "Draw Bufo in a bakery with a bag of flour on the table, using both hands to "
            "press a rolling pin onto a round piece of dough.",
            add=["A bag of flour sits on the work surface"],
        ),
        _t(
            "sign",
            "Draw Bufo in a bakery, using both hands to press a rolling pin onto a round "
            "piece of dough, beside a chalkboard that reads exactly WARM.",
            add=["A chalkboard reads exactly WARM"],
        ),
    ],
    "tennis_court": [
        _t(
            "colour",
            "Draw Bufo on a tennis court, holding a red racket with its strings touching a "
            "yellow tennis ball.",
            add=["The racket is red"],
        ),
        _t(
            "colour",
            "Draw Bufo on a tennis court, holding a green racket with its strings touching a "
            "yellow tennis ball.",
            add=["The racket is green"],
        ),
        _t(
            "count",
            "Draw Bufo on a tennis court, holding a blue racket with its strings touching a "
            "yellow tennis ball, with a second ball on the ground.",
            add=["Exactly two tennis balls are visible"],
        ),
        _t(
            "sign",
            "Draw Bufo on a tennis court, holding a blue racket with its strings touching a "
            "yellow tennis ball, under a scoreboard that reads exactly 40.",
            add=["A scoreboard reads exactly 40"],
        ),
        _t(
            "pose",
            "Draw Bufo on a tennis court, holding a blue racket out to his side with its "
            "strings touching a yellow tennis ball.",
            add=["The racket arm is extended to the side"],
        ),
        _t(
            "setting",
            "Draw Bufo on a tennis court with an umpire chair at its edge, holding a blue "
            "racket with its strings touching a yellow tennis ball.",
            add=["An umpire chair is visible at the court edge"],
        ),
        _t(
            "setting",
            "Draw Bufo on a tennis court under a blue sky with white clouds, holding a blue "
            "racket with its strings touching a yellow tennis ball.",
            add=["White clouds are visible in the sky"],
        ),
        _t(
            "sign",
            "Draw Bufo on a tennis court, holding a blue racket with its strings touching a "
            "yellow tennis ball, under a scoreboard that reads exactly LOVE.",
            add=["A scoreboard reads exactly LOVE"],
        ),
    ],
}


def build(existing: list[dict], seed: int) -> list[dict]:
    by_id = {row["id"]: row for row in existing}
    rng = random.Random(seed)
    families = []
    for row in existing:
        if row["split"] == "train" and row["family"] not in families:
            families.append(row["family"])
    if set(families) != set(TEMPLATES):
        raise SystemExit(
            f"template families differ from train families: {set(families) ^ set(TEMPLATES)}"
        )
    out = []
    for family in families:
        base_id = f"gap-train-{family}-base"
        base = by_id[base_id]
        for number, template in enumerate(TEMPLATES[family], start=1):
            requirements = []
            unused = dict(template["replace"])
            for requirement in base["requirements"]:
                requirements.append(unused.pop(requirement, requirement))
            if unused:
                raise SystemExit(f"{family} {number}: replace keys not in base: {sorted(unused)}")
            requirements += template["add"]
            framing = rng.choice(FRAMINGS)
            out.append(
                {
                    "id": f"aug-train-{family}-{number:02d}",
                    "family": family,
                    "split": "train",
                    "prompt": f"{template['scene']} {framing} Visible details: "
                    + "; ".join(requirements)
                    + ".",
                    "requirements": requirements,
                    "provenance": PROVENANCE,
                    "augmented_from": base_id,
                    "change": template["kind"],
                }
            )
    return out


def _dump(rows: list[dict]) -> str:
    return yaml.safe_dump(
        [{k: v for k, v in r.items() if k != "change"} for r in rows],
        sort_keys=False,
        width=100,
        allow_unicode=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--docs", type=Path, default=ROOT / "docs" / "distill" / "augmented-prompts.md"
    )
    args = parser.parse_args(argv)
    text = args.out.read_text()
    rows = yaml.safe_load(text)
    present = [r for r in rows if r["id"].startswith("aug-train-")]
    base_rows = [r for r in rows if not r["id"].startswith("aug-train-")]
    generated = build(base_rows, args.seed)
    block = _dump(generated)
    if present:
        if [{k: v for k, v in g.items() if k != "change"} for g in generated] != present:
            raise SystemExit(
                "existing augmented prompts differ from generated ones; not overwriting"
            )
    else:
        args.out.write_text(text + ("" if text.endswith("\n") else "\n") + block)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    lines = [
        "# Augmented train prompts (Stage B)",
        "",
        f"- Generator: `scripts/augment_prompts.py --seed {args.seed}` (templates per family, no model calls)",
        f"- Provenance tag: `{PROVENANCE}`",
        f"- Prompts added: {len(generated)} (8 per train family, ids `aug-train-<family>-<nn>`)",
        f"- sha256 of `config/distill_prompts.yaml` with the augmented block: `{digest}`",
        "- Every prompt changes exactly one element of the family's base prompt.",
        "",
        "| id | derived from | changed element | requirements |",
        "|---|---|---|---|",
    ]
    for g in generated:
        lines.append(
            f"| {g['id']} | {g['augmented_from']} | {g['change']} | {len(g['requirements'])} |"
        )
    args.docs.parent.mkdir(parents=True, exist_ok=True)
    args.docs.write_text("\n".join(lines) + "\n")
    print(f"{len(generated)} augmented prompts; sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
