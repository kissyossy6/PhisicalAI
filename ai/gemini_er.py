"""Thin wrapper around Gemini Robotics-ER (gemini-robotics-er-2-preview) for the tidy-up task.

The robot never gets object coordinates from the simulator: every "what / where"
comes from these calls on the camera image.

  plan()        task orchestration: instruction + image -> ordered pick/place steps
  locate()      per step, in one call: success detection of the previous step +
                2D pointing of the next object + spatial reasoning (free spot)
  point() / free_spot() / verify()   the same three capabilities as separate calls
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field

from PIL import Image

MODEL_ID = os.environ.get("GEMINI_ER_MODEL", "gemini-robotics-er-2-preview")

POINT_FMT = "Points are in [y, x] format normalized to 0-1000."


def parse_json(text: str):
    """Pull the first JSON object/array out of a model response."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.JSONDecoder().raw_decode(text[start:])[0]


@dataclass
class Call:
    kind: str
    prompt: str
    response: str
    seconds: float
    parsed: object = None


@dataclass
class GeminiER:
    api_key: str | None = None
    model: str = MODEL_ID
    log: list[Call] = field(default_factory=list)

    def __post_init__(self):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=self.api_key or os.environ["GEMINI_API_KEY"])

    # ------------------------------------------------------------------ core
    def _ask(self, kind: str, img: Image.Image, prompt: str, thinking_budget: int = 0, retries: int = 3):
        types = self._types
        config = types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )
        last_err = None
        attempt = 0
        while attempt < retries:
            t0 = time.time()
            try:
                res = self.client.models.generate_content(model=self.model, contents=[img, prompt], config=config)
                text = res.text or ""
                parsed = parse_json(text)
                self.log.append(Call(kind, prompt, text, time.time() - t0, parsed))
                return parsed
            except Exception as e:
                last_err = e
                m = re.search(r"retry in ([\d.]+)s", str(e))
                if "429" in str(e) and m and float(m.group(1)) < 180:
                    # free tier rate limit: the simulation is paused, so waiting costs nothing in the video
                    print(f"  (rate limited, waiting {float(m.group(1)):.0f}s)")
                    time.sleep(float(m.group(1)) + 1)
                    continue
                if "429" in str(e):  # daily quota: waiting won't help
                    break
                attempt += 1  # network hiccup or malformed JSON -> ask again
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"Gemini {kind} failed: {last_err}")

    # ------------------------------------------------------------ capabilities
    def plan(self, img: Image.Image, instruction: str, containers: list[str]):
        prompt = textwrap.dedent(f"""\
            You are the high-level planner of a single-arm robot (parallel gripper) that
            works on the table in the image. The robot can only do one primitive:
            pick_and_place(object, destination).

            User instruction (may be Japanese): "{instruction}"

            Available destinations: {", ".join(containers)}.

            1. List every movable item on the table that is NOT already inside a
               destination, with a short unique English label (e.g. "red cube", "apple").
            2. Decide, following the instruction and common sense, which destination each
               item should go to. Items the instruction does not ask to move get "stay".
            3. Order the steps so that the robot can execute them one by one.

            Answer only with JSON:
            {{
              "summary_ja": "<1 sentence in Japanese: how you interpreted the instruction>",
              "items": [{{"label": "<label>", "point": [y, x], "category": "<what it is>",
                          "destination": "<one of the destinations or stay>",
                          "reason_ja": "<short reason in Japanese>"}}],
              "destinations": [{{"label": "<destination>", "point": [y, x]}}],
              "steps": [{{"object": "<item label>", "destination": "<destination>"}}]
            }}
            {POINT_FMT}""")
        return self._ask("plan", img, prompt, thinking_budget=2048)

    def locate(self, img: Image.Image, obj: str, container: str, prev: tuple[str, str] | None = None):
        """One call per step: check the previous step, find the next object and a free spot.

        Merging verify + point + free spot keeps a full run within the free tier
        (20 requests/day for gemini-robotics-er-2-preview).
        """
        check = (
            f'First, look carefully: is the {prev[0]} now resting inside the {prev[1]}? '
            'Put the answer in "previous_success" (true/false) and "previous_reason_ja" '
            "(a short Japanese reason).\n"
            if prev
            else 'Set "previous_success" to null and "previous_reason_ja" to "".\n'
        )
        prompt = textwrap.dedent(f"""\
            A robot is tidying the table in the image.
            {check}
            Next, the robot will pick up the {obj} and put it into the {container}.
            - "object": point to the {obj} that is not inside any container. If it is not
              visible, use null.
            - "spot": point to one empty spot inside the {container} where the {obj} fits
              without touching items already there and without touching the rim.
            Answer only with JSON:
            {{"previous_success": ..., "previous_reason_ja": "...",
              "object": {{"point": [y, x], "label": "{obj}"}},
              "spot": {{"point": [y, x], "label": "spot for {obj}"}}}}
            {POINT_FMT}""")
        return self._ask("locate", img, prompt, thinking_budget=512)

    def point(self, img: Image.Image, label: str):
        prompt = textwrap.dedent(f"""\
            Point to the {label} on the table. If there are several, point to the one that
            is not inside any container. If it is not visible, return [].
            Answer only with JSON: [{{"point": [y, x], "label": "{label}"}}]
            {POINT_FMT}""")
        res = self._ask("point", img, prompt)
        return res[0] if isinstance(res, list) and res else None

    def free_spot(self, img: Image.Image, container: str, obj: str):
        prompt = textwrap.dedent(f"""\
            The robot will put the {obj} into the {container}. Point to one empty spot
            inside the {container} where the {obj} fits without touching the items that
            are already there and without touching the rim. Prefer spots near the
            centre if it is free.
            Answer only with JSON: [{{"point": [y, x], "label": "spot for {obj}"}}]
            {POINT_FMT}""")
        res = self._ask("free_spot", img, prompt, thinking_budget=512)
        return res[0] if isinstance(res, list) and res else None

    def verify(self, img: Image.Image, obj: str, container: str):
        prompt = textwrap.dedent(f"""\
            Look carefully. Is the {obj} now resting inside the {container}?
            Answer only with JSON: {{"success": true or false, "reason_ja": "<short Japanese reason>"}}""")
        return self._ask("verify", img, prompt, thinking_budget=512)


class ReplayER(GeminiER):
    """Feeds back the responses recorded in a previous run's log.json, in order.

    The simulation is deterministic, so this reproduces the run exactly (used to
    re-render the video without spending API quota). No live inference happens.
    """

    def __init__(self, log_path: str):
        self.log = []
        self._recorded = json.load(open(log_path))["gemini_calls"]

    def _ask(self, kind, img, prompt, thinking_budget=0, retries=3):
        rec = self._recorded[len(self.log)]
        assert rec["kind"] == kind, f"replay diverged: expected {rec['kind']}, got {kind}"
        parsed = parse_json(rec["response"])
        self.log.append(Call(kind, prompt, rec["response"], rec["seconds"], parsed))
        return parsed
