"""Natural-language tidy-up with Gemini Robotics-ER 1.6 + MuJoCo Franka Panda.

    python run_task.py --instruction "食べ物はお皿に、おもちゃは箱に片付けて" --seed 2 --disturb

Loop:  see -> plan (Gemini) -> for each step: point (Gemini) -> depth refine -> pick
       -> free spot (Gemini) -> place -> verify (Gemini) -> retry if needed
       -> at the end look again and re-plan until nothing is left.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import imageio.v2 as imageio
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from ai.gemini_er import GeminiER
from ai.overlay import compose_frame, draw_badge, draw_plan, draw_points
from sim.env import ITEM_BODIES, TidyEnv

CONTAINERS = {
    # label shown to Gemini -> (center xy, floor z, usable radius)
    "white plate": (np.array([0.45, -0.38]), 0.013, 0.07),
    "orange toy box": (np.array([0.45, 0.38]), 0.008, 0.07),
}
TITLE = "Gemini Robotics-ER 1.6 × MuJoCo ― ことばで片付けるロボット"


def container_of(label: str) -> str | None:
    label = (label or "").lower()
    for name in CONTAINERS:
        if name in label or name.split()[-1] in label:
            return name
    return None


def ground_truth(env: TidyEnv) -> dict:
    """Where every item really is (only for evaluation, never shown to the robot)."""
    res = {}
    for b in ITEM_BODIES:
        p = env.body_pos(b)
        res[b] = "table"
        for name, (c, _, _) in CONTAINERS.items():
            if np.linalg.norm(p[:2] - c) < 0.11 and p[2] > -0.1:
                res[b] = name
        if p[2] < -0.1:
            res[b] = "floor"
    return res


class Director:
    """Owns the video: every simulation frame is composed with the current captions."""

    def __init__(self, env: TidyEnv, path: str, instruction: str):
        self.env = env
        self.writer = imageio.get_writer(path, fps=25, quality=8, macro_block_size=8)
        self.pip: Image.Image | None = None
        self.caption, self.sub, self.tag = "", "", ""
        self.instruction = instruction
        self.n_frames = 0
        env.frame_callback = self._on_frame

    def _on_frame(self, env: TidyEnv):
        frame = compose_frame(env.render_video(), self.pip, TITLE, self.caption, self.sub, self.tag)
        self.writer.append_data(frame)
        self.n_frames += 1

    def say(self, caption: str, sub: str = "", hold: float = 0.0, pip: Image.Image | None = None):
        self.caption, self.sub = caption, sub
        if pip is not None:
            self.pip = pip
        print(f"[{self.n_frames / 25:6.1f}s] {caption} | {sub}")
        if hold:
            self.env.wait(hold)

    def close(self):
        self.writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="机の上を片付けて。食べ物はお皿に、おもちゃは箱にしまってね。")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--disturb", action="store_true", help="a 'human' puts one item back on the table mid-task")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    os.makedirs(args.out, exist_ok=True)
    env = TidyEnv()
    env.reset(seed=args.seed)
    gem = GeminiER()
    director = Director(env, os.path.join(args.out, "video.mp4"), args.instruction)
    snap_id = 0

    def look():
        return Image.fromarray(env.render_rgb()), env.render_depth()

    def save(img: Image.Image, name: str):
        nonlocal snap_id
        snap_id += 1
        img.save(os.path.join(args.out, f"{snap_id:02d}_{name}.png"))

    stats = {"steps": [], "replans": 0, "grasp_retries": 0}
    t_start = time.time()

    # ---------------------------------------------------------------- intro
    img, _ = look()
    director.tag = "指示: " + args.instruction
    director.say("ロボットへの指示は、自然言語ひとつだけ", "物体の座標はシミュレータから一切もらわず、カメラ画像だけで判断します", hold=3.0, pip=img)

    disturbed = False
    for rnd in range(args.max_rounds):
        # ----------------------------------------------------------- plan
        img, depth = look()
        director.say("① 計画: Gemini がシーンを見て手順を考える", "Task orchestration（thinking 有効）", pip=img)
        plan = gem.plan(img, args.instruction, list(CONTAINERS))
        plan_img = draw_plan(img, plan)
        save(plan_img, f"plan_round{rnd}")
        json.dump(plan, open(os.path.join(args.out, f"plan_round{rnd}.json"), "w"), ensure_ascii=False, indent=2)
        steps = [s for s in plan.get("steps", []) if container_of(s.get("destination"))]
        if rnd > 0:
            stats["replans"] += 1
        if not steps:
            director.say("Gemini: もう片付けるものはありません", plan.get("summary_ja", ""), hold=2.5, pip=plan_img)
            break
        reasons = " / ".join(f'{it["label"]}→{it.get("destination")}' for it in plan.get("items", [])[:6])
        director.say(f"計画: {len(steps)} ステップ  「{plan.get('summary_ja', '')[:34]}」", reasons[:70], hold=4.0, pip=plan_img)

        queue = [(s["object"], container_of(s["destination"])) for s in steps]
        prev: tuple[str, str] | None = None
        n_total, k = len(queue), 0
        while queue:
            obj, dest = queue.pop(0)
            center, floor_z, radius = CONTAINERS[dest]
            # ------------------------------------------------ look (one Gemini call)
            img, depth = look()
            director.say(f"② 見る: 「{obj}」はどこ？ {dest} のどこが空いている？", f"ステップ {k + 1}/{n_total}" + (f"（ついでに前の「{prev[0]}」が入ったかも確認）" if prev else ""), pip=img)
            res = gem.locate(img, obj, dest, prev)
            vimg = img
            if prev:
                ok = bool(res.get("previous_success"))
                reason = res.get("previous_reason_ja", "") or ""
                vimg = draw_badge(img, ("OK  " if ok else "NG  ") + reason[:22], ok)
                save(vimg, f"verify_{prev[0].replace(' ', '_')}")
                stats["steps"][-1]["verified"] = ok
                director.say(f"⑤ 確認: 「{prev[0]}」{'成功' if ok else '失敗 → あとでやり直し'}", reason[:60], hold=1.5, pip=vimg)
                if not ok:
                    queue.append(prev)
                    n_total += 1
            prev = None
            pt = res.get("object")
            if not pt or not pt.get("point"):
                director.say(f"「{obj}」が見つからない → スキップ", hold=1.5)
                continue
            grasp = env.refine_grasp(pt["point"], depth)
            pimg = draw_points(img, [pt])
            save(pimg, f"point_{obj.replace(' ', '_')}")
            director.say(f"③ ポインティング: 「{obj}」を発見", f"2D 点 {pt['point']} ＋ 深度 → 3D ({grasp.pos[0]:.2f}, {grasp.pos[1]:.2f}, {grasp.pos[2]:.2f}) m, 向き {np.rad2deg(grasp.yaw):.0f}°", hold=1.2, pip=pimg)
            place_xy = center
            spot = res.get("spot")
            if spot and spot.get("point"):
                off = env.point_to_world(spot["point"], depth)[:2] - center
                if np.linalg.norm(off) > radius:  # keep it inside the container
                    off *= radius / np.linalg.norm(off)
                place_xy = center + off
                pimg = draw_points(pimg, [spot], color=(255, 64, 129))
                save(pimg, f"spot_{obj.replace(' ', '_')}")
                director.say(f"③ 空間推論: {dest} の空いている場所", "既に置いた物に当たらない位置を Gemini が選ぶ", hold=1.0, pip=pimg)

            # ------------------------------------------------ act
            rec = {"round": rnd, "object": obj, "destination": dest, "verified": None}
            director.say(f"④ 実行: 「{obj}」をつかむ", "IK による直線軌道 + 平行グリッパ")
            if not env.pick(grasp):
                stats["grasp_retries"] += 1
                director.say("つかみ損ねた → もう一度見てやり直し", "グリッパの開き幅で把持失敗を検出")
                env.park()
                queue.insert(0, (obj, dest))
                n_total += 1
                continue
            director.say(f"④ 実行: {dest} へ運ぶ")
            env.place(place_xy, floor_z)
            env.park()
            stats["steps"].append(rec)
            prev = (obj, dest)
            k += 1

            # ------------------------------------------------ disturbance
            if args.disturb and not disturbed and k == 3:
                disturbed = True
                gt = ground_truth(env)
                moved = next((b for b in ITEM_BODIES if gt[b] != "table"), None)
                if moved:
                    director.say("（人がこっそり 1 つ机に戻した！）", "途中で環境が変わっても、最後に見直して気づけるか？")
                    env.set_body_pos(moved, (0.62, 0.02), 0.4)
                    env.wait(1.5)
                    stats["disturbed"] = moved

        director.say("最後にもう一度シーンを見て、やり残しがないか確認", "Closed-loop re-planning（最後のステップの成功判定も兼ねる）", hold=0.5)

    gt = ground_truth(env)
    img, _ = look()
    save(img, "final")
    director.say("完了！", f"机の上に残った物: {sum(v == 'table' for v in gt.values())} 個", hold=3.0, pip=img)
    director.close()

    stats.update(
        ground_truth=gt,
        instruction=args.instruction,
        seed=args.seed,
        gemini_calls=[{"kind": c.kind, "seconds": round(c.seconds, 2), "response": c.response} for c in gem.log],
        wall_seconds=round(time.time() - t_start, 1),
        video_seconds=round(director.n_frames / 25, 1),
    )
    json.dump(stats, open(os.path.join(args.out, "log.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: stats[k] for k in ("ground_truth", "replans", "grasp_retries", "video_seconds")}, ensure_ascii=False, indent=1))
    print("Gemini calls:", len(gem.log), "avg latency", round(np.mean([c.seconds for c in gem.log]), 2), "s")


if __name__ == "__main__":
    main()
