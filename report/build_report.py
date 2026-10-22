"""Build the report (HTML -> PDF with headless Chrome) from a run directory.

    python report/build_report.py runs/demo [runs/red_only ...] --account <omnicampus account>
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
JA = {"white plate": "お皿", "orange toy box": "おもちゃ箱", "table": "机の上", "floor": "床"}
NAMES = {"apple": "りんご", "orange": "オレンジ", "lemon": "レモン", "red_block": "赤い積み木", "blue_block": "青い積み木", "green_block": "緑の積み木"}


def img_tag(path: str, cls: str = "") -> str:
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{data}">'


def first(run: str, pattern: str) -> str | None:
    files = sorted(glob.glob(os.path.join(run, pattern)))
    return files[0] if files else None


def run_section(run: str) -> tuple[str, dict]:
    log = json.load(open(os.path.join(run, "log.json")))
    calls = log["gemini_calls"]
    by_kind: dict[str, list[float]] = {}
    for c in calls:
        by_kind.setdefault(c["kind"], []).append(c["seconds"])
    lat = "".join(
        f"<tr><td>{k}</td><td>{len(v)}</td><td>{sum(v) / len(v):.1f} 秒</td></tr>" for k, v in by_kind.items()
    )
    gt = "".join(f"<tr><td>{NAMES[b]}</td><td>{JA.get(w, w)}</td></tr>" for b, w in log["ground_truth"].items())
    steps = log["steps"]
    n_ok = sum(s["verified"] is True for s in steps)
    retries = log["grasp_retries"] + sum(s["verified"] is False for s in steps)
    stats = dict(steps=len(steps), ok=n_ok, retries=retries, replans=log["replans"], calls=len(calls), video=log["video_seconds"])
    plan = first(run, "*plan_round0.png")
    final = first(run, "*final.png")
    point = first(run, "*point_*.png")
    spot = first(run, "*spot_*.png")
    verify = first(run, "*verify_*.png")
    figs = "".join(
        f'<figure>{img_tag(p)}<figcaption>{c}</figcaption></figure>'
        for p, c in [(plan, "計画（青: 物体、桃: 行き先、矢印: Gemini の割り当て）"), (point, "ポインティング（つかむ直前に位置を取り直す）"),
                     (spot, "空間推論（桃: 容器の中の空いている場所）"), (verify, "成功判定"), (final, "最終状態")] if p
    )
    disturbed = log.get("disturbed")
    sec = f"""
    <h3>指示:「{html.escape(log['instruction'])}」（seed {log['seed']}{'、途中で妨害あり' if disturbed else ''}）</h3>
    <div class="grid">{figs}</div>
    <div class="cols">
      <table><tr><th>物体</th><th>最終的な場所（評価用の正解）</th></tr>{gt}</table>
      <table><tr><th>Gemini の呼び出し</th><th>回数</th><th>平均応答時間</th></tr>{lat}</table>
    </div>
    <p class="note">実行したステップ {stats['steps']}（次のステップの前の確認で成功 {stats['ok']}、残り {stats['steps'] - stats['ok'] - sum(s["verified"] is False for s in steps)} は最後の再計画で確認）、やり直し {retries} 回、再計画 {log['replans']} 回、
    動画の長さ {stats['video']} 秒{f'。妨害で戻された物: {NAMES.get(disturbed, disturbed)}' if disturbed else ''}。</p>
    """
    return sec, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--account", default="ACCOUNT")
    ap.add_argument("--youtube", default="（YouTube の URL）")
    ap.add_argument("--github", default="（GitHub の URL）")
    args = ap.parse_args()

    sections = [run_section(r) for r in args.runs]
    body = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    body = body.replace("{{RUNS}}", "".join(s for s, _ in sections))
    body = body.replace("{{YOUTUBE}}", html.escape(args.youtube)).replace("{{GITHUB}}", html.escape(args.github))
    out_html = os.path.join(HERE, "report.html")
    open(out_html, "w", encoding="utf-8").write(body)
    pdf = os.path.join(HERE, f"説明資料_PAI最終課題_{args.account}.pdf")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + out_html],
        check=True, capture_output=True,
    )
    print(pdf)


if __name__ == "__main__":
    main()
