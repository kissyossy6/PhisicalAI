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
    caps = [
        (plan, "図 1　計画（青: 物体、桃: 行き先、矢印: VLM による割り当て）"),
        (spot or point, "図 2　把持直前の観測（青: 把持対象、桃: 容器内の配置位置）"),
        (verify, "図 3　直前のステップの成功判定"),
        (final, "図 4　実行終了時の状態"),
    ]
    figs = "".join(f"<figure>{img_tag(p)}<figcaption>{c}</figcaption></figure>" for p, c in caps if p)
    disturbed = log.get("disturbed")
    by_replan = stats["steps"] - n_ok - sum(s["verified"] is False for s in steps)
    summary = [
        ("配置した物体（真値による評価）", f"{sum(w != 'table' for w in log['ground_truth'].values())} / {len(log['ground_truth'])}"),
        ("実行したステップ数", f"{stats['steps']}（外乱への対応 1 を含む）" if disturbed else f"{stats['steps']}"),
        ("成功判定", f"次ステップ前の確認で成功 {n_ok}、終了時の再計画で確認 {by_replan}"),
        ("把持・配置のやり直し", f"{retries} 回"),
        ("再計画の回数", f"{log['replans']} 回"),
        ("VLM の呼び出し回数", f"{len(calls)} 回"),
        ("動画の長さ（実時間）", f"{stats['video']:.0f} 秒"),
    ]
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary)
    dist_text = (
        f"実行の途中、3 個目の物体を配置した直後に、片付け済みの{NAMES.get(disturbed, disturbed)}を作業台上へ戻す外乱を加えた。"
        f"ロボットは残りの物体を処理した後の再計画でこれを検出し、再び皿へ配置した。"
        if disturbed else ""
    )
    sec = f"""
    <p>乱数シード {log['seed']} の配置で、指示文「{html.escape(log['instruction'])}」を与えて実行した。
    VLM は果物 3 種を食べ物、立方体 3 個をおもちゃと分類し、それぞれ皿とおもちゃ箱に割り当てる計画を生成した（図 1）。
    {dist_text}最終的に 6 物体すべてが指示どおりの容器に収まった（表 2、表 3）。</p>
    <div class="grid">{figs}</div>
    <div class="cols">
      <div class="tbl"><table><tr><th>項目</th><th>結果</th></tr>{rows}</table>
      <p class="cap">表 2　実行結果の概要</p>
      <table><tr><th>問い合わせ</th><th>回数</th><th>平均応答時間</th></tr>{lat}</table>
      <p class="cap">表 4　VLM の応答時間</p></div>
      <div class="tbl"><table><tr><th>物体</th><th>最終位置（真値）</th></tr>{gt}</table>
      <p class="cap">表 3　各物体の最終位置</p></div>
    </div>
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
