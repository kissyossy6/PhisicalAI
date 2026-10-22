# ことばで片付けるロボット ― Gemini Robotics-ER × MuJoCo Franka

Physical AI 応用1講座 最終課題。

「机の上を片付けて。食べ物はお皿に、おもちゃは箱にしまってね。」という**自然言語の指示ひとつ**で、
シミュレータ上の Franka Panda アームが机の上の物を分類して片付けます。

ロボットはシミュレータから物体の座標を**一切もらいません**。何がどこにあり、どこへ運ぶかは、
すべて Gemini Robotics-ER がカメラ画像を見て判断します。

ベースにしたサンプル: [Gemini Robotics-ER 1.6 を試す（npaka）](https://note.com/npaka/n/n93116b196748)。
このサンプルは画像への推論（ポインティング・物体検出・軌道・空間推論・オーケストレーション）を試すところまでです。
本作ではその出力を**実際のロボットの動作につなげ**、さらに確認と再計画まで含めた閉ループにしました。

## しくみ

```
 カメラ画像 (RGB-D, 800x600)
      │
      ▼
 ① 計画      Gemini: 指示 + 画像 → 物体リスト・分類・行き先・手順（thinking 有効）
      │
      ▼  各ステップごとに
 ② 物体の位置  Gemini: 「red cube はどこ？」→ 2D 点 [y, x]
      │        深度画像で 3D に逆投影し、周辺の点群から中心・高さ・向き(yaw)を補正
 ③ 置き場所    Gemini: 容器の中の「空いている場所」を指す（空間推論）
 ④ 実行        IK（減衰最小二乗法）による直線軌道で、つかむ → 運ぶ → 置く
 ⑤ 確認        Gemini: 画像を見て成功か判定。失敗ならもう一度見てやり直し
      │
      ▼
 最後にもう一度シーンを見て再計画 → やり残し（人が戻した物など）があれば続ける
```

| 役割 | 担当 |
|---|---|
| 何を・どこへ・どの順で（意味の理解） | Gemini Robotics-ER |
| 物体はどこ（2D） | Gemini Robotics-ER のポインティング |
| 正確な 3D 位置・向き | 深度画像の幾何処理（`sim/env.py: refine_grasp`） |
| 動かし方 | IK と動作プリミティブ（`sim/env.py`） |
| うまくいったか | Gemini Robotics-ER の成功判定 |

## 動かし方

macOS（Apple Silicon）で確認しています。Linux でも動くはずです。

```bash
./setup.sh                       # venv 作成・依存のインストール・Franka モデルの取得
# .env に GEMINI_API_KEY=... を書く（Google AI Studio で無料取得可）
.venv/bin/python run_task.py --seed 2 --disturb --out runs/demo
```

- `--instruction "..."` で指示を変えられます（例: `"赤いものだけ箱に入れて"`）。
- `--disturb` を付けると、途中で「人」が片付け済みの物をひとつ机に戻します。最後の見直しで気づいて直すかを試せます。
- `--replay runs/demo_live/log.json` を付けると、記録済みの Gemini の応答を同じ順に使って動画だけを描き直します（API を呼びません）。
  提出動画はこの方法で、本番の実行（`runs/demo_live/log.json`）から字幕のモデル名だけを直して描き直したものです。
- 出力: `runs/<name>/video.mp4`（動画）、`log.json`（Gemini の全応答・応答時間・最終状態）、各段階の注釈付き画像。

### モデルについて

記事の `gemini-robotics-er-1.6-preview` は 2026年10月時点で API から提供が終わっていたため、
後継の **`gemini-robotics-er-2-preview`** を使っています（環境変数 `GEMINI_ER_MODEL` で変更可）。
プロンプトの形式（`[y, x]` を 0〜1000 に正規化）は記事と同じです。

②③と前のステップの⑤は、Gemini への **1 回の呼び出し**（`locate`）でまとめて聞いています。
無料枠（毎分5回・1日20回）でも1回の実行（約10回の呼び出し）が収まるようにするためです。

毎分の上限に当たった場合は、サーバーが指定する秒数だけ待って再試行します。
待っている間はシミュレーションも止まるので、動画の中の時間は進みません。

## ファイル

| パス | 内容 |
|---|---|
| `run_task.py` | 全体の流れ（計画 → 実行 → 確認 → 再計画）と動画の作成 |
| `ai/gemini_er.py` | Gemini Robotics-ER の呼び出し（plan / point / free_spot / verify） |
| `ai/overlay.py` | 注釈の描画と動画のレイアウト |
| `sim/scene.xml` | MuJoCo のシーン（机・果物・積み木・皿・おもちゃ箱・カメラ） |
| `sim/env.py` | 環境: RGB-D の描画、2D→3D 変換、把持位置の補正、IK、動作プリミティブ |
| `report/` | レポート（PDF）とその元になる HTML |

## クレジット

- Franka Emika Panda のモデル: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0）。`setup.sh` で取得します。
- 物理シミュレータ: [MuJoCo](https://mujoco.org/)
- VLM: Google Gemini Robotics-ER
