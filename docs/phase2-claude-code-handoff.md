# Phase 2 本測定 (Track B) の実装依頼 — Claude Code 向け

対象リポジトリ: `tatsuya-fukuoka/quantization-exp`（このリポジトリで作業すること）
前提: Phase 0 スモーク完走済み（結果は `results/`、コミット 13bee98）。Phase 1 (Track A 本測定) も完走済み（結果は `results_main/` の A0〜A5i と `results_main/summary.tsv`、コミット 9e473f5）。**どちらも上書きしないこと。**
設計の正: `docs/quantization-verification-design.md`（v3）。先行する依頼文は `docs/phase0-claude-code-handoff.md` と `docs/phase1-claude-code-handoff.md`。

## 目的

本測定（Track B、全4条件）を Google Colab（L4）で回すためのノートブック
`colab/phase2_main_b.ipynb` を用意する。あわせて GPTQ／AWQ の量子化スクリプトを新規に追加する。

## 本測定の条件（確定事項）

- 件数: Phase 1 と同一。JCommonsenseQA 200件 / JSQuAD 200件 / **MGSM は全250件**。データは `data_main/` を使い回す（無ければ `prep_data.py --outdir data_main --n 200 --n-mgsm 250 --ppl-exclude-n 100 --with-imatrix` で生成）。PPL テキストは Phase 0/1 と同一。
- Track B 4条件（すべて vLLM、言語モデル部分のみ `--language-model-only` 相当）:
  - B0: BF16（元の `Qwen/Qwen3.5-9B` をそのまま vLLM で評価。Track B の基準）
  - B1: FP8 dynamic（`scripts/quantize_fp8.py` で自前量子化。Phase 0 で100件実施済みだが、本測定の件数で再測する）
  - B2: GPTQ W4A16（llm-compressor `oneshot`、日本語キャリブレーション 512 サンプル）
  - B3: AWQ W4A16（llm-compressor `oneshot`、キャリブレーションは B2 と同一サンプル）
- 量子化の対象は言語モデルの Linear 層のみ。Vision Encoder と lm_head は対象外（FP8 の `quantize_fp8.py` の方針に揃える）。
- 評価条件は Phase 0/1 と同一に固定: greedy（temperature 0、seed 0)、ctx 8192、thinking OFF、max_tokens は JCSQA 16 / JSQuAD 64 / MGSM 768 / 定性 512。定性10問も全条件で保存する。
- vLLM は nightly、llm-compressor は別々の uv venv に分離する（Phase 0 の流儀）。
- 結果は `results_main/` に B0〜B3 の JSON を追加する形（A 条件のファイルは触らない）。集計は `results_main/summary_b.tsv`、timings は `results_main/timings_b.tsv` に分ける（Track A の `summary.tsv` / `timings.tsv` を上書きしない）。

## 新規スクリプト: GPTQ／AWQ 量子化

- `scripts/quantize_fp8.py` と CLI の流儀を揃えた量子化スクリプトを追加する（GPTQ 用と AWQ 用に分けるか、`--method` で切り替える統合スクリプトにするかは実装判断でよい。単体テストで引数と対象レイヤーの指定が確認できる形にすること）。
- キャリブレーション: `prep_data.py` が生成する imatrix 用テキスト（`data_main/imatrix_calib.txt`）から **512 サンプル**を固定 seed で抽出し、B2/B3 で同一のものを使う。サンプルの切り出し方（分割方法・最大トークン長）は実装で決めてテストに固定し、結果 JSON の `model` 情報に量子化パラメータとキャリブレーション条件（サンプル数・seed・元テキスト）を記録する。
- 出力は compressed-tensors 形式のモデルディレクトリ（`models/` 配下、`.gitignore` 対象であることを確認）。
- llm-compressor の API はバージョンで差が大きい。使うバージョンを固定し、API の前提（oneshot の引数、GPTQ／AWQ のレシピ指定方法、ignore リストの書き方）は机上で確認できた範囲と推測の範囲を完了報告で分けて明記すること。

## ノートブック `colab/phase2_main_b.ipynb` の要件

Phase 0/1 のノートブックの構成・流儀を踏襲する。Phase 1 との最大の違いは **条件ごとの push** を最初から組み込むこと（下記）。

- **冪等・自己完結**: 新規ランタイムでも動くこと。clone 済みなら pull（**開始時に必ず origin の最新を取り込む**。results_main/ には A 条件の結果が既にある前提）、モデル/量子化済みモデルがあれば再取得・再量子化をスキップ。
- **「すべてのセルを実行」で完走できる**こと（対話プロンプトなし）。Colab シークレット `GITHUB_TOKEN` の使い方は Phase 0/1 と同じ（**セルの中からのみ使う**こと。ターミナルからは読めないことが Phase 1 で判明している）。
- **条件ごとの push（必須）**: 各条件の評価が終わり `results_main/<条件>.json` が書かれたら、その場でセルから `results_main/` の当該条件分を git add / commit / push する（コミットメッセージは `Add Phase 2 result (B0)` の形式）。push が失敗した場合は、黙って続行せず、セル出力に明確な警告を残すこと。最後のセルでも一括 commit / push を行い、取りこぼしを回収する。
  - 背景: Phase 1 では最後に一括 push する設計だったため、(a) 12:58 開始分はランタイム喪失で push 前の結果が全て消え、(b) 完走時も外部の途中保存コミットと競合してノートブック側の push が届かなかった。条件ごと push で両方を防ぐ。
- 評価の実行順: **B0 を最初**にし、B1 → B2 → B3。`results_main/<条件>.json` が既にある条件は評価セルをスキップする（セッションが落ちても続きから再開できる。開始時の pull で他所から push された結果も拾える）。
- B0 (BF16) の VRAM 対策: 言語モデル部分のみ読み込み、ctx 8192、`gpu_memory_utilization` は L4 (24GB) に載るよう高め (0.95 を起点) に設定する。**OOM になった場合に備え、失敗が分かる形**（例外で止まる、または結果 JSON に失敗として記録される）にし、勝手に評価条件（ctx や件数）を変えて通さないこと。A100 への切り替えはこのノートブックの範囲外（判断は人間が行う）。
- 最後に thinking OFF 確認（全条件で `<think>` を含む定性出力が 0 件であること）と集計（`make_summary.py` 相当で `results_main/summary_b.tsv` を生成）を入れ、一括 push のセルで締める。
- 各工程は `run_timed.py` 相当で timings_b に記録する。

## やらないこと

- Track A（GGUF 系）のノートブック・スクリプトの変更。Phase 0/1 の `results/` と `results_main/` の既存ファイル（A 条件の JSON、`summary.tsv`、`timings.tsv`）の変更。
- `colab/phase0_smoke.ipynb` / `colab/phase1_main_a.ipynb` の変更。
- 評価条件（プロンプト、採点、max_tokens 等）の変更。

## 完了報告に含めること

- 変更・追加したファイルの一覧と、各スクリプトの要点。
- テスト結果（新規・既存とも。既存テストがすべて通ること）。
- ノートブックのセル構成（番号と役割）と、想定所要時間の見積もり（Phase 0 実測ベース: B1 の評価は100件で約5分。量子化は FP8 が約2分。GPTQ／AWQ の量子化所要は未実測のため、その旨を明記して机上見積もりを出すこと）。
- 机上で確認できなかった点・推測で置いた点（特に llm-compressor / vLLM nightly の API 前提と B0 の L4 収容可否）があれば、その旨を明記する。
