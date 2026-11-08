# Claude Code 依頼文 — 量子化実測 Phase 0 実装

作成: 2026-11-09
依頼者: Muse（ユーザーの承認済み設計に基づく）

## 最初に読むもの

`~/workspace/goals/paid-article-verification-theme/files/quantization-verification-design.md`（検証設計 v3・確定版）。対象モデル、条件一覧、評価指標、フェーズ分けは全てそちらが正。本依頼はそのうち **Phase 0（スモーク）を Colab で実行できる形にする実装** です。

## 作る場所

GitHub リポジトリ `tatsuya-fukuoka/quantization-exp`（private）のローカルクローン `~/workspace/quantization-exp/` で作業する。リポジトリは初期化済みで、`.gitignore`（重み・imatrix・データ・.env を除外）と README の初期コミットが push まで完了している。作業の区切りごとに commit し、最後に push すること。モデル重みやデータの実体は置かない（実行時に Colab 側でダウンロードする）。
Colab ノートブックの先頭では `git clone https://github.com/tatsuya-fukuoka/quantization-exp` で取得する想定（private のため、Colab 側ではユーザーが GitHub 認証を通す前提。認証方法は README に手順を書く）。

## 成果物

1. `README.md` — 目的、Colab での実行手順（ランタイムは L4 を選択）、Phase 0 の所要時間の測り方、結果の見方
2. `colab/phase0_smoke.ipynb` — 上から順に実行すると Phase 0 が完走するノートブック。セル構成:
   - 環境確認（GPU 名・VRAM・CUDA 表示）
   - llama.cpp の clone と CUDA ビルド（最新 main。ビルド番号を記録）
   - モデル取得（`Qwen/Qwen3.5-9B`、公開リポジトリ、gated でないこと）
   - GGUF 変換（言語モデル部分のみ、F16）と量子化 A0／A3／A3i の生成（imatrix は `TFMC/imatrix-dataset-for-japanese-llm` の `c4_en_ja_imatrix.txt` を使用、`llama-imatrix --chunks 32`）
   - Track B の準備（vLLM nightly のインストール、llm-compressor による FP8 dynamic 量子化 = B1）
   - 評価の実行（下記ハーネスを呼ぶ）
   - 結果集計（summary.tsv の表示）
3. `scripts/` — ノートブックから呼ぶ実体スクリプト群
   - `build_llama.sh` / `make_gguf.sh`（変換・量子化・imatrix 生成。条件を引数で受ける）
   - `quantize_fp8.py`（llm-compressor oneshot、FP8 dynamic。量子化対象は言語モデルの Linear のみ、Vision は対象外）
   - `prep_data.py`（評価データの取得と整形: JCommonsenseQA と JSQuAD は JGLUE 公式、MGSM は公式の日本語ファイル。スモーク用に先頭100件へ絞る処理も含む）
   - `eval_gguf.py`（GGUF 条件の評価。llama-server を OpenAI 互換 API で叩く方式を基本とする）
   - `eval_vllm.py`（vLLM 条件の評価。オフライン LLM クラスでよい）
   - `score.py`（採点のみを分離: JCommonsenseQA=accuracy、JSQuAD=F1/EM、MGSM=数値の exact match。ダミーデータでの単体テスト付き）
   - `prompts_qualitative.txt`（定性評価の固定10プロンプト。日本語で、日常・要約・計算・コード・敬語・長文理解などを散らす）
4. `results/` の出力形式
   - 条件ごとに `results/<condition>.json`（条件名、モデル情報、バージョン類、件数、各指標、生の生成結果へのパス）
   - `results/summary.tsv`（列: 条件／種別／ファイルサイズGB／JCSQA acc／JSQuAD F1／MGSM acc／PPL／tok/s／VRAM GB）

## Phase 0 の条件（4つだけ）

- A0: GGUF F16（基準）
- A3: GGUF Q4_K_M（標準）
- A3i: GGUF Q4_K_M（imatrix）
- B1: FP8 dynamic（llm-compressor → vLLM、`--language-model-only` で起動）

各タスク先頭100件（MGSM も100件）、日本語 perplexity は固定の日本語テキスト（prep_data で用意する評価用テキスト）で測る。

## 共通の評価条件（厳守）

- thinking は OFF 固定（GGUF 側は llama-server の chat template kwargs、vLLM 側は chat template の指定で `enable_thinking=false`）。OFF になっているかは、定性プロンプトの生出力で目視確認できるよう生出力を保存する
- greedy（temperature 0）、seed 固定
- プロンプト形式は全条件で同一（同一の整形コードを通す）
- コンテキスト長は評価用に 8192 へ絞る

## やってはいけないこと

- 課金が発生するサービスの利用（Colab は無料枠前提。重み・データは公開のものだけ）
- 設計の変更（条件の追加・削除、指標の変更はしない。詰まった場合は実装を止めて、詰まった箇所と理由を報告する）
- この VM での GPU 実行の試行（この VM に GPU はない。コードの静的確認と score.py の単体テストまで）

## 完了報告の形式

- 作ったファイルの一覧と各役割を1行ずつ
- score.py の単体テスト結果
- Colab で実行する際に想定される所要時間の見積もり（ダウンロード／ビルド／量子化／評価の内訳）
- 設計書と齟齬が出た点、判断に迷って仮置きした点があれば明記する
