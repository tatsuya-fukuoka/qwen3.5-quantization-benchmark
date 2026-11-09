# Phase 2 修正依頼: vLLM の max_num_seqs を明示指定できるようにする

## 背景（2026-11-09 Colab L4 実走で発生した失敗）

Phase 2（Track B 本測定、`colab/phase2_main_b.ipynb`）を Colab L4 で開始したところ、最初の条件 B0（BF16）の評価が vLLM のエンジン初期化で失敗し、実行が停止した。

- 失敗時刻: 2026-11-09 20:38 JST、B0 評価開始から約3.2分
- エラー原文（抜粋）: `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (177)`
- 原因: Qwen/Qwen3.5-9B は Mamba 層を含むハイブリッドモデルで、BF16（約19GB）・L4（22GB）・`gpu_memory_utilization 0.95` の条件では Mamba キャッシュのブロックが 177 しか確保できず、`scripts/eval_vllm.py` が `LLM()` に `max_num_seqs` を渡していないため vLLM 既定値 256 が使われ、初期化チェックで落ちた。CUDA OOM ではない。データ・モデル取得・量子化は無事で、B 系の結果はまだ何も書かれていない。Track A の結果は無変更。

## ユーザーの決定（2026-11-09）

GPU は A100 に切り替えず **L4 のまま**、`max_num_seqs` を下げる設定変更で再開する。これはエンジン側の同時実行数の調整であり、評価の中身（データ・プロンプト・greedy・seed・ctx・件数）は一切変えない。

## 修正内容

### 1. `scripts/eval_vllm.py`

- CLI 引数 `--max-num-seqs`（`type=int`、**既定は None**）を追加する。
- 指定された場合のみ `LLM(..., max_num_seqs=<値>)` を渡す。指定が無い場合は従来どおり `LLM()` にこの引数を渡さない（＝vLLM 既定値のまま）。これにより Phase 0 の B1 実行時の挙動・再現性は一切変わらない。
- 結果 JSON の `settings` に `"max_num_seqs"` を記録する。指定時はその値、未指定時は `null`（または既定任せであることが分かる値）にし、`gpu_memory_utilization` の隣に置く。
- それ以外の変更はしない: `max_model_len=common.CTX`、`seed`、`language_model_only=True`、SamplingParams（temperature=0、seed）、PPL 計算、記録項目の既存部分はすべて現状維持。

### 2. `colab/phase2_main_b.ipynb`

- セル14 `run_condition` の評価呼び出しに `"--max-num-seqs", "128"` を追加し、B0〜B3 の**全条件に同じ値**を渡す。
- 値 128 の根拠: B0 で確保できた Mamba キャッシュブロックは 177 で、同時実行数はそれ未満でなければならない。128 は余裕を持って下回る値として採用する（コメントかマークダウンでこの根拠を1行残すこと）。
- ノートブックの他のセル・実行順・件数・`GPU_MEM_UTIL`・push の仕組みは変更しない。Phase 0／Phase 1 のノートブック（`colab/phase0_smoke.ipynb`、`colab/phase1_main_a.ipynb`）には触れない。

### 3. テスト

- 既存テストの流儀に沿って、`--max-num-seqs` の引数解釈と「指定時のみ LLM に渡る」ことを確認するテストを追加する（vLLM 本体はこの VM に無いので、import を伴わない形で。既存の `scripts/test_pipeline.py` 等のパターンを踏襲すること）。
- 既存テスト全件（現在34件）と新規テストを、この VM で `python3` により実行して全件 OK を確認する（pytest は未導入なので unittest 形式で直接実行する既存のやり方に従う）。

## 変えてはいけないもの

- 評価条件: データ（data_main、200／200／250件）、プロンプト、temperature=0、seed=0、ctx=8192、max_tokens、thinking OFF。
- `GPU_MEM_UTIL` の条件別値、量子化レシピ（`scripts/quantize_fp8.py`、`scripts/quantize_w4a16.py`）、データ準備（`scripts/prep_data.py`）。
- `results/` と `results_main/` の既存結果ファイル（A 条件・Phase 0 の結果 JSON、summary.tsv、timings.tsv）。

## 完了条件

1. 上記の修正とテスト追加が入っている。
2. テスト全件 OK（件数と結果を報告に含める）。
3. 変更は `scripts/eval_vllm.py`、`colab/phase2_main_b.ipynb`、テストファイルのみで、他ファイルに差分が無いことを `git diff --stat` で確認できる。
4. コミットして `main` へ push し、コミットハッシュを報告する。コミットメッセージ例: `Add --max-num-seqs option for Phase 2 vLLM eval (fix B0 Mamba cache init failure)`。
5. 報告には次を含める: 変更ファイル一覧、テスト結果、`LLM()` への引数の渡し方の実装要点、結果 JSON への記録方法。
