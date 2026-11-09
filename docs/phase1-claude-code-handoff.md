# Phase 1 本測定 (Track A) の実装依頼 — Claude Code 向け

対象リポジトリ: `tatsuya-fukuoka/quantization-exp`（このリポジトリで作業すること）
前提: Phase 0 スモークは完走済み。結果は `results/`（コミット 13bee98）に保存済みで、**上書きしないこと**。
設計の正: `docs/quantization-verification-design.md`（v3）。Phase 0 の依頼文は `docs/phase0-claude-code-handoff.md`。

## 目的

本測定（Track A、全9条件）を Google Colab（L4）で回すためのノートブック
`colab/phase1_main_a.ipynb` を用意する。あわせて、件数を条件別に制御できるよう
既存スクリプトへ小さい改修を入れる。

## 本測定の条件（確定事項）

- 件数: JCommonsenseQA 200件 / JSQuAD 200件 / **MGSM は全250件**。PPL テキストは Phase 0 と同一（JSQuAD valid の未使用 context から 50,000 文字）。
- Track A 9条件: A0 F16 / A1 Q8_0 / A2 Q5_K_M / A3 Q4_K_M / A3i Q4_K_M+imatrix / A4 Q3_K_M / A4i Q3_K_M+imatrix / A5 Q2_K / A5i Q2_K+imatrix。
- 評価条件は Phase 0 と同一に固定: greedy（temperature 0、seed 0）、ctx 8192、thinking OFF、max_tokens は Phase 0 と同じ（JCSQA 16 / JSQuAD 64 / MGSM 768 / 定性 512）。定性10問も全条件で保存する。
- imatrix は Phase 0 と同じもの（TFMC c4_en_ja、32チャンク）を使い回す。新規生成はしない（生成済み .dat が無い環境では Phase 0 と同じ手順で生成してよい）。

## スクリプト改修（最小限）

1. `scripts/prep_data.py`: MGSM だけ件数を変えられるようにする（例: `--n-mgsm` を追加し、既定は `--n` と同じ）。JCSQA/JSQuAD は `--n` のまま。末尾の件数チェックもタスク別に正しく動くこと。
2. 評価スクリプト（`eval_gguf.py` / `eval_vllm.py`）: 本測定では「データファイルの全行を評価する」ことが保証されるようにする。現行の `--n` の切り捨てセマンティクスを確認し、データ側の件数（200/200/250）がそのまま評価件数になるよう調整する（例: `--n 0` で全行、あるいは `--n` の既定を全行側に変更。Phase 0 の再現性は壊さないこと）。出力 JSON の `n` が JCSQA 200 / JSQuAD 200 / MGSM 250 になることを単体で確認できる形にする。
3. 改修にはテストを付ける（既存の unittest 形式に合わせる）。既存テストもすべて通ること。

## ノートブック `colab/phase1_main_a.ipynb` の要件

Phase 0 のノートブック（`colab/phase0_smoke.ipynb`）の構成・流儀を踏襲する。

- **冪等・自己完結**: 新規ランタイムでも、Phase 0 のランタイム継続でも動くこと。clone 済みなら pull、llama.cpp のバイナリがあればビルドをスキップ、モデル/GGUF があれば再取得・再変換をスキップする形にする。
- **「すべてのセルを実行」で完走できる**こと（対話プロンプトなし）。Colab シークレット `GITHUB_TOKEN` の使い方は Phase 0 と同じ。
- データは `data_main/` に分ける（`prep_data.py --outdir data_main --n 200 --n-mgsm 250 --with-imatrix`）。Phase 0 の `data/` は触らない。
- 結果は `results_main/` に分ける（`--results-dir results_main`）。集計は `results_main/summary.tsv`、timings は `results_main/timings.tsv` に残す（Phase 0 の `results/` と混ぜない）。
- GGUF 評価のポートは **18080** を使うこと。8080 は Colab 自身のプロセスが占有していて llama-server が bind できない（Phase 0 実測）。`--llama-dir` は clone 先リポジトリ内の `llama.cpp` を指すこと。
- 量子化は `make_gguf.sh` の条件引数を使う（A1/A2/A4/A4i/A5/A5i の型表は実装済みのはず。無ければ足す）。A0 の F16 変換と A3/A3i も、ファイルが無ければ作る。
- 評価の実行順: **A0 を最初**にし、残りは A1 → A2 → A3 → A3i → A4 → A4i → A5 → A5i。`results_main/<条件>.json` が既にある条件は評価セルをスキップする（セッションが落ちても続きから再開できるように）。
- 最後に thinking OFF 確認（全条件で `<think>` を含む定性出力が 0 件であること）と集計（summary 生成）を入れ、`results_main/` 一式を GitHub へ push するセルで締める。
- 各工程は `run_timed.py` 相当で timings に記録する（Phase 0 の流儀に合わせる）。

## やらないこと

- Track B（vLLM 系）は Phase 2 で別ノートブックにする。今回のノートブックには入れない。
- Phase 0 の `results/` と `colab/phase0_smoke.ipynb` の変更。
- 評価条件（プロンプト、採点、max_tokens 等）の変更。

## 完了報告に含めること

- 変更・追加したファイルの一覧と、各スクリプト改修の要点。
- テスト結果（新規・既存とも）。
- ノートブックのセル構成（番号と役割）と、想定所要時間の見積もり（Phase 0 実測ベース: GGUF 評価は MGSM 件数に比例。A0 の MGSM は 100件で 1111 秒だった）。
- 机上で確認できなかった点・推測で置いた点があれば、その旨を明記する。
