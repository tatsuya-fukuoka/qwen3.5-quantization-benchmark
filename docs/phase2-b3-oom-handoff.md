# Phase 2 B3（AWQ）CUDA OOM の調査・修正依頼（Claude Code 向け）

量子化実測 Phase 2（Track B）の B3（AWQ W4A16）が Colab L4 で CUDA OOM により失敗した。
原因を llm-compressor 0.15.0 のソースで特定し、**L4（22GB）のまま B3 が完走できる対策**を
`scripts/quantize_w4a16.py` に実装してほしい。実機（Colab）での再実行はこの依頼の範囲外（こちらで行う）。

## 失敗の事実（実測）

- 環境: Colab L4（22.03 GiB）、llmcompressor 0.15.0 / compressed-tensors 0.20.1 /
  transformers 5.21.0 / torch 2.14.1（venv-llmc）。対象 Qwen/Qwen3.5-9B（BF16 約19GB、マルチモーダル＋Mamba ハイブリッド）。
- B3 は 2026-11-09 23:24:55 JST 開始、**23秒で exit 1**。AWQ モデル未生成。
- エラー全文（見守りがセル出力から回収）:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.72 GiB (GPU 0; 22.06 GiB total capacity; 12.91 GiB already allocated; 4.46 GiB free; 17.39 GiB reserved in total by PyTorch)`
  発生箇所: llm-compressor の oneshot → sequential pipeline（`scripts/quantize_w4a16.py` の oneshot 呼び出し、
  `llmcompressor/pipelines/sequential/helpers.py` 付近）、キャリブレーション用フォワード段階。
  ※別の記録では「Tried to allocate 16.00 MiB」の行もあり、同一失敗の別アロケーション行の可能性がある。7.72 GiB の単一確保が本命とみてよい。
- 対照: **同一スクリプト・同一キャリブレーションで B2（GPTQ）は正常完了**（量子化 855秒、コミット 1b086b7 で結果保存済み）。
  B0（BF16）・B1（FP8）も完了済み。落ちるのは AWQ 経路だけ。

## 現行コードの要点（scripts/quantize_w4a16.py）

- モデルは `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=bfloat16, device_map="auto")` で読み込み。
- レシピ: GPTQ は `GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE, dampening_frac=0.01, block_size=128)`、
  AWQ は `AWQModifier(targets=["Linear"], scheme="W4A16", ignore=IGNORE)`（mappings は既定＝auto）。
- `oneshot_kwargs()` は model / dataset / recipe / processor（tokenizer）/ max_seq_length / num_calibration_samples のみ。
  **pipeline・sequential 系の引数は一切渡していない**（0.15.0 の既定値まかせ）。

## 調査してほしいこと

1. llmcompressor 0.15.0 のソース（PyPI の wheel を取得して読むこと）で:
   - `pipelines/sequential/helpers.py` の該当箇所が何を確保しているのか（7.72 GiB の単一確保の正体）。
   - sequential pipeline の既定の subgraph 分割（sequential_targets の既定値）と、中間活性化（IntermediatesCache）の
     置き場所・オフロードの既定値。
   - **GPTQ が通って AWQ が落ちる理由**（AWQ 固有の処理: smoothing のための活性化保持、mappings の auto 推定、
     duo_scaling 等が余分にメモリを使うのか）。
2. 対策候補の検討（この順で優先）:
   - oneshot へ sequential 系の引数を明示する（サブグラフを層単位に細分化する、キャッシュを CPU へオフロードする等、
     0.15.0 に実在する引数のみ）。
   - モデルの読み込み方・配置の工夫（例: 量子化対象外の Vision 部分を GPU に載せない）。ただし下の不変条件を守ること。
   - AWQModifier 側の引数でメモリを抑えるものがあればそれも。
3. 採用した対策について、**なぜ L4 の 22GB に収まるはずか**のメモリ見積もり（重み・活性化・キャッシュの内訳）を付けること。

## 不変条件（これを変える対策は不可）

- キャリブレーションは B2 と完全同一を維持する: `calib/hybrid_256x1024.json`
  （256件×1024トークン、samples_sha256 = cc3f7050…）。B3 だけ件数・系列長・内容を変える対策は、
  GPTQ vs AWQ の直接比較が崩れるため不可。
- 量子化レシピの実質を変えない: W4A16、targets=Linear、ignore は現行どおり（lm_head・visual・vision）、
  AWQ の mappings は既定のまま。**やむを得ず変える場合は、レポートに理由と「B2 との比較条件がどうずれるか」を明記すること。**
- 評価条件・ノートブックの評価パラメータは変更しない（ノートブック側の変更は、量子化コマンドへの
  引数追加がどうしても必要な場合のみ、その旨をレポートに明記してから）。
- B0/B1/B2・Track A の成果物（results_main/ の既存ファイル、calib/）は変更しない。
- GPU は L4 のまま（A100 への切替はユーザーが選ばなかった）。課金が発生する操作はしない。

## 実装の流儀（前回までの修正と同じ）

- oneshot へ渡す引数は純関数（既存の `oneshot_kwargs()` を拡張する形）に集約し、モックテストで検証できるようにする。
- テストを追加し、`scripts/test_*.py` の全件を unittest で実行して全件 OK を確認すること。
- 調査レポートを `docs/b3-awq-oom-investigation.md` に書くこと:
  (1) 7.72 GiB 確保の正体と発生箇所 (2) GPTQ と AWQ の差 (3) 採用した対策と根拠・メモリ見積もり
  (4) 却下した代替案と理由 (5) B2 との比較条件がずれていないことの確認。
- 変更は quantize_w4a16.py とテスト、必要ならノートブックの B3 量子化コマンド引数のみ。コミットして push すること
  （push が接続リセットで失敗したら、その旨を最終報告に書いてコミットだけ残すこと）。
- Colab の実機実行はしない（GPU がない環境での作業になる。実機確認はこちらで行う）。

## 完了の定義

- 調査レポートが docs/b3-awq-oom-investigation.md にある。
- 修正がコミットされ、テスト全件 OK。
- 最終報告に: 原因の要約、変更ファイル一覧、テスト結果、push の成否、実機で確認すべき点（AWQ mappings の
  auto 推定が通るか等、残る未確認事項）を書くこと。
