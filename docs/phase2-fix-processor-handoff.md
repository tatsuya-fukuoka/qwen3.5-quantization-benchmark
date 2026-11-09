# Phase 2 修正依頼: B2 の oneshot が processor 初期化で失敗する（torchvision 不足）

## 発生した失敗（2026-11-09 22:56〜23:03 JST、Colab 実機）

- §5b は成功し、パック済みキャリブレーションは GitHub 保存済み（コミット `11f9b67`、Muse 側で 256 件 × 1024 トークンと `samples_sha256` の一致を照合済み）。再開時は §5b の照合で再利用される。
- B2（GPTQ）の量子化セルが開始 17 秒で失敗し、実行停止。B3 以降は未実行。量子化モデルは作られていない。
- エラー要旨: llm-compressor の `oneshot` の `pre_process` が processor を初期化しようとして失敗。
  - 上流: `scripts/quantize_w4a16.py` の `oneshot(...)` → `llmcompressor/entrypoints/oneshot.py` → `pre_process`
  - `ImportError: Qwen3VLVideoProcessor requires the Torchvision library but it was not found in your environment.`（`AutoProcessor.from_pretrained` 経由で Qwen3VL の VideoProcessor が torchvision を要求）
  - llm-compressor 側のメッセージ: 「dataset が提供される場合は processor が必要。解決するには processor を作って `oneshot` に直接渡すこと」

## 調査してほしいこと

1. llm-compressor 0.15.0 のソース（wheel を取得して読む）で、`pre_process` が processor を初期化する条件と、`oneshot` の `processor` 引数に何を渡せば初期化を回避できるかを確認する。テキストのみ・トークナイズ済みデータセットの今回のケースで、既に読み込み済みの `AutoTokenizer` を渡すことで足りるか。
2. 先行する B1（FP8、`scripts/quantize_fp8.py`）は前回のランタイムで量子化に成功している。`quantize_fp8.py` の `oneshot` 呼び出しと `quantize_w4a16.py` の違い（processor の渡し方、データセットの形など）を比較し、なぜ B1 はこの問題に当たらなかったのか（または当たる条件が違ったのか）を特定する。

## 修正方針

- 第一候補: `quantize_w4a16.py` で processor（または tokenizer）を `oneshot` に明示的に渡し、`AutoProcessor` の自動初期化を回避する。ソース上の根拠を修正報告に書くこと。
- 代替候補（上の方法がソース上成り立たない場合のみ）: ノートブックの venv-llmc 導入セルに torchvision を追加する。その場合は torch とのバージョン整合と、既存の導入手順への影響を明記する。
- どちらの場合も、量子化レシピ・キャリブレーション・評価条件は変えない。`quantize_fp8.py` に同種の潜在問題があれば、変更はせず報告だけする（B1 の結果は確定済み）。

## テスト・完了条件

- `oneshot` への引数の渡し方をモック等で確認するテストを追加し、既存テスト全件（53 件）が通ることを確認する。
- commit して push する（push が接続リセットで失敗したら、その旨を明記してローカルコミットのまま報告する）。
- 最後に要約（原因、B1 との違い、修正内容、テスト結果、未確認事項）を報告する。

## 制約

- B0／B1 の結果、Track A の成果物、`calib/` のパック済みファイルには触れない。
- Colab の実行はしない（修正とテストまで）。課金なし。
