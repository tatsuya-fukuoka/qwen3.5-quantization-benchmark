# 確認依頼: llm-compressor 0.15.0 のキャリブレーション処理をソースで確認する

## 背景

- Phase 2 の B2／B3 は llmcompressor **0.15.0**（Colab 実行時の実績バージョン）を使う。
- Web 調査で main ブランチについては次が確認できた: `DatasetArguments` に `concatenate_data`（既定 False、「datapoints を連結して max_seq_length を埋める」）があり、`get_processed_dataset` は入力データセットに `input_ids` 列がある場合はそのまま返す（＝トークナイズ済みデータセットには連結処理が効かない）。
- ただし **0.15.0 で同じかどうかは未確認**。ここをソースで確定させたい。

## 確認タスク

1. llmcompressor 0.15.0 の配布物（wheel または sdist）を pip download 等で取得し、ソースを読む。インストールや GPU は不要。
2. 次の各点を 0.15.0 のソースの該当箇所（ファイルパスと関数名）を添えて答える:
   - (a) `concatenate_data` は 0.15.0 に存在するか。既定値と意味。oneshot の引数としてどう渡すか。
   - (b) トークナイズ済み（`input_ids` 列あり）データセットを渡した場合の処理経路。前処理（連結・分割・切り詰め）は何が起きるか。
   - (c) 生テキスト（`text` 列）データセットを渡した場合、トークナイズ後に長いサンプルはどう扱われるか（切り詰めのみか、分割もするか）。`concatenate_data=True` の場合の処理。
   - (d) 可変長サンプルをキャリブレーションでバッチ化する際のパディングと attention_mask の扱い（パディングが GPTQ／AWQ の統計に混ざる構造かどうか、ソースから分かる範囲で）。
   - (e) `num_calibration_samples` がデータセット件数より大きい場合の挙動（エラーか、少ないまま進むか）。
3. 結論として、現在の `scripts/quantize_w4a16.py` の方式（トークナイズ済み Dataset を直接渡す）で問題ないか、`concatenate_data` を使うほうがよい構成があるかを述べる。コード変更はしない。

## 成果物

- 短いレポートを `docs/llmcompressor-0150-calibration-source-check.md` に作成し、commit して push する。
- 既存テスト全件が通ることを確認する（コードは変更しないので確認のみ）。
- 不明な点は「未確認」と書く。推測で断定しない。
