# Phase 2 修正依頼: §5b の fineweb-2 ストリーミング失敗（config 未指定）

## 発生した失敗（2026-11-09 22:34 JST、Colab 実機）

- 最新ノートブック（51e8c1a）で「すべてのセルを実行」。セットアップ・L4 確認・データ準備は通過し、B0／B1 はスキップされた。
- §5b（セル14、`scripts/build_calibration.py`）が約8秒で失敗し実行停止。B2 は未開始。パック済みファイルは作られていない。
- 途中まで正常だった出力: `existing corpus: 120 docs, 93306 tokens`（調査時の実測と完全一致）。
- エラー末尾（ターミナルで同スクリプトを再現して取得）:
  `ValueError: BuilderConfig 'default' not found. Available: ['aai_Latn', ..., 'jpn_Jpan', ...]`
- 呼び出しは `iter_fineweb2` 内の `load_dataset(FINEWEB_REPO, data_files=FINEWEB_SHARD, split="train", streaming=True, revision=rev)`。fineweb-2 は config を持つデータセットで、config 名（`jpn_Jpan`）を指定していないため `datasets` が `default` を探して失敗した、というのが原因の見立て。

## 修正タスク

1. `scripts/build_calibration.py` の `iter_fineweb2` を修正し、`load_dataset` に config 名 `jpn_Jpan` を明示する（`name="jpn_Jpan"`）。config 名は定数化し、設計書 `docs/calibration-hybrid-design.md` の §2 の取得コード例も実態に合わせて直す。
2. 同じ箇所に潜む他の実機差異も点検する（例: `data_files` に config 配下のシャードを指定する形が config 指定と両立するか、`id`／`text` 列名、`.shuffle(seed, buffer_size)` の streaming での扱い）。可能なら、この VM に `datasets` を venv へ入れて、fineweb-2 のストリーミングで先頭数件の取得までを実際に試す（ネットワークが通らなければ無理に進めず「未確認」と報告する。トークナイズや構築全体は不要、load_dataset の疎通まででよい）。
3. テストを追加する: `load_dataset` へ渡す引数（repo・config 名・data_files・streaming・revision）が期待どおりであることを、モックまたは引数構築の分離で確認する（`datasets` 未導入でも通る形にする）。既存テスト全件（51 件）が通ることも確認する。
4. ノートブック側の変更は不要のはずだが、§5b セルの呼び出し引数に変更が必要なら最小限で行い、テストで担保する。
5. commit して push する。最後に要約（修正内容、実ストリーミング疎通を試した結果、テスト結果、未確認事項）を報告する。

## 制約

- B0／B1 の結果、Track A の成果物、量子化・評価の中身には触れない。
- Colab の実行はしない（修正とテストまで）。課金なし。
