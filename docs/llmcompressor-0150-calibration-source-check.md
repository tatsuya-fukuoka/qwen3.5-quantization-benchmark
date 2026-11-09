# llm-compressor 0.15.0 キャリブレーション処理のソース確認

対象: `llmcompressor==0.15.0`（PyPI の `llmcompressor-0.15.0-py3-none-any.whl` を `pip download --no-deps` で取得して展開、インストール・GPU なし）。
以下のパスはすべて wheel 内の `llmcompressor/` からの相対パス。コード変更はしていない。

## (a) `concatenate_data`

- **存在する**。`args/dataset_arguments.py` の `DatasetArguments.concatenate_data: bool`、**既定 `False`**。
- 意味: help 文は "Whether or not to concatenate datapoints to fill max_seq_length"。実装は `transformers/data/base.py` の `TextGenerationDataset.group_text`（トークナイズ済みの全行を連結し、`max_seq_length` ごとに分割、端数は捨てる）。
- 渡し方: `entrypoints/oneshot.py::oneshot()` の明示キーワード引数 `concatenate_data: bool = False`（L402）。`oneshot(..., concatenate_data=True)` で渡す。

## (b) トークナイズ済み（`input_ids` 列あり）データセット

- 経路: `datasets/utils.py::get_processed_dataset`。`dataset_args.dataset` が `column_names` に `"input_ids"` を持つ場合、**即 `return dataset`**（"dataset is already tokenized"）。
- よって連結（`group_text`）・分割・切り詰め・パディングの前処理は**一切行われない**。`concatenate_data=True` を渡しても効かない（`group_text` は `TextGenerationDataset.__call__` 内にあり、この経路では呼ばれない）。`max_seq_length` もデータセット側には適用されない（バッチ化時の `DataCollatorWithTruncation` のみ、後述）。
- Web 調査で確認された main の挙動と 0.15.0 は同じ。

## (c) 生テキスト（`text` 列）データセット

- 経路: `get_processed_dataset` → `TextGenerationDataset.__call__`（`transformers/data/base.py`）。`tokenize()` が `max_length=max_seq_length, truncation=True` でトークナイズ。**切り詰めのみ**で、長いサンプルの分割（複数サンプル化）はしない。
- `padding`: `concatenate_data=True` なら False、そうでなければ `pad_to_max_length`（既定 False）なら `"max_length"`、それ以外 False（`__init__`）。既定ではトークナイズ時にパディングしない。
- `concatenate_data=True` の場合: トークナイズ後に `group_text` が全サンプルを連結し `max_seq_length` 長のチャンクに分割（端数は捨てる）。結果、全サンプルが同一長になる。件数は元の件数と一致しなくなる（総トークン数 ÷ `max_seq_length`）。`max_seq_length` が None の場合は tokenizer の `model_max_length` になる点に注意。

## (d) 可変長バッチのパディング・attention_mask

- バッチ化: `datasets/utils.py::_make_collate_fn`。`oneshot` の `data_collator` 既定は `"truncation"`、`batch_size` 既定は 1。
  - `"truncation"`（既定）: `DataCollatorWithTruncation` がバッチ内の最短長に**切り詰める**（`max_seq_length` があればそれも上限）。パディングは入らない。`input_ids` / `attention_mask` / `labels` / `loss_mask` を同じ長さに切る。
  - `"padding"`: `DataCollatorWithPadding(tokenizer)` でバッチ内最長に**パディング**（pad_token 無しなら eos を使用）。ソース中の警告に "calibrates on padding tokens not seen at runtime" とあり、パディングトークンが統計に入りうることを実装側も認めている。
- `batch_size=1`（既定）なら、どちらの collator でもパディング・切り詰めは発生しない。
- パディングのマスク: `modifiers/utils/pytorch_helpers.py::apply_pad_mask_to_batch`（padding 位置の `input_ids` を 0 にして GPTQ の Hessian に寄与させない、という docstring）が定義されているが、**0.15.0 の wheel 内で grep した限り、定義以外に参照が無い（呼び出し箇所なし）**。GPTQ／AWQ のモディファイア内で `attention_mask` を使って統計を重み付け・除外する処理も、`modifiers` / `pipelines`（トレース用 `transformers_helpers.py` を除く）の grep では見つからなかった（AutoRound 以外）。
  - したがって「`data_collator="padding"` + `batch_size>1` ではパディングが GPTQ／AWQ の統計に混ざりうる構造」と読める。ただしモデル forward 側が `attention_mask` を使ってパディング位置の注意を落とすため、パディング位置の出力は他トークンに影響しない可能性はある。この影響の定量は**未確認**（実行はしていない）。

## (e) `num_calibration_samples` > データセット件数

- `datasets/utils.py::_make_sampler`: `num_samples > len(dataset)` のとき **警告のみ**（"Requested N samples but the provided dataset only has M samples."）で `num_samples = len(dataset)` に丸めて続行する。エラーにならない。
- 補足: `shuffle_calibration_samples` 既定 True（`RandomSampler`）。False の場合は `LengthAwareSampler`（長さ降順）。

## 結論

- 現行 `scripts/quantize_w4a16.py` の方式（`tok(..., max_length=512, truncation=True, add_special_tokens=False)` で事前トークナイズした `Dataset` を直接渡し、`max_seq_length=512, num_calibration_samples=len(samples)` を指定）は、0.15.0 のソース上**問題ない**。
  - `input_ids` 列があるため `get_processed_dataset` が即返し、llm-compressor 側の再トークナイズ・連結・分割は行われない。サンプル内容は B2/B3 で完全に同一のまま保たれる。
  - `batch_size=1`（既定）・`data_collator="truncation"`（既定）なので、パディングは発生せず、パディングが GPTQ／AWQ 統計に混ざることもない。
  - `num_calibration_samples=len(samples)` は件数と一致するため、丸め警告も出ない。
- 注意点（現行方式の副作用）: トークナイズ済みデータでは `max_seq_length` は collator の上限としてしか効かない。切り詰めは自前の `tok(..., truncation=True)` に依存している（現行コードは実施済み）。
- `concatenate_data=True` を使うほうがよい構成: 現行の比較（B2 vs B3、同一サンプル・1 サンプル 1 文書）では**不要**。使うなら、生テキスト（`text` 列）を渡す構成に変える必要があり、サンプル数・内容が元の 512 件と一致しなくなる（総トークン÷512 のチャンク）ため、他の条件（imatrix 等と同一キャリブレーションにする設計）と揃わなくなる。短文が多く 512 トークンを満たさないデータで統計量を増やしたい場合のみ検討対象。
- 未確認: 実機での挙動、パディング混入の量子化精度への影響。

## 既存テスト

`python3 -m unittest discover -s scripts -p 'test_*.py'` → 35 件すべて OK（pytest は未インストールのため unittest で実行。コード変更なし）。
