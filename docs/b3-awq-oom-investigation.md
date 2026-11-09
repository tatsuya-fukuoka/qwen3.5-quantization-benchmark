# B3（AWQ W4A16）CUDA OOM 調査レポート

対象: Colab L4（22 GiB）/ llmcompressor 0.15.0 / compressed-tensors 0.20.1 / Qwen/Qwen3.5-9B。
調査方法: PyPI の wheel（`llmcompressor-0.15.0`、`compressed_tensors-0.20.1`）を取得してソースを読んだ。
**GPU が無い環境での静読のみ。下の数値は見積もりであり実測ではない。実機確認が必要な点は末尾に列挙する。**

## 結論（要約）

- B3 が落ちたのは、AWQ が **GPU 上に** 置くデータ（親モジュールの forward 引数キャッシュ 256 件分、
  fp16 / 量子化後出力の 2 リスト）が、**GPU に常駐しうるモデル重み**と同居したため。GPTQ にはこの GPU 常駐データが無い。
- 「7.72 GiB の単一確保」の正体は、ソースからは**特定できなかった**（理由は §1）。
  そのため対策は単一テンソルではなく、AWQ 側の GPU 常駐分の総量を下げる形にした。
- 対策（`scripts/quantize_w4a16.py`、AWQ のときだけ有効。B2 の経路は不変）:
  1. `AWQModifier(offload_device=cpu)`（キャッシュを CPU へ）
  2. モデルを `device_map="cpu"` で読み込み（oneshot がデコーダ 1 層ずつ GPU に載せる）
  3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  4. `sequential_offload_device="cpu"` を明示（0.15.0 の既定値と同じ。固定のため）
- キャリブレーション・W4A16・targets・ignore・AWQ mappings は不変。

## 1. 7.72 GiB 確保の正体と発生箇所

ソースで確認できた事実:

- `pipelines/sequential/helpers.py` には、この規模のテンソルを確保する処理が無い。含まれるのはトレース・分割
  （`trace_subgraphs` / `topological_partition` / `partition_graph`）と、OOM を捕まえて文言を差し替える
  `handle_sequential_oom`（`helpers.py:680`）だけ。したがって「helpers.py 付近」は、`SequentialPipeline.__call__` が
  `@handle_sequential_oom` で包まれているために例外がそこで再送出されることを指していると考えられる。
  実際の確保元は、その配下のフォワード（サブグラフ実行）か AWQ のフックである。
- 依頼文のエラー文は素の `torch.OutOfMemoryError` で、ラッパーが付ける
  「Sequential pipeline ran out of memory…」は含まれない。回収したのは `__cause__`（元の例外）側だと思われる。
  **完全なトレースバックが無いと確保した行は特定できない。**
- 単一テンソルとして 7.72 GiB に一致する候補は見つからなかった（seq=1024・hidden=4096・vocab=248320・
  intermediate=12288 の bf16 / fp32 の組み合わせを試算。例: 256 件ぶんの hidden は 2.0 GiB、logits は 1 件 0.47 GiB）。
  「16.00 MiB」は fp32 の 1024×4096 の 1 件ぶんに一致するので、キャッシュが積み上がっていく過程の別行という解釈と矛盾しない。
- 失敗時の数字（allocated 12.91 GiB / free 4.46 GiB / reserved 17.39 GiB）から、reserved と allocated の差は約 4.5 GiB。
  断片化が効いていた。

## 2. GPTQ が通って AWQ が落ちる理由

共通: パイプラインは `independent`（既定、`args/dataset_arguments.py:218`）。AWQ は
`modifiers/awq/__init__.py` の互換 shim が `[AWQ(transform), QuantizationModifier]` のリストを返し、
`independent` が各モディファイアを個別に回す。AWQ の方は `requires_calibration_data=True` なので sequential、
QuantizationModifier(W4A16) は data-free。サブグラフは既定で `no_split_params`（デコーダ層）＝1 層 1 サブグラフ、
`sequential_targets_per_subgraph=1`、層間活性化（`IntermediatesCache`）の退避先は `sequential_offload_device="cpu"`。
ここまでは GPTQ と AWQ で同じ。

AWQ だけの差（`modifiers/transform/awq/base.py`）:

| 処理 | 場所 | GPU 常駐量（seq1024・256 件・bf16） |
|---|---|---|
| 親モジュール（self_attn / linear_attn / mlp）の forward 引数を全バッチ分キャッシュ | `_setup_activation_cache_hooks` の `IntermediatesCache(None, self.offload_device)` | 1 親あたり約 2.3 GiB（hidden 8 MiB×256 + 付随引数）。**`offload_device` の既定は None＝GPU のまま**（非 MoE。`on_initialize`） |
| 平滑化前の出力 `fp16_outputs` を全バッチ分リストで保持 | `_apply_smoothing` → `_run_samples` | 約 2.0 GiB |
| グリッド探索（n_grid=20）の各点で全バッチを再実行し `int_w_outputs` を保持 | `_compute_best_scale` | 約 2.0 GiB（`fp16_outputs` と同時） |
| 元重みの clone | `orig_layer_weights` | 数百 MiB |

GPTQ は GPU に持つのは Hessian（Linear ごと）程度で、キャリブレーションのバッチ数に比例する GPU 常駐データが無い。
同じモデル配置・同じデータでも AWQ だけが落ちることと整合する。docstring（`base.py`）にも
「OOM になるなら offload_device=cpu を検討」とある。

さらに配置の問題: `oneshot` は `device_map="auto"` で読んだモデルを `from_accelerate` で CT オフロードに変換する
（`entrypoints/utils.py:89`）。パイプラインは `set_onload_device(model, cuda)` を呼ぶが、**既に GPU 上にあるモジュールの
オフロード先は GPU のまま**（`dispatch.py` の優先順位 2）。BF16 で約 19 GB のモデルが L4 に収まる配置になっていれば、
重みが常駐して AWQ の作業領域は 3 GiB 前後しか残らない。失敗時の allocated 12.91 GiB がこれと整合するかは、
`hf_device_map` が記録されていないので**確認できていない**（未確認事項）。

mappings の auto 推定は OOM の原因ではない。0.15.0 には `Qwen3_5ForConditionalGeneration` 用の動的マッピング
（`dynamic_mappings.py:227`、`build_hybrid_attention_mappings`）があり、層種別（full / linear attention）に応じて
`input_layernorm → q,k,v` / `linear_attn の in_proj_*`、`post_attention_layernorm → gate,up`、`up → down` を作る。

## 3. 採用した対策と根拠・メモリ見積もり

| # | 対策 | 実在の API（0.15.0） | 数値への影響 |
|---|---|---|---|
| 1 | `AWQModifier(offload_device=torch.device("cpu"))` | `base.py:149` の公開フィールド。shim 経由でも `awq_kwargs` として渡る | なし（保管場所のみ） |
| 2 | AWQ のみ `from_pretrained(device_map="cpu")` | transformers の標準引数。CPU 上のモジュールは `set_onload_device` で「CPU オフロード＋cuda オンロード」になり、サブグラフごとに `subgraph_onload_modules` で GPU に載る | なし（演算は同じ GPU カーネル） |
| 3 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（`setdefault`、AWQ のみ） | PyTorch の標準設定 | なし |
| 4 | `oneshot(sequential_offload_device="cpu")` を明示 | `oneshot` のシグネチャに実在（`entrypoints/oneshot.py:427`）。既定値と同値 | なし |

見積もり（GPU、サブグラフ 1 層の smoothing 中のピーク。実測ではない）:

| 項目 | GiB |
|---|---|
| 当該デコーダ層の重み（約 0.2B パラメータ × bf16） | 約 0.5 |
| サブグラフ外で常駐する Vision タワー（`find_modules_outside_subgraphs` の警告対象になりうる） | 約 1.0 |
| `fp16_outputs`（256 × 8 MiB） | 2.0 |
| `int_w_outputs`（同上） | 2.0 |
| `orig_layer_weights` clone・MLP 中間活性化・プリフェッチ | 約 0.4 |
| CUDA コンテキスト | 約 0.4 |
| 合計 | 約 6.3 |

断片化・一時バッファに 2 倍の余裕を見ても約 13 GiB で、22 GiB に収まる見込み。最終サブグラフの logits
（1 件 0.47 GiB、fp32 化で最大約 1 GiB）はこの余裕内。
CPU 側は、モデル 約 19 GB ＋ パイプライン層間キャッシュ 約 2.3 GiB ＋ AWQ 親キャッシュ（attn と mlp で約 4.6 GiB）
＝ 約 27 GB 前後。**Colab の RAM が 25 GB 級の標準ランタイムだと不足しうる**（L4 の高 RAM ランタイムなら通る見込み）。

## 4. 却下した代替案

- **キャリブレーションの縮小（件数・系列長）**: 不変条件違反（GPTQ vs AWQ の直接比較が崩れる）。
- **`sequential_targets` の細分化（`Linear` 等）**: AWQ は親モジュール単位で forward を再実行するため、
  サブグラフを Linear 単位にするとマッピングの親（self_attn / mlp）が 1 つのサブグラフに収まらず、
  `_resolve_sequential_targets` も attention を必ず外側ターゲットに足すので効果が薄い。レシピ実質にも触れる恐れ。
- **`sequential_targets_per_subgraph`**: 既定が最小の 1。減らせない。`oneshot()` のシグネチャにも無い。
- **`pipeline="sequential"` の明示**: shim のリスト（AWQ＋Quantization）を 1 パスで回すことになり挙動が変わる。不要なので触らない。
- **mappings の手動指定**: 0.15.0 が Qwen3.5 用の動的マッピングを持つため不要。不変条件（既定のまま）にも反する。
- **`duo_scaling=False` / `n_grid` 縮小**: 一時メモリはほぼ変わらず、AWQ の結果が変わる。却下。
- **Vision を GPU に載せない読み込み**: `device_map="cpu"` で包含される。個別の `device_map` dict は組まない
  （モジュール名が transformers のバージョン依存で、誤ると別の OOM を生むため）。
- **`load_offloaded_model` / `load_context()` の利用**: 0.15.0 の推奨経路だが、ロード挙動が GPTQ（B2）と変わる。
  今回の最小変更には不要と判断。
- **A100 への切替**: ユーザーが選ばなかった。

## 5. B2 との比較条件がずれていないことの確認

- キャリブレーション: `--calib-packed calib/hybrid_256x1024.json` の経路・読み込み・検証コードは未変更
  （`samples_sha256` は `cc3f7050…`、`quant_meta.json` にも従来どおり記録される）。`max_seq_length=1024`、
  `num_calibration_samples=256` も同じ。テストで、AWQ 時に増える oneshot 引数は `sequential_offload_device` のみで、
  それ以外のキーと値は GPTQ と一致することを検証している。
- レシピ: `W4A16` / `targets=Linear` / `ignore=IGNORE`（lm_head・visual・vision）。AWQ の `mappings` は未指定（既定）。
  テストで `mappings` が渡されないことを検証している。
- 変更はいずれも「どこに置くか」（キャッシュ・重みの保管先、アロケータ設定）であり、演算内容は変わらない。
  ただし CPU↔GPU の転送経由になるため、**bf16 の丸めまで含めて B2 と同じ手順で再現できることは保証していない**
  （転送は値を変えないが、プリフェッチ用の別ストリームを使う）。実質的な比較条件のずれは無い。
- B2 側の経路（`device_map="auto"`、環境変数、`oneshot` の引数）は不変。GPTQ では `model_device_map` が `"auto"` を返す。
- `results_main/`・`calib/`・ノートブックは変更していない。ノートブックの B3 量子化コマンドは引数追加が不要だった。

## 実機で確認してほしい点（未確認事項）

1. `device_map="cpu"` のモデルが oneshot の `from_accelerate`（`hf_device_map={"": "cpu"}`）を通って、
   サブグラフ単位に cuda へオンロードされること。ログに GPU 使用量が層ごとに上下するか、`nvidia-smi` で確認。
2. 想定ピーク（約 6〜13 GiB）に対する実測。もし依然として OOM なら、完全なトレースバック
   （どの行の確保か）と `torch.cuda.memory_summary()` を取ること。
3. Colab の RAM（約 27 GB 前後を想定）。
4. 動的マッピング（Qwen3.5 用）の解決が通ること。ログに
   `Built dynamic hybrid attention AWQ mappings: … linear projections: …` が出るか、
   `Skipping … mapping` の警告が大量に出ていないか。
5. shim 経由（`modifiers.awq.AWQModifier`、`DeprecationWarning`）の `[AWQ, QuantizationModifier]` が実機で通ること。
   B3 が実際にここまで進んだのは初めてである。
6. AWQ 完了後の `save_pretrained(save_compressed=True)` と `quantized` 層数の検証が通ること。
7. 所要時間: CPU オフロードとグリッド探索（n_grid=20）のため B2（855 秒）より長くなる見込み。
