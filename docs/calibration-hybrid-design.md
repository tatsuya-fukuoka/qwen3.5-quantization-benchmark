# ハイブリッド・キャリブレーション（256 サンプル × 1024 トークン）設計書

作成日: 2026-11-09 / 依頼: `docs/phase2-hybrid-calibration-handoff.md` / 土台: `docs/calibration-512-investigation.md`、`docs/calibration-256-investigation.md`、`docs/llmcompressor-0150-calibration-source-check.md`
範囲: 設計と実装・テストまで。Colab は実行していない。実データ（fineweb-2・Qwen トークナイザ・`imatrix_calib.txt`）はこの VM では取得・実行していない（後述の「未確認」）。

## 1. 方式の要約

1. 既存コーパス `data_main/imatrix_calib.txt`（120 文書、文書ごとトークナイズの合計 93,306 トークン）を**全量**使う。
2. 不足分を `HuggingFaceFW/fineweb-2` の `jpn_Jpan` で補充する。
3. `既存（先頭）＋補充` を区切りなしで 1 本のトークン列に連結 → 1024 トークンごとに切る（端数破棄）→ 256 チャンクを確定 → 256 × 1024 = **262,144 トークン**。
4. B2（GPTQ）／B3（AWQ）は同一のパック済みファイルを読む。

## 2. 補充ソースの選定

| | `HuggingFaceFW/fineweb-2` `jpn_Jpan`（採用） | `wikimedia/wikipedia` `20231101.ja` |
|---|---|---|
| ライセンス | ODC-By 1.0（Hub API の `cardData.license` = `odc-by` を確認。gated でない） | CC BY-SA 3.0 / GFDL（同 API で確認）。継承条件あり |
| 評価データとの重複リスク | 低い（Web 全般。ただし 0 ではないため §4 の検査を入れる） | **高い**。JSQuAD の context は日本語 Wikipedia 由来で、`ppl_ja.txt` も JSQuAD valid の context から作っている。評価データそのものを補充に混ぜる恐れ |
| 文体の近さ | 既存コーパス（C4 由来の Web テキスト）と同系統 | 百科事典調で偏る |
| 取得 | streaming 可。データは 175 シャード・約 717GB（Hub API の tree で確認）なので**全量取得は不可**、1 シャードに固定して streaming | 同様に streaming 可 |

**採用: fineweb-2。** 決め手は (1) 評価データとの重複リスクが構造的に小さい、(2) 既存コーパス（Web テキスト）と文体が近い、(3) ライセンスの継承条件が緩い。Wikipedia は評価（JSQuAD・PPL）と同一出典のため、キャリブレーションへの評価データ混入を避けたい本実験の制約と相性が悪い。

### Colab での取得方法とダウンロード量

- `load_dataset("HuggingFaceFW/fineweb-2", name="jpn_Jpan", data_files="data/jpn_Jpan/train/000_00000.parquet", split="train", streaming=True, revision=<解決した sha>)` → `.shuffle(seed=0, buffer_size=1000)`。**`name="jpn_Jpan"`（config 名）が必須**: 省略すると `ValueError: BuilderConfig 'default' not found` で失敗する（Colab 実機で発生、datasets 5.1.0 の VM で再現・修正確認済み）。
- 1 シャード（約 4.8GB の parquet）を指定するのは、全シャードを対象にするとシャードの並びのシャッフルで取得先が散らばるのを避けるため。streaming なので必要分のみ読む。
- 必要トークン量（§3）は約 23.4 万トークン ≒ 数百 KB のテキスト。バッファ 1000 文書を満たすため、実際のダウンロードは「バッファ充填分＋parquet の行グループ単位」になる。**実ダウンロード量は未測定**（数十 MB 程度を見込むが、これは推定であり確認していない）。
- revision は `HfApi().dataset_info(...).sha` で解決して記録する（本日の確認値は `af9c1333…`）。再構築時は `--hf-revision` で固定できる。

## 3. 必要トークン量

- 最低限の補充量: 262,144 − 93,306 = **168,838 トークン**（テスト `test_real_scale_arithmetic` で算術を固定）。
- 既存コーパスを全量使うので、既存分を含むチャンク `ceil(93,306 / 1024) = 92` 個は**必須チャンク**（先頭 92 チャンク。最後の 1 つは既存の末尾と補充の先頭が同居）。残り 164 チャンクを補充プールから seed 固定で抽出する。
- **プールに余裕を持たせる（採用）**: プールが 256 + 64 = **320 チャンク**（= 327,680 トークン）に達するまで補充文書を取る。補充側のプールは 228 チャンクで、そこから 164 を抽出。取得トークン量は最低 327,680 − 93,306 = 234,374（必要量の約 1.4 倍）。
  - 理由: (1) 抽出に自由度ができ、サンプルがストリーム先頭の文書列に偏らない。(2) 評価データとの重複検査・短文除外・重複除外で文書が落ちても不足しにくい。(3) 余分な取得は約 6.6 万トークンで小さい。
  - 余白 64 は経験的な値で、最適性は検証していない。
- ストリームが尽きて一意チャンクが 256 未満なら `ValueError`（プール不足で失敗。黙って水増ししない）。
- 重複・オーバーラップによる水増しはしない: チャンクは重ならず、内容が同一のチャンクは 1 つに畳む（畳んだ結果、既存分が欠ければ失敗）。

## 4. 連結順・決定性・固定する条件

| 項目 | 決定 |
|---|---|
| 連結順 | 既存コーパス（ファイル順）→ 補充文書（シャッフル後のストリーム順）。文書間に区切りトークンは入れない（§3 の 168,838 の算術と一致させるため） |
| 既存コーパスの文書分割 | 空行を除く各行を 1 文書。フィルタ・重複除去・切り詰めなし（全量） |
| 補充のシャッフル | `seed=0`、`buffer_size=1000`、シャード `000_00000.parquet` 固定 |
| 補充文書のフィルタ | 200 文字未満は除外／同一文面の重複は除外／先頭 3000 文字で切る（1 文書がプールを占有しないため）／評価データと重複するものは除外 |
| 評価データとの重複検査 | 評価データ（`ppl_ja.txt`、`jcommonsenseqa.jsonl`、`jsquad.jsonl`、`mgsm.jsonl` の全文字列）の 30 文字 n-gram 集合を作り、1 つでも一致した補充文書を除外。除外数は meta に記録 |
| トークナイズ | 各文書を `add_special_tokens=False` で個別にトークナイズして連結（BOS/EOS なし）。Qwen3.5-9B のトークナイザ |
| チャンク化 | 1024 固定、端数（プール末尾）は破棄。チャンクは文書境界をまたぐ（連結方式の仕様） |
| 抽出 | 必須（先頭 92）＋ `random.Random(0).sample` で 164。出力は元の位置順 |
| 評価データ | 補充には使わない（上記検査）。**既存コーパス側と評価データの重複は今回検査していない（未確認、§9）** |

### 記録する項目（`<out>.meta.json`）

方式、サンプル数・系列長・総トークン数、seed、pool_margin、プールのチャンク数、破棄した端数トークン数、採用チャンクの添字、既存コーパス（ファイル名・sha256・文書数・トークン数・使用数）、補充（データセット名・config・split・データファイル・**revision**・シャッフル seed／バッファ・ライセンス・取得トークン数・使用数・消費文書数・除外数内訳・**消費文書 ID と各トークン数**・重複検査の対象）、トークナイザ（クラス名・語彙サイズ・transformers 版・`tokenizer.json` の sha256）、`packed_sha256`、`samples_sha256`（各 ID を int32 LE で並べた sha256）。

fineweb-2 の文書 ID は `id` 列（UUID）を記録する。`id` 列は実ストリーミングで存在を確認済み（形式は `<urn:uuid:...>`、他に `text`・`url`・`dump` 等の列）。

## 5. パック済みファイルの保存方針（再現性の担保）

**採用: GitHub に保存し、ランタイム再開時は検証のうえ再利用する。**

- 形式: `calib/hybrid_256x1024.json` = `{"format":"packed-token-ids-v1","seq_len":1024,"num_samples":256,"samples":[[...]x256]}`（空白なし）。262,144 個の整数で、概算 1.7〜2MB（実測前）。別途 `calib/hybrid_256x1024.meta.json`（上記メタ）。
- `.gitignore`: `calib/` は無視されない（`data_main/` は無視のまま。`scripts/test_calibration_hybrid.py::TestGitignore` で固定）。`.gitignore` の変更は不要。
- ノートブック 5b: ファイルがあれば `build_calibration.py --verify-only`（`packed_sha256`・`samples_sha256`・既存コーパス sha256・`tokenizer.json` sha256 を照合）。OK なら構築せず再利用、NG なら**上書きせず停止**。無ければ構築 → `push_paths` で push（失敗は警告のみ）。
- 採用理由: (1) fineweb-2 の streaming は revision とシャッフル seed を固定しても、ライブラリ版差などで同一列になる保証が無い。ファイルを残せば B2・B3 が別ランタイムでも同一サンプルを使える。(2) 小さい。(3) どの文書を使ったかの監査証跡になる。
- 留意: トークン ID 列は fineweb-2（ODC-By 1.0）由来の派生物。リポジトリは private。公開する場合は帰属表示（データセット名・revision・文書 ID は meta にある）を付ける。
- 不採用案: 毎回ストリームから再構築する（再現性が弱い）。

## 6. 実装の変更一覧

| ファイル | 内容 |
|---|---|
| `scripts/calib_pack.py`（新規） | 純関数（torch 不要）: `concat_ids`／`chunk_fixed`（端数破棄）／`select_chunks`（内容重複除去・必須チャンク＋seed 抽出・不足で `ValueError`）／`collect_supplement`（フィルタ・打ち切り）／`build_hybrid`／`samples_digest_ids`／重複検査 |
| `scripts/build_calibration.py`（新規） | 入力（既存コーパス＋ fineweb-2 streaming または JSONL）→ パック済み JSON＋メタ。`--verify-only` 付き。`--dummy-tokenizer`（テスト用） |
| `scripts/quantize_w4a16.py` | `--calib-packed` 追加（`load_packed_calibration`／`packed_calibration_meta`）。パック済みから `input_ids`／`attention_mask` の Dataset を直接作る。`--calib-text` 経路は後方互換で残す。`--max-seq-len` 既定を `None` に（行ベース経路では従来どおり 512、パック経路では seq_len に固定） |
| `colab/phase2_main_b.ipynb` | 冒頭説明・§4 の説明を更新、**§5b を新設**（検証／構築／push）、B2・B3 を `--calib-packed` に切替。B0／B1 のセル・push 機構・`summary_b.tsv`／`timings_b.tsv` は変更なし |
| `scripts/test_calibration_hybrid.py`（新規） | 16 件（下記） |
| `docs/calibration-hybrid-design.md`（本書） | |

`quant_meta.json` の `calibration` には、`method: hybrid-pack`、`num_samples: 256`、`seq_len/max_seq_len: 1024`、`total_tokens`、`existing_tokens`／`supplement_tokens`、`seed`、`packed_sha256`、`samples_sha256`、出典（既存ファイル名・sha256、補充のデータセット・revision・文書数）、トークナイザを記録する。

触れていないもの: `results_main/B0.json`・`B1.json`、Track A の成果物、`scripts/prep_data.py`（`--with-imatrix` で `imatrix_calib.txt` を得る既存の仕組みをそのまま使う）。

## 7. テスト

既存 35 件＋新規 16 件＝**51 件**。新規は依頼の (a)〜(f) に対応: (a) 同入力・seed で再現、(b) 全サンプルちょうど seq_len、(c) 端数破棄、(d) プール不足で例外、(e) 重複チャンクなし、(f) ダミートークナイザでの `build_calibration.py` 経路（疎通・メタ整合・`--verify-only` の OK／改ざん検出）。加えて、既存全量使用と出典内訳、必須チャンクが n 超で例外、補充フィルタ（短文・重複・評価データ重複）、`quantize_w4a16` のパック読み込みと形状不正の拒否、`quant_meta` 用 calibration 欄、`.gitignore`、ノートブックが B2／B3 で新経路を使うこと。

## 8. 設計書 v3 への修正文案と記事の説明文案

**v3 の修正文案（Track B のキャリブレーション記述）**

> 変更前: キャリブレーション 512 サンプル（`imatrix_calib.txt` から抽出）
> 変更後: キャリブレーションは **256 サンプル × 1024 トークン**（262,144 トークン）。Track A の imatrix と同じ日英混在コーパス（C4 由来、120 文書、約 9.3 万トークン）を全量使い、不足分を `HuggingFaceFW/fineweb-2`（日本語）で補充した**ハイブリッド**を、連結して 1024 トークンごとに切り seed 固定で確定する。GPTQ（B2）と AWQ（B3）は完全に同一のサンプル集合を使う。

**記事での説明文（2〜3 文）**

> GPTQ と AWQ のキャリブレーションには、256 サンプル × 1024 トークン（計 262,144 トークンで、GPTQ 論文の総トークン量と同量）を使いました。素材は、imatrix にも使った日英混在コーパス（約 9.3 万トークン）を全量と、不足分を公開日本語 Web テキスト（FineWeb-2）で補ったものです。GPTQ と AWQ では完全に同じサンプルを使い、評価データと重複する文書は補充から除いています。

## 9. Track A との関係（正直な記述）

Track A の imatrix は `c4_en_ja_imatrix.txt` のみ、Track B のキャリブレーションはそのコーパス＋ FineWeb-2 で、**出典は完全には同一でない**（B 側に約 64% の補充が入る: 168,838 / 262,144）。記事の文案:

> 注意: Track A（GGUF の imatrix）と Track B（GPTQ／AWQ）のキャリブレーションデータは完全には同じではありません。共通の日英混在コーパスを両方で使っていますが、Track B ではそれだけでは量が足りないため、日本語 Web テキストで約 6 割を補っています。そのため A と B の差は「量子化方式の差」だけでなく「キャリブレーションデータの差」も含みます。

B2 と B3 の間は完全に同一なので、B2 vs B3 の比較には影響しない。

## 10. 未確認事項

- fineweb-2 の実ストリーミング: 2026-11-09 に VM（datasets 5.1.0）で先頭 3 件の取得まで確認済み（`name="jpn_Jpan"`＋`data_files` 併用、`id`・`text` 列あり、shuffle 動作）。**未確認**: 実ダウンロード量、`datasets` 版差（Colab 側の版）による同一列性、補充に要する実消費文書数。
- 実トークナイザ（Qwen3.5-9B）での実測: 既存コーパスが 93,306 トークン・120 文書になるか（行分割の定義が前回調査と一致するか）、パック済みファイルの実サイズ、補充の消費文書数。ノートブック 5b は既存コーパスが全量使われることを assert する。
- 既存コーパス（C4 由来）と評価データの重複（この VM に `data_main/` なし）。補充側のみ検査している。
- 評価データ重複検査は 30 文字 n-gram の完全一致のみ。言い換え・翻訳版は検出しない。
- 0.15.0 実機での挙動: トークナイズ済み Dataset の直接渡し（ソース上は問題なしと確認済み）、1024 トークン × 256 サンプルでの L4 メモリ・所要時間（GPTQ／AWQ とも未実測）、AWQ の mappings 自動推定が Qwen3.5 で通るか（従来からの未確認事項）。
- 精度への影響（256 × 1024 のハイブリッドが十分か、補充比率の影響）は未測定。
- 文書境界に区切りトークンを入れない仕様（チャンクが文書をまたぐ）の影響は未評価。
