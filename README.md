# quantization-exp

日本語 LLM の量子化による精度劣化実測（Qwen3.5-9B / GGUF / vLLM）の実験コード。
設計の正は [`docs/quantization-verification-design.md`](docs/quantization-verification-design.md)（v3 確定）。
Phase 0（スモーク）／Phase 1（Track A 本測定）／Phase 2（Track B 本測定）までを含む。

## 目的

量子化で日本語性能がどれだけ落ちるかを、同一条件（プロンプト・greedy・thinking OFF・ctx 8192）で測る。
Phase 0 は本測定の前にハーネスの形式・採点・所要時間を確認するための 4 条件だけの試走。

| ID | 条件 |
|---|---|
| A0 | GGUF F16（基準） |
| A3 | GGUF Q4_K_M（標準） |
| A3i | GGUF Q4_K_M（imatrix） |
| B1 | FP8 dynamic（llm-compressor → vLLM、言語モデルのみ） |

評価: JCommonsenseQA（accuracy）／JSQuAD（F1・EM）／MGSM（数値 exact match）各先頭 100 件、日本語 perplexity、定性 10 プロンプト。

## Colab での実行手順

1. **GitHub 認証（private リポジトリのため）**
   - GitHub → Settings → Developer settings → Fine-grained personal access tokens で、**このリポジトリのみ**・`Contents: Read-only` のトークンを作る
   - Colab の左メニュー「🔑 シークレット」に名前 `GITHUB_TOKEN` で登録し、「ノートブックからのアクセス」をオンにする
   - 認証操作（Google／GitHub のサインイン）は自分で公式画面から行う
2. `colab/phase0_smoke.ipynb` を Colab で開く（GitHub タブからの読み込み、またはファイルをアップロード）
3. **ランタイム → ランタイムのタイプを変更 → L4 GPU**
4. 上から順にセルを実行する（セル 0 が `git clone`、以降は `scripts/` を呼ぶだけ）
5. 最後のセルが `results/` を zip にしてダウンロードする。Colab のディスクはセッション終了で消えるので必ず保存する

> 注意: Colab の無料枠で L4 が選べるとは限らない（無料枠は通常 T4・16GB）。L4 は有料プラン／コンピューティングユニットが必要になる場合がある。
> 課金なしで進める場合は T4 で載るか（BF16 約 19GB は載らない）の判断が必要で、設計変更に当たるため実装では対応していない。

## Phase 0 の所要時間の測り方

重い工程は `python scripts/run_timed.py <ラベル> -- <コマンド>` 経由で実行され、`results/timings.tsv`
（ラベル／秒／分／終了コード／開始時刻）に追記される。ダウンロードはノートブック内で同形式に記録。
工程の内訳（ダウンロード／ビルド／量子化／評価）はこのファイルを見ればよい。

## 結果の見方

```
results/
  summary.tsv        条件／種別／ファイルサイズGB／JCSQA acc／JSQuAD F1／MGSM acc／PPL／tok/s／VRAM GB
  timings.tsv        工程ごとの所要時間
  <条件>.json        条件名・モデル情報・バージョン類・件数・各指標・生成結果へのパス
  <条件>/            gen_<task>.jsonl（生出力）, qualitative.jsonl, server.log, perplexity.log
  env/               llama.cpp のビルド番号、nvidia-smi、モデル revision
```

- 指標は 0–1 の割合。`tok/s` は定性 10 プロンプトを **1 件ずつ逐次生成**したときの値で、**Track 内でのみ比較する**（GGUF と vLLM を直接比べない）
- `VRAM GB` は評価中の GPU 使用メモリ総量のピーク（nvidia-smi）。vLLM は `gpu_memory_utilization` 分を先取りするため GGUF と直接比較しない
- `ファイルサイズGB`（B1）は `*.safetensors` の合計で、BF16 のまま残した Vision Encoder の重みを含む
- **thinking OFF の確認**: ノートブックのセル 8 が `qualitative.jsonl` の生出力を表示する。`<think>` や長い思考過程が出ていれば OFF が効いていない
- PPL: GGUF は `llama-perplexity`（ctx 2048、後半採点）、vLLM は同方式を `prompt_logprobs` で自前実装。細部が異なるため **Track をまたぐ PPL の絶対比較はしない**
- 再現性: seed=0、llama.cpp ビルド番号・vLLM／llm-compressor バージョン・量子化パラメータは各 `<条件>.json` と `results/env/` に記録される

## Phase 1（本測定 Track A）

`colab/phase1_main_a.ipynb`（依頼: [`docs/phase1-claude-code-handoff.md`](docs/phase1-claude-code-handoff.md)）。GGUF 9 条件を
JCSQA 200 / JSQuAD 200 / MGSM 全 250 件で測る。「すべてのセルを実行」で完走し、`results_main/<条件>.json` がある条件は飛ばす（再開可）。
データは `data_main/`、結果は `results_main/`（Phase 0 の `data/`・`results/` とは分離）。llama-server のポートは 18080。
最後に `results_main/` を GitHub へ push するため、`GITHUB_TOKEN` は **Contents: Read and write** にしておく（読み取りのみだと push が失敗し zip 退避になる）。

評価スクリプトの `--n 0` は「データファイルの全行」を評価する（既定 100 は Phase 0 のまま）。

## Phase 2（本測定 Track B）

`colab/phase2_main_b.ipynb`。vLLM＋llm-compressor で B0 BF16／B1 FP8 dynamic／B2 GPTQ W4A16／B3 AWQ W4A16 の
4 条件を、Track A と同一の評価条件（JCSQA 200／JSQuAD 200／MGSM 250）で測る。GPTQ／AWQ の
キャリブレーションは `calib/hybrid_256x1024.json`（256 件×1024 トークン。生成方法と出典は同梱の
`.meta.json` に記録）。結果は `results_main/summary_b.tsv`。

## データの出典

- 評価データ: JCommonsenseQA／JSQuAD（JGLUE）、MGSM（google-research/url-nlp の mgsm_ja）
- Track A の imatrix 生成: TFMC/imatrix-dataset-for-japanese-llm の `c4_en_ja_imatrix.txt`
- Track B のキャリブレーション補充: HuggingFaceFW/fineweb-2（jpn_Jpan、ODC-By 1.0）。取得条件・除外数などの詳細は `calib/hybrid_256x1024.meta.json`

## ローカルでできる確認（GPU 不要）

```
python3 -m unittest discover -s scripts -p 'test_*.py'   # 全テスト
python3 scripts/test_score.py       # 採点コードの単体テスト
python3 scripts/test_pipeline.py    # データ整形と評価ループの配管テスト（ダミー生成器）
python3 scripts/test_make_gguf.py   # make_gguf.sh の冪等性テスト（スタブ）
```

## ディレクトリ

| パス | 内容 |
|---|---|
| `colab/phase0_smoke.ipynb` | Phase 0 を完走させるノートブック |
| `scripts/build_llama.sh` | llama.cpp の clone と CUDA ビルド、ビルド番号の記録 |
| `scripts/make_gguf.sh` | GGUF 変換・量子化・imatrix 生成（条件 ID を引数に取る） |
| `scripts/quantize_fp8.py` | llm-compressor による FP8 dynamic 量子化（B1） |
| `scripts/prep_data.py` | 評価データ取得・整形・先頭 N 件への絞り込み、PPL 用テキスト作成 |
| `scripts/common.py` | プロンプト整形・生成設定・評価ループ（GGUF／vLLM 共通） |
| `scripts/eval_gguf.py` / `eval_vllm.py` | 各ランタイムでの評価 |
| `scripts/score.py` / `test_score.py` | 採点と単体テスト |
| `scripts/prompts_qualitative.txt` | 定性評価の固定 10 プロンプト |
| `scripts/run_timed.py` / `make_summary.py` | 所要時間の記録／summary.tsv の生成 |

## ライセンス

コード（スクリプト、ノートブック）は MIT License（`LICENSE` ファイル）です。
ただし、第三者のデータセットに由来する内容は MIT の対象外で、それぞれの元のライセンスに従います。

- `calib/` のキャリブレーションテキスト: TFMC/imatrix-dataset-for-japanese-llm（c4 由来）と HuggingFaceFW/fineweb-2（ODC-By 1.0）由来のテキストを含む
- `results/`、`results_main/` の生成結果に含まれる評価データの問題文・文脈: JCommonsenseQA／JSQuAD（JGLUE）および MGSM の各ライセンスに従う
