#!/usr/bin/env bash
# GGUF 変換・量子化・imatrix 生成。条件 ID を引数に取る:   make_gguf.sh A0 | A3 | A3i  (複数可)
#   A0  : HF → F16 GGUF（言語モデル部分のみ。--mmproj は付けない）
#   A3  : Q4_K_M（標準）
#   A3i : Q4_K_M（imatrix。F16 から c4_en_ja_imatrix.txt で生成, --chunks 32）
# A1 Q8_0 / A2 Q5_K_M / A4 Q3_K_M / A4i Q3_K_M+imatrix / A5 Q2_K / A5i Q2_K+imatrix も同様（Phase 1 で使用）。
# 出力は *.partial.* に書いてから mv する（途中で落ちた不完全ファイルを「あり」と誤認して飛ばさないため）。
# 環境変数: LLAMA_DIR / MODEL_DIR / OUT_DIR / IMATRIX_TEXT / IMATRIX_CHUNKS
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-llama.cpp}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-models/gguf}"
IMATRIX_TEXT="${IMATRIX_TEXT:-data/imatrix_calib.txt}"
IMATRIX_CHUNKS="${IMATRIX_CHUNKS:-32}"
BIN="$LLAMA_DIR/build/bin"
F16="$OUT_DIR/qwen3.5-9b-F16.gguf"
IMAT="$OUT_DIR/imatrix_c4_en_ja_${IMATRIX_CHUNKS}chunks.dat"
mkdir -p "$OUT_DIR"

ensure_f16() {
  [ -f "$F16" ] && return 0
  local tmp="${F16%.gguf}.partial.gguf"
  rm -f "$tmp"
  python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype f16 --outfile "$tmp"
  mv "$tmp" "$F16"
}
ensure_imatrix() {
  [ -f "$IMAT" ] && return 0
  ensure_f16
  local tmp="${IMAT%.dat}.partial.dat"
  rm -f "$tmp"
  "$BIN/llama-imatrix" -m "$F16" -f "$IMATRIX_TEXT" -o "$tmp" --chunks "$IMATRIX_CHUNKS" -ngl 99
  mv "$tmp" "$IMAT"
}
quant() {  # quant <TYPE> [imatrix]
  local type="$1" imat="${2:-}" out
  if [ -n "$imat" ]; then out="$OUT_DIR/qwen3.5-9b-$type-imatrix.gguf"; else out="$OUT_DIR/qwen3.5-9b-$type.gguf"; fi
  if [ -f "$out" ]; then echo "exists: $out"; return 0; fi
  ensure_f16
  local tmp="${out%.gguf}.partial.gguf"
  rm -f "$tmp"
  if [ -n "$imat" ]; then
    ensure_imatrix
    "$BIN/llama-quantize" --imatrix "$IMAT" "$F16" "$tmp" "$type"
  else
    "$BIN/llama-quantize" "$F16" "$tmp" "$type"
  fi
  mv "$tmp" "$out"
}

[ "$#" -ge 1 ] || { echo "usage: $0 <A0|A3|A3i|...>..." >&2; exit 2; }
for c in "$@"; do
  case "$c" in
    A0)  ensure_f16 ;;
    A1)  quant Q8_0 ;;
    A2)  quant Q5_K_M ;;
    A3)  quant Q4_K_M ;;
    A3i) quant Q4_K_M imatrix ;;
    A4)  quant Q3_K_M ;;
    A4i) quant Q3_K_M imatrix ;;
    A5)  quant Q2_K ;;
    A5i) quant Q2_K imatrix ;;
    *) echo "unknown condition: $c" >&2; exit 2 ;;
  esac
done
ls -l "$OUT_DIR"
