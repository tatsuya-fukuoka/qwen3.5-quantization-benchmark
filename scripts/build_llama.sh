#!/usr/bin/env bash
# llama.cpp を最新 main から clone し、CUDA 付きでビルドする。ビルド番号を results/env/llama_build.txt に記録。
# 環境変数: LLAMA_DIR(既定 llama.cpp) / CUDA_ARCH(既定 89 = L4。A100 は 80。"native" も可)
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-89}"
RESULTS_DIR="${RESULTS_DIR:-results}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
  # ビルド番号は git のコミット数から決まるため shallow clone にしない
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  git -C "$LLAMA_DIR" pull --ff-only
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" \
  --target llama-server llama-quantize llama-imatrix llama-perplexity

# 変換スクリプト用の Python 依存（torch は入れ替えない）
pip install -q -e "$LLAMA_DIR/gguf-py" transformers sentencepiece protobuf safetensors numpy

mkdir -p "$RESULTS_DIR/env"
{
  echo "commit: $(git -C "$LLAMA_DIR" rev-parse HEAD)"
  echo "commit_date: $(git -C "$LLAMA_DIR" log -1 --format=%cI)"
  echo "build_number: $(git -C "$LLAMA_DIR" rev-list --count HEAD)"
  echo "cuda_arch: $CUDA_ARCH"
  "$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | tail -n 3
} | tee "$RESULTS_DIR/env/llama_build.txt"
