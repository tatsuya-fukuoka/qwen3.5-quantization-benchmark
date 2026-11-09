#!/usr/bin/env python3
"""ハイブリッド・キャリブレーションのパッキング（純関数。torch / transformers / datasets 不要）。

流れ: 文書ごとのトークン ID 列のリスト → 連結 → 固定長チャンク（端数破棄）→ 重複除去 → seed 抽出。
既存コーパス（全量使用）が先頭に来るため、先頭 keep_first 個のチャンクは必ず採用し、残りを seed で抽出する。
"""
from __future__ import annotations

import hashlib
import random
import struct
from typing import Callable, Iterable, Iterator

SEQ_LEN = 1024
NUM_SAMPLES = 256
SEED = 0
POOL_MARGIN = 64          # プールが NUM_SAMPLES + POOL_MARGIN チャンクに達するまで補充文書を取る
FORMAT = "packed-token-ids-v1"
SHINGLE = 30              # 評価データとの重複検査に使う文字 n-gram 長


def concat_ids(docs: list[list[int]]) -> list[int]:
    """文書境界に区切りトークンは入れない（単純連結）。"""
    out: list[int] = []
    for d in docs:
        out.extend(d)
    return out


def chunk_fixed(ids: list[int], seq_len: int = SEQ_LEN) -> list[list[int]]:
    """固定長 seq_len で切り、端数（seq_len 未満の末尾）は捨てる。"""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    n = len(ids) // seq_len
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]


def select_chunks(chunks: list[list[int]], n: int = NUM_SAMPLES, seed: int = SEED,
                  keep_first: int = 0) -> tuple[list[int], list[list[int]]]:
    """重複（内容一致）チャンクを除いたうえで n 個を選ぶ。先頭 keep_first 個は必ず採用、残りは seed 抽出。

    返り値は (元チャンクの添字リスト（昇順）, チャンクのリスト)。候補が n 未満なら ValueError。
    """
    seen: set[tuple[int, ...]] = set()
    uniq: list[int] = []
    for i, c in enumerate(chunks):
        t = tuple(c)
        if t in seen:
            continue
        seen.add(t)
        uniq.append(i)
    forced = [i for i in uniq if i < keep_first]
    rest = [i for i in uniq if i >= keep_first]
    if len(forced) > n:
        raise ValueError(f"必須チャンク {len(forced)} が n={n} を超える")
    if len(forced) + len(rest) < n:
        raise ValueError(f"プール不足: 一意なチャンク {len(forced) + len(rest)} < {n}")
    picked = sorted(forced + random.Random(seed).sample(rest, n - len(forced)))
    return picked, [chunks[i] for i in picked]


def samples_digest_ids(samples: list[list[int]]) -> str:
    """ID 列ベースの sha256（各 ID を int32 little-endian、サンプル間に区切りなし。全サンプル同長が前提）。"""
    h = hashlib.sha256()
    for s in samples:
        h.update(struct.pack(f"<{len(s)}i", *s))
    return h.hexdigest()


def shingle_set(texts: Iterable[str], k: int = SHINGLE) -> set[int]:
    out: set[int] = set()
    for t in texts:
        for i in range(len(t) - k + 1):
            out.add(hash(t[i:i + k]))
    return out


def is_contaminated(text: str, shingles: set[int], k: int = SHINGLE) -> bool:
    return any(hash(text[i:i + k]) in shingles for i in range(len(text) - k + 1))


def collect_supplement(docs: Iterable[dict], tokenize: Callable[[str], list[int]], existing_tokens: int,
                       seq_len: int = SEQ_LEN, target_chunks: int = NUM_SAMPLES + POOL_MARGIN,
                       min_chars: int = 200, max_chars: int = 3000,
                       shingles: set[int] | None = None) -> dict:
    """補充文書（{"id", "text"}）を順に消費し、既存分と合わせて target_chunks 個のチャンクが取れるまで集める。

    - 短い文書（min_chars 未満）・完全一致の重複・評価データと重複する文書は消費しない（skipped に計上）。
    - 長い文書は先頭 max_chars 文字に切る（1 文書がプールを占有しないため）。
    - 打ち切り: (existing_tokens + 補充トークン) // seq_len >= target_chunks。ストリームが尽きたらそこで終了。
    """
    used: list[dict] = []
    ids_docs: list[list[int]] = []
    skipped = {"short": 0, "duplicate": 0, "contaminated": 0}
    seen: set[str] = set()
    total = existing_tokens
    for d in docs:
        if total // seq_len >= target_chunks:
            break
        text = d["text"].strip()
        if len(text) < min_chars:
            skipped["short"] += 1
            continue
        text = text[:max_chars]
        if text in seen:
            skipped["duplicate"] += 1
            continue
        if shingles is not None and is_contaminated(text, shingles):
            skipped["contaminated"] += 1
            continue
        seen.add(text)
        ids = tokenize(text)
        ids_docs.append(ids)
        used.append({"id": d["id"], "tokens": len(ids)})
        total += len(ids)
    return {"docs": ids_docs, "used": used, "skipped": skipped, "tokens": total - existing_tokens}


def build_hybrid(existing_docs: list[list[int]], supp_docs: list[list[int]], seq_len: int = SEQ_LEN,
                 n: int = NUM_SAMPLES, seed: int = SEED) -> dict:
    """既存（全量）＋補充を連結 → チャンク化 → 既存を含むチャンクを必須として n 個選ぶ。"""
    e_tokens = sum(len(d) for d in existing_docs)
    all_ids = concat_ids(existing_docs) + concat_ids(supp_docs)
    chunks = chunk_fixed(all_ids, seq_len)
    keep_first = -(-e_tokens // seq_len)  # ceil
    idx, samples = select_chunks(chunks, n, seed, keep_first=keep_first)
    # 出典内訳: チャンク i の既存トークン数 = [i*L, (i+1)*L) と [0, e_tokens) の交わり
    e_in = sum(max(0, min(e_tokens, (i + 1) * seq_len) - i * seq_len) for i in idx)
    if e_in != e_tokens:
        raise ValueError(f"既存コーパスを全量使えていない: {e_in} != {e_tokens}（重複チャンクで除外された可能性）")
    return {"samples": samples, "chunk_indices": idx, "pool_chunks": len(chunks),
            "existing_tokens": e_tokens, "existing_tokens_used": e_in,
            "supplement_tokens_used": n * seq_len - e_in, "supplement_tokens_fetched": len(all_ids) - e_tokens,
            "dropped_tail_tokens": len(all_ids) - len(chunks) * seq_len}
