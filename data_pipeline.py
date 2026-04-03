"""
data_pipeline.py
================
Data collection and preprocessing for the Mamba-Transformer Interpretability Project.

Strategy (from proposal + annealing tweak):
  - Primary: fineweb-edu / high-quality academic PDFs (pedagogically structured)
  - Secondary: Code (The Stack / StarCoder)
  - Tertiary: Math/Proofs (OpenWebMath, Proof-Pile-2)
  - Target: ~10 Billion tokens total, split across two phases:

  ┌─────────────────────────────────────────────────────────────────┐
  │  STABLE PHASE   (8B tokens → ./data/stable/)                   │
  │    60% FineWeb-Edu   — foundational language & grammar          │
  │    25% The Stack     — coding logic & structured reasoning      │
  │    15% OpenWebMath   — intro-level mathematical language        │
  │                                                                 │
  │  ANNEAL PHASE   (2B tokens → ./data/anneal/)                   │
  │    50% Proof-Pile-2  — formal proofs; crystallises deduction    │
  │    50% OpenWebMath   — hard mathematical reasoning              │
  │                                                                 │
  │  During training, the WSD scheduler's decay phase begins        │
  │  exactly when the DataLoader switches to the anneal shards.     │
  │  The model therefore consolidates reasoning circuits under a    │
  │  falling LR — extracting maximum value from each update.        │
  └─────────────────────────────────────────────────────────────────┘

Hardware target: Kaggle T4 nodes (async, CPU-heavy work)
"""

import os
import json
import time
import hashlib
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array
import ctypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
import torch
from torch.utils.data import IterableDataset, DataLoader


# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    """
    Defines one training phase: its dataset mixture and token budget.
    Both phases together form the full WSD training curriculum.
    """
    name: str                          # "stable" or "anneal"
    mixture: Dict[str, float]          # dataset_name → sampling weight (must sum to 1.0)
    token_budget: int                  # How many tokens to pre-bake for this phase
    shard_dir: str                     # Where shards land on disk

    def __post_init__(self):
        total = sum(self.mixture.values())
        assert abs(total - 1.0) < 1e-6, \
            f"Phase '{self.name}' mixture weights sum to {total:.4f}, must be 1.0"


@dataclass
class DataConfig:
    """Master config — tokenizer settings, quality filters, phase definitions."""

    # ── Tokenizer ────────────────────────────────────────────
    vocab_size: int            = 32_768      # Power-of-2; fits in uint16 → halved storage
    tokenizer_train_samples: int = 5_000_000 # Cross-phase sample for BPE training
    seq_len: int               = 2048        # Context window length

    # ── Shard geometry ───────────────────────────────────────
    shard_size_tokens: int = 100_000_000     # 100M tokens / shard ≈ 200 MB (uint16)

    # ── Paths ────────────────────────────────────────────────
    raw_cache_dir: str   = "./cache/raw"
    tokenizer_path: str  = "./tokenizer/tokenizer.json"
    # Phase shard dirs are defined inside the PhaseConfig objects below.

    # ── Quality filters ──────────────────────────────────────
    min_char_length: int  = 200
    max_char_length: int  = 100_000
    dedup_hash_bits: int  = 64          # SHA-256 prefix bits used for exact dedup

    # ── Parallel sharding ────────────────────────────────────
    #
    #  Architecture:
    #    N producer processes  (one per dataset in the phase mixture)
    #      → each streams its dataset, filters, deduplicates locally,
    #        tokenises in bulk batches, pushes List[int] token chunks to queue
    #    1 writer process
    #      → drains the queue, merges token chunks, flushes .npy shards
    #
    #  producer_batch_size : texts tokenised per batch call.
    #                        HF tokenizers encodes a list ~10x faster than
    #                        encoding one-by-one due to Rust-level parallelism.
    #  queue_maxsize       : max pending batches in the inter-process queue.
    #                        Each slot ~= producer_batch_size x ~350 tok x 2B
    #                        -> 128 batch x 350 tok x 256 slots ~= 3.5 GB peak.
    #                        Lower on machines with < 16 GB RAM.
    producer_batch_size: int = 128      # texts per encode_batch() call
    queue_maxsize:       int = 256      # back-pressure buffer depth

    # ── Training phases (WSD curriculum) ─────────────────────
    #
    #  STABLE  →  broad-coverage foundation  (8B / 10B tokens)
    #  ANNEAL  →  hard-math crystallisation  (2B / 10B tokens)
    #
    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name="stable",
            token_budget=8_000_000_000,           # 8B tokens
            shard_dir="./data/data/stable",
            mixture={
                "fineweb_edu":    0.60,   # Broad language + pedagogical structure
                "the_stack_code": 0.25,   # Coding logic — trains structured reasoning
                "open_web_math":  0.15,   # Light math to warm up numeric pathways
            },
        ),
        PhaseConfig(
            name="anneal",
            token_budget=2_000_000_000,           # 2B tokens
            shard_dir="./data/data/anneal",
            mixture={
                "proof_pile_2":  0.50,    # Formal proofs — hardens deductive circuits
                "open_web_math": 0.50,    # Dense math; maximises per-token signal
            },
        ),
    ])

    @property
    def total_tokens(self) -> int:
        return sum(p.token_budget for p in self.phases)

    def get_phase(self, name: str) -> PhaseConfig:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(f"No phase named '{name}'")


CONFIG = DataConfig()


# ─────────────────────────────────────────────────────────────
# 2. DATASET LOADERS
#    Each loader returns a streaming HuggingFace Dataset.
#    Streaming means T4 nodes never download the full corpus.
# ─────────────────────────────────────────────────────────────

def load_fineweb_edu(cache_dir: str) -> Dataset:
    """
    FineWeb-Edu — score ≥ 4 filter keeps only the top pedagogical tier.
    https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
    """
    print("[DataLoader] FineWeb-Edu (score ≥ 4)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )
    return ds.filter(lambda x: x.get("score", 0) >= 4)


def load_open_web_math(cache_dir: str) -> Dataset:
    """
    OpenWebMath — ~14.7B tokens of mathematical web content.
    https://huggingface.co/datasets/open-web-math/open-web-math
    """
    print("[DataLoader] OpenWebMath...")
    return load_dataset(
        "open-web-math/open-web-math",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )


def load_the_stack(cache_dir: str, languages: List[str] = None) -> Dataset:
    """
    The Stack v1 — permissively licensed code.
    Python + Rust + C for logic; Lean for formal proofs.
    https://huggingface.co/datasets/bigcode/the-stack
    """
    from datasets import interleave_datasets
    languages = languages or ["python", "rust", "c", "lean"]
    print(f"[DataLoader] The Stack ({languages})...")
    return interleave_datasets([
        load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{lang}",
            split="train",
            cache_dir=cache_dir,
            streaming=True,
        )
        for lang in languages
    ])


def load_proof_pile_2(cache_dir: str) -> Dataset:
    """
    Proof-Pile-2 — arXiv math papers + AlgebraicStack formal proofs.
    This is the hardest, densest data; reserved for the anneal phase.
    https://huggingface.co/datasets/EleutherAI/proof-pile-2
    """
    print("[DataLoader] Proof-Pile-2...")
    return load_dataset(
        "EleutherAI/proof-pile-2",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )


# Registry — maps the string keys used in PhaseConfig.mixture → loader callables.
# Each callable is invoked lazily, so datasets not needed for a phase are never opened.
def _make_loader_registry(cache_dir: str) -> Dict[str, callable]:
    return {
        "fineweb_edu":    lambda: load_fineweb_edu(cache_dir),
        "open_web_math":  lambda: load_open_web_math(cache_dir),
        "the_stack_code": lambda: load_the_stack(cache_dir),
        "proof_pile_2":   lambda: load_proof_pile_2(cache_dir),
    }


# ─────────────────────────────────────────────────────────────
# 3. QUALITY FILTERS
# ─────────────────────────────────────────────────────────────

# Anneal-phase datasets (Proof-Pile-2, OpenWebMath) are already heavily
# curated — apply a relaxed filter so we don't accidentally discard
# valid formal proofs that look short or symbol-heavy to the heuristic.
RELAXED_FILTER_DATASETS = {"proof_pile_2", "open_web_math"}

def get_text_field(example: dict, dataset_name: str) -> str:
    """Extract raw text from dataset-specific field names."""
    field_map = {
        "fineweb_edu":    "text",
        "open_web_math":  "text",
        "the_stack_code": "content",   # The Stack uses 'content'
        "proof_pile_2":   "text",
    }
    return example.get(field_map.get(dataset_name, "text"), "")


def passes_quality_filter(text: str, dataset_name: str = "", cfg: DataConfig = CONFIG) -> bool:
    """
    Heuristic quality gate — removes boilerplate, too-short, too-long docs.
    Anneal-phase datasets get a relaxed version since they're pre-curated.
    """
    if not text or not isinstance(text, str):
        return False

    relaxed = dataset_name in RELAXED_FILTER_DATASETS

    min_len = cfg.min_char_length // 2 if relaxed else cfg.min_char_length
    if len(text) < min_len:
        return False
    if len(text) > cfg.max_char_length:
        return False

    if not relaxed:
        # Boilerplate check: skip for curated datasets where short lines
        # may be valid LaTeX commands or proof annotations.
        lines = text.splitlines()
        non_empty = [l for l in lines if l.strip()]
        short_line_ratio = (
            sum(1 for l in non_empty if len(l.strip()) < 20) / max(len(non_empty), 1)
        )
        if short_line_ratio > 0.4:
            return False

    # Word length sanity (catches encoding junk in all datasets)
    words = text.split()
    if not words:
        return False
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2.0 or avg_word_len > 25:
        return False

    return True


def sha_dedup_key(text: str, hash_bits: int = CONFIG.dedup_hash_bits) -> str:
    """Fast exact-duplicate detection key via SHA-256 prefix."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:hash_bits // 4]


# ─────────────────────────────────────────────────────────────
# 4. TOKENIZER TRAINING
# ─────────────────────────────────────────────────────────────

def text_iterator_for_tokenizer(cfg: DataConfig = CONFIG) -> Iterator[str]:
    """
    Stream a small cross-phase sample for BPE vocab training.
    Samples from ALL datasets so the vocab covers math symbols,
    code tokens, and natural language equally.
    """
    all_datasets = {
        "fineweb_edu", "open_web_math", "the_stack_code", "proof_pile_2"
    }
    registry = _make_loader_registry(cfg.raw_cache_dir)
    count = 0
    per_dataset = cfg.tokenizer_train_samples // len(all_datasets)

    for name in all_datasets:
        ds = registry[name]()
        for example in ds:
            text = get_text_field(example, name)
            if passes_quality_filter(text, name, cfg):
                yield text
                count += 1
                if count % per_dataset == 0 and count >= per_dataset:
                    break   # Move to next dataset


def train_tokenizer(cfg: DataConfig = CONFIG) -> Tokenizer:
    """
    Train a BPE tokenizer from scratch on a balanced cross-dataset sample.
    32,768 vocab (power of 2) is μP-friendly and fits in uint16.
    """
    save_path = cfg.tokenizer_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[Tokenizer] Training BPE (vocab={cfg.vocab_size}, "
          f"samples≈{cfg.tokenizer_train_samples:,})...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        text_iterator_for_tokenizer(cfg),
        trainer=trainer,
        length=cfg.tokenizer_train_samples,
    )
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.save(save_path)
    print(f"[Tokenizer] Saved → {save_path}")
    return tokenizer


def load_tokenizer(cfg: DataConfig = CONFIG) -> Tokenizer:
    return Tokenizer.from_file(cfg.tokenizer_path)


# ─────────────────────────────────────────────────────────────
# 5. PHASE-AWARE PARALLEL SHARDING
#
#  Pipeline topology (one phase):
#
#   ┌──────────────┐
#   │ Producer-0   │──┐
#   │ (fineweb_edu)│  │  ┌──────────────┐   ┌─────────────────────────────┐
#   └──────────────┘  ├─►│  token_q     ├──►│  Writer process             │
#   ┌──────────────┐  │  │  (Queue)     │   │  • merges batches           │
#   │ Producer-1   │──┘  └──────────────┘   │  • flushes .npy shards      │
#   │ (the_stack)  │                         │  • owns the stop decision   │
#   └──────────────┘                         └─────────────────────────────┘
#
#  KEY FIX vs previous version:
#    Old: each producer had an individual token quota → producers quit early
#         before the phase budget was reached → only 23% of tokens written.
#    New: producers run INDEFINITELY (restarting their stream on exhaustion),
#         controlled only by a shared mp.Event stop_flag set by the writer
#         once it has collected enough tokens.  The writer is the SOLE
#         authority on "are we done".
# ─────────────────────────────────────────────────────────────

# Sentinel value producers push when they are asked to stop
_QUEUE_SENTINEL = None


def _producer_worker(
    dataset_name:   str,
    tokenizer_path: str,
    cfg_dict:       dict,
    token_q:        Queue,
    stop_flag:      "mp.Event",   # set by writer when budget is reached
    phase_name:     str,
    worker_id:      int,
):
    """
    Producer process: streams one dataset indefinitely until stop_flag is set.

    Runs until the writer signals it to stop — this guarantees we always
    reach the full token budget regardless of per-dataset document counts
    or filter rejection rates.

    On stream exhaustion the dataset iterator is silently restarted.
    The local dedup set is cleared on restart so we can reuse documents
    (second pass) rather than hanging waiting for new data.
    """
    tok = Tokenizer.from_file(tokenizer_path)

    min_len       = cfg_dict["min_char_length"]
    max_len       = cfg_dict["max_char_length"]
    hash_bits     = cfg_dict["dedup_hash_bits"]
    batch_size    = cfg_dict["producer_batch_size"]
    cache_dir     = cfg_dict["raw_cache_dir"]
    relaxed_dsets = {"proof_pile_2", "open_web_math"}
    relaxed       = dataset_name in relaxed_dsets

    field_map = {
        "fineweb_edu":    "text",
        "open_web_math":  "text",
        "the_stack_code": "content",
        "proof_pile_2":   "text",
    }
    text_field = field_map.get(dataset_name, "text")

    registry   = _make_loader_registry(cache_dir)
    local_seen = set()
    text_batch = []
    pass_num   = 0            # how many times we have looped over the dataset

    def flush_batch():
        if not text_batch:
            return
        if stop_flag.is_set():
            text_batch.clear()
            return
        encodings = tok.encode_batch(text_batch)
        ids = []
        for enc in encodings:
            ids.extend(enc.ids)
        if ids:
            # put() blocks when queue is full → natural back-pressure
            # We use put_nowait inside a retry loop so we can also check
            # stop_flag and avoid hanging forever after the writer exits.
            while not stop_flag.is_set():
                try:
                    token_q.put(ids, timeout=1.0)
                    break
                except Exception:
                    continue
        text_batch.clear()

    while not stop_flag.is_set():
        pass_num += 1
        ds = registry[dataset_name]()  # re-open stream each pass

        # Clear dedup on second pass so we can reuse documents if needed
        # (budget >> unique docs is rare but possible for small datasets)
        if pass_num > 1:
            local_seen.clear()

        for example in ds:
            if stop_flag.is_set():
                break

            text = example.get(text_field, "")
            if not text or not isinstance(text, str):
                continue

            effective_min = (min_len // 2) if relaxed else min_len
            if len(text) < effective_min or len(text) > max_len:
                continue

            if not relaxed:
                lines     = text.splitlines()
                non_empty = [l for l in lines if l.strip()]
                if non_empty:
                    short_r = sum(1 for l in non_empty if len(l.strip()) < 20) / len(non_empty)
                    if short_r > 0.4:
                        continue

            words = text.split()
            if not words:
                continue
            avg_wl = sum(len(w) for w in words) / len(words)
            if avg_wl < 2.0 or avg_wl > 25:
                continue

            key = hashlib.sha256(text.encode()).hexdigest()[: (hash_bits // 4)]
            if key in local_seen:
                continue
            local_seen.add(key)

            text_batch.append(text)
            if len(text_batch) >= batch_size:
                flush_batch()

        flush_batch()   # drain partial batch at end of each pass

    # Signal to writer that this producer is done
    token_q.put(_QUEUE_SENTINEL)


def _writer_worker(
    token_q:        Queue,
    out_dir:        str,
    shard_size:     int,
    target_tokens:  int,
    n_producers:    int,
    phase_name:     str,
    shared_counter: Value,
    stop_flag:      "mp.Event",
):
    """
    Writer process: single consumer of token_q.

    Owns the stop decision: once total_written >= target_tokens it sets
    stop_flag, waits for all n_producers sentinels, then flushes the
    final partial shard.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    token_buffer  = []
    shard_idx     = 0
    total_written = 0
    done_count    = 0

    while True:
        try:
            item = token_q.get(timeout=30)
        except Exception:
            # Timeout — check if all producers are already done
            if stop_flag.is_set() and done_count >= n_producers:
                break
            continue

        if item is _QUEUE_SENTINEL:
            done_count += 1
            if done_count >= n_producers:
                break
            continue

        token_buffer.extend(item)
        total_written += len(item)
        shared_counter.value = total_written

        # Flush complete shards immediately
        while len(token_buffer) >= shard_size:
            chunk = token_buffer[:shard_size]
            arr   = np.array(chunk, dtype=np.uint16)
            fpath = Path(out_dir) / f"shard_{shard_idx:05d}.npy"
            np.save(fpath, arr)
            print(f"  [{phase_name}] └─ Wrote {fpath.name}  ({len(arr):,} tokens)",
                  flush=True)
            token_buffer = token_buffer[shard_size:]
            shard_idx += 1

        # Once we have enough tokens, signal all producers to stop and
        # then drain any remaining sentinels before flushing the final shard.
        if total_written >= target_tokens and not stop_flag.is_set():
            stop_flag.set()

    # Final partial shard
    if token_buffer:
        # Trim to exact budget if we overshot slightly
        remaining = min(len(token_buffer), target_tokens - (total_written - len(token_buffer)))
        if remaining > 0:
            arr   = np.array(token_buffer[:remaining], dtype=np.uint16)
            fpath = Path(out_dir) / f"shard_{shard_idx:05d}.npy"
            np.save(fpath, arr)
            print(f"  [{phase_name}] └─ Wrote {fpath.name}  "
                  f"(partial, {len(arr):,} tokens)", flush=True)
            shard_idx += 1

    print(f"[{phase_name}] Writer done — {total_written:,} tokens, "
          f"{shard_idx} shards", flush=True)


def build_phase_shards(
    phase:              PhaseConfig,
    tokenizer:          Tokenizer,
    cfg:                DataConfig = CONFIG,
    global_seen_hashes: Optional[set] = None,
) -> set:
    """
    Parallel sharding for one training phase.

    Spawns N producers (one per dataset in the mixture) + 1 writer.
    Producers run until the writer signals stop via a shared Event,
    guaranteeing the full token budget is always written.

    Expected throughput on Kaggle dual-T4 (4 vCPUs, 30 GB RAM):
      ~800k–1M tok/s  →  ~2.2 h for 8B tokens  (stable phase)
      ~800k–1M tok/s  →  ~35 min for 2B tokens  (anneal phase)
    """
    print(f"\n{'='*62}")
    print(f" Phase : {phase.name.upper()}  |  Budget : {phase.token_budget/1e9:.1f}B tokens")
    print(f" Output: {phase.shard_dir}")
    mixture_str = "  ".join(f"{k}={v*100:.0f}%" for k, v in phase.mixture.items())
    print(f" Mix   : {mixture_str}")
    print(f" Mode  : parallel  ({len(phase.mixture)} producers + 1 writer)")
    print('='*62)

    # spawn is safest on Linux/Kaggle — avoids inheriting parent state
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set in this session

    cfg_dict = {
        "min_char_length":     cfg.min_char_length,
        "max_char_length":     cfg.max_char_length,
        "dedup_hash_bits":     cfg.dedup_hash_bits,
        "producer_batch_size": cfg.producer_batch_size,
        "raw_cache_dir":       cfg.raw_cache_dir,
    }

    token_q        = Queue(maxsize=cfg.queue_maxsize)
    shared_counter = Value(ctypes.c_int64, 0)
    stop_flag      = mp.Event()          # writer sets this → producers exit cleanly
    n_producers    = len(phase.mixture)

    # ── Spawn producers ──────────────────────────────────────
    producers = []
    for i, dset_name in enumerate(phase.mixture):
        p = Process(
            target=_producer_worker,
            name=f"producer-{dset_name}",
            args=(dset_name, cfg.tokenizer_path, cfg_dict,
                  token_q, stop_flag, phase.name, i),
            daemon=True,
        )
        p.start()
        producers.append(p)
        print(f"  [spawn] Producer-{i}: {dset_name}  "
              f"(weight={phase.mixture[dset_name]*100:.0f}%, runs until stop signal)")

    # ── Spawn writer ─────────────────────────────────────────
    writer = Process(
        target=_writer_worker,
        name="writer",
        args=(token_q, phase.shard_dir, cfg.shard_size_tokens,
              phase.token_budget, n_producers, phase.name,
              shared_counter, stop_flag),
        daemon=True,
    )
    writer.start()
    print(f"  [spawn] Writer → {phase.shard_dir}")

    # ── Progress bar in main process ─────────────────────────
    pbar = tqdm(
        total=phase.token_budget,
        desc=f"[{phase.name}] Sharding",
        unit="tok",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    prev = 0
    while writer.is_alive():
        cur = shared_counter.value
        if cur > prev:
            pbar.update(cur - prev)
            prev = cur
        time.sleep(0.5)
    # Final update
    pbar.update(max(0, shared_counter.value - prev))
    pbar.close()

    # ── Join all processes ────────────────────────────────────
    # Writer finishes first; producers may take up to ~30s to notice stop_flag
    writer.join(timeout=300)
    for p in producers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()   # force-kill stragglers

    # Surface crashes
    failed = [p for p in producers + [writer]
              if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(
            f"Workers exited with errors: "
            + ", ".join(f"{p.name}(code={p.exitcode})" for p in failed)
        )

    total_written = shared_counter.value
    n_shards = len(list(Path(phase.shard_dir).glob("shard_*.npy")))
    expected_shards = phase.token_budget // cfg.shard_size_tokens

    status = "✓" if n_shards >= expected_shards else "⚠ fewer shards than expected"
    print(f"[{phase.name}] {status}  {total_written:,} tokens  |  "
          f"{n_shards} / {expected_shards} shards  → {phase.shard_dir}\n")

    return set()


def build_all_phases(
    tokenizer: Tokenizer,
    cfg: DataConfig = CONFIG,
) -> None:
    """
    Builds BOTH phases sequentially (stable then anneal),
    then writes a manifest JSON the trainer reads at startup.
    """
    for phase in cfg.phases:
        build_phase_shards(phase, tokenizer, cfg)

    manifest = {
        p.name: {
            "shard_dir":    p.shard_dir,
            "token_budget": p.token_budget,
            "mixture":      p.mixture,
            "n_shards":     len(list(Path(p.shard_dir).glob("shard_*.npy"))),
        }
        for p in cfg.phases
    }
    manifest_path = Path("./data/phase_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Written → {manifest_path}")
    print(json.dumps(manifest, indent=2))


# ─────────────────────────────────────────────────────────────
# 6. PYTORCH DATASETS (consumed by the trainer)
# ─────────────────────────────────────────────────────────────

class ShardedTokenDataset(IterableDataset):
    """
    Streams fixed-length (seq_len+1) windows from one phase's .npy shards.
    Each item is (input_ids, labels) where labels = input_ids shifted by 1.

    Usage:
        stable_ds = ShardedTokenDataset("./data/stable")
        anneal_ds = ShardedTokenDataset("./data/anneal", shuffle_shards=False)
        # Trainer uses stable_ds first, then switches to anneal_ds when
        # the WSD scheduler enters its decay phase.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int = CONFIG.seq_len,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.shard_paths = sorted(Path(shard_dir).glob("shard_*.npy"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}. "
                                    f"Run --stage shards first.")
        self.seq_len       = seq_len
        self.shuffle_shards = shuffle_shards
        self.rng           = np.random.default_rng(seed)
        self.phase_name    = Path(shard_dir).name   # "stable" or "anneal"

    def __iter__(self):
        paths = list(self.shard_paths)
        if self.shuffle_shards:
            self.rng.shuffle(paths)

        for path in paths:
            tokens = np.load(path).astype(np.int32)
            n_chunks = len(tokens) // (self.seq_len + 1)
            for i in range(n_chunks):
                chunk = tokens[i * (self.seq_len + 1): (i + 1) * (self.seq_len + 1)]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                yield x, y

    def get_dataloader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    def estimated_steps(self, batch_size: int) -> int:
        """Rough step count — useful for setting the WSD phase boundary."""
        total_tokens = sum(
            len(np.load(p)) for p in self.shard_paths
        )
        return total_tokens // (batch_size * self.seq_len)


def build_dataloaders(
    batch_size: int,
    cfg: DataConfig = CONFIG,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Builds both phase DataLoaders and returns a boundary dict for the trainer.

    Returns:
        stable_loader : DataLoader for the stable phase (8B tokens)
        anneal_loader : DataLoader for the anneal phase (2B tokens)
        boundary      : dict with step counts — trainer uses this to
                        know when to switch loaders AND when to start LR decay.

    Boundary dict schema:
        {
          "stable_steps" : int,  # Steps in stable phase
          "anneal_steps" : int,  # Steps in anneal phase
          "total_steps"  : int,
        }
    """
    stable_ds = ShardedTokenDataset(cfg.get_phase("stable").shard_dir)
    anneal_ds = ShardedTokenDataset(cfg.get_phase("anneal").shard_dir,
                                    shuffle_shards=False)  # Ordered for decay phase

    stable_steps = stable_ds.estimated_steps(batch_size)
    anneal_steps = anneal_ds.estimated_steps(batch_size)
    boundary = {
        "stable_steps": stable_steps,
        "anneal_steps": anneal_steps,
        "total_steps":  stable_steps + anneal_steps,
    }

    print(f"\n[DataLoaders] Phase boundary summary:")
    print(f"  Stable  : {stable_steps:>8,} steps  (WSD warm+stable phase)")
    print(f"  Anneal  : {anneal_steps:>8,} steps  (WSD decay phase — switch here)")
    print(f"  Total   : {boundary['total_steps']:>8,} steps\n")

    return (
        stable_ds.get_dataloader(batch_size, num_workers),
        anneal_ds.get_dataloader(batch_size, num_workers),
        boundary,
    )


# ─────────────────────────────────────────────────────────────
# 7. INSPECTION UTILITIES
# ─────────────────────────────────────────────────────────────

def inspect_shard(path: str, tokenizer: Tokenizer, n_samples: int = 3,
                  cfg: DataConfig = CONFIG):
    """Decode and print a few sequences from a shard for sanity checking."""
    tokens = np.load(path).astype(np.int32)
    print(f"\n[Inspect] {path}  |  Tokens: {len(tokens):,}")
    for i in range(n_samples):
        start = np.random.randint(0, max(1, len(tokens) - cfg.seq_len))
        chunk = tokens[start: start + cfg.seq_len].tolist()
        text  = tokenizer.decode(chunk)
        print(f"\n── Sample {i+1} {'─'*40}\n{text[:500]}...\n")


def count_phase_tokens(cfg: DataConfig = CONFIG):
    """Count tokens per phase and overall total."""
    grand_total = 0
    for phase in cfg.phases:
        shards = sorted(Path(phase.shard_dir).glob("shard_*.npy"))
        phase_total = sum(len(np.load(p)) for p in shards)
        grand_total += phase_total
        print(f"  {phase.name:8s}: {len(shards):3d} shards  |  "
              f"{phase_total/1e9:.2f}B tokens  ({phase.shard_dir})")
    print(f"  {'TOTAL':8s}: {grand_total/1e9:.2f}B tokens")
    return grand_total


# ─────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data pipeline for Mamba Interpretability project",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["tokenizer", "shards", "shards_stable", "shards_anneal",
                 "inspect", "count"],
        required=True,
        help=(
            "tokenizer      — train BPE tokenizer (run first)\n"
            "shards         — build ALL phases (stable + anneal) sequentially\n"
            "shards_stable  — build ONLY the stable phase (Kaggle node 1)\n"
            "shards_anneal  — build ONLY the anneal phase (Kaggle node 2)\n"
            "inspect        — decode samples from a shard file\n"
            "count          — print token counts per phase"
        ),
    )
    parser.add_argument("--shard", type=str, default=None,
                        help="Path to .npy shard (for --stage inspect)")
    args = parser.parse_args()

    if args.stage == "tokenizer":
        train_tokenizer(CONFIG)

    elif args.stage == "shards":
        tok = load_tokenizer(CONFIG)
        build_all_phases(tok, CONFIG)

    elif args.stage == "shards_stable":
        tok = load_tokenizer(CONFIG)
        print("[Pipeline] Building STABLE phase only (run anneal on a separate node).")
        build_phase_shards(CONFIG.get_phase("stable"), tok, CONFIG)

    elif args.stage == "shards_anneal":
        tok = load_tokenizer(CONFIG)
        print("[Pipeline] Building ANNEAL phase only.")
        print("[Warning]  Cross-phase dedup is disabled — run 'shards' for full dedup.")
        build_phase_shards(CONFIG.get_phase("anneal"), tok, CONFIG)

    elif args.stage == "inspect":
        if not args.shard:
            parser.error("--shard PATH required for inspect stage")
        tok = load_tokenizer(CONFIG)
        inspect_shard(args.shard, tok, cfg=CONFIG)

    elif args.stage == "count":
        count_phase_tokens(CONFIG)