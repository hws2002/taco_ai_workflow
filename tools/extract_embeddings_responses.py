"""
Step 3: Extract SBERT embeddings per response (incremental cache)
- Input: s1 responses json (default: test/output_full/s1_ai_responses.json)
- Output cache PKL: test/output_full/response_embeddings.pkl
- Model: paraphrase-multilingual-mpnet-base-v2 by default (override via --model)
- Cache dir: models_cache by default (override via --cache-dir)
Cache format: dict[response_id] = {
  'conversation_id': ..., 'response_id': ..., 'embedding': np.ndarray,
  'conversation_title': ..., 'timestamp': ...
}
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os


def load_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_cache(path: Path):
    if not path.exists():
        return {}
    with path.open('rb') as f:
        return pickle.load(f)


def save_cache(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f)


def run(
    input_path: Path,
    out_pkl: Path,
    model_name: str,
    cache_dir: str,
    batch_size: int = 32,
    device: str = "auto",
    fp16: bool = False,
    normalize_embeddings: bool = False,
    chunk_size: int = 4096,
    save_every: int = 2000,
    num_workers: int = 0,
    multi_process: bool = False,
    mp_devices: str = "auto",
    long_strategy: str = "truncate",
):
    rows = load_rows(input_path)
    # basic schema check
    if not rows:
        print(f"No rows in {input_path}")
        return
    required = {"response_id", "conversation_id", "content"}
    missing = required - set(rows[0].keys())
    if missing:
        print(f"Warning: input schema missing keys {missing}. Proceeding with best-effort.")
    cache: Dict[str, Dict[str, Any]] = load_cache(out_pkl)

    # build list of new items to encode
    to_encode = [r for r in rows if r['response_id'] not in cache]

    print(f"Total rows: {len(rows)} | New to encode: {len(to_encode)} | Cached: {len(cache)}")
    if to_encode:
        if cache_dir:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_name)

        if not multi_process:
            if device == "auto":
                use_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                use_device = device
            try:
                model.to(use_device)
            except Exception as e:
                print(f"Warning: failed to move model to {use_device}: {e}. Using default device.")
                use_device = str(model._target_device) if hasattr(model, "_target_device") else "cpu"

            if fp16 and use_device.startswith("cuda"):
                try:
                    model.half()
                except Exception as e:
                    print(f"Warning: failed to switch model to fp16: {e}")

        try:
            max_len = getattr(model, 'max_seq_length', None)
            print(f"Model max_seq_length: {max_len}")
        except Exception:
            max_len = None
        # crude estimate using tokenizer if available
        over_count = 0
        try:
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is not None:
                sample_texts = [r.get('content', '') for r in to_encode[:2000]]
                for t in sample_texts:
                    if not t:
                        continue
                    ids = tokenizer.encode(t, add_special_tokens=True, truncation=False)
                    if max_len and len(ids) > max_len:
                        over_count += 1
            print(f"Estimated long responses > max_seq_length among first {min(2000, len(to_encode))}: {over_count}")
        except Exception as e:
            print(f"Note: could not estimate long responses: {e}")

        def iter_chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i + n]

        encoded_count = 0
        # If chunk-mean is requested, force single-process to simplify per-text handling
        if long_strategy == "chunk-mean" and multi_process:
            print("Warning: --long-strategy chunk-mean is not compatible with --multi-process; falling back to single-process.")
            multi_process = False

        # helper: encode one text with optional chunk-mean
        def encode_single_text(text: str):
            if long_strategy != "chunk-mean":
                vec = model.encode(
                    [text],
                    batch_size=max(1, batch_size),
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    num_workers=max(0, num_workers),
                )[0]
                return np.asarray(vec, dtype=np.float32)
            # chunk-mean path
            if tokenizer is None or not max_len:
                # fallback to truncate behavior if we cannot tokenize
                vec = model.encode(
                    [text],
                    batch_size=max(1, batch_size),
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    num_workers=max(0, num_workers),
                )[0]
                return np.asarray(vec, dtype=np.float32)
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(ids) <= max_len:
                vec = model.encode(
                    [text],
                    batch_size=max(1, batch_size),
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    num_workers=max(0, num_workers),
                )[0]
                return np.asarray(vec, dtype=np.float32)
            chunk_texts = []
            for i in range(0, len(ids), max_len):
                chunk_ids = ids[i:i+max_len]
                chunk_texts.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
            chunk_vecs = model.encode(
                chunk_texts,
                batch_size=max(1, batch_size),
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                num_workers=max(0, num_workers),
            )
            chunk_vecs = np.asarray(chunk_vecs, dtype=np.float32)
            # if not normalized, average then return; if normalized, average normals (equivalent to mean of unit vecs)
            mean_vec = chunk_vecs.mean(axis=0)
            # optional re-normalize to unit for cosine stability if normalize_embeddings is True
            if normalize_embeddings:
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    mean_vec = mean_vec / norm
            return mean_vec.astype(np.float32)

        if multi_process:
            if mp_devices == "auto":
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                else:
                    cpu_workers = max(1, (os.cpu_count() or 2) - 1)
                    target_devices = ["cpu"] * cpu_workers
            elif mp_devices.strip().lower() == "cpu":
                cpu_workers = max(1, (os.cpu_count() or 2) - 1)
                target_devices = ["cpu"] * cpu_workers
            elif mp_devices.strip().lower() == "cuda":
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                else:
                    print("Warning: --mp-devices cuda requested but no CUDA devices found. Falling back to CPU.")
                    cpu_workers = max(1, (os.cpu_count() or 2) - 1)
                    target_devices = ["cpu"] * cpu_workers
            else:
                target_devices = [d.strip() for d in mp_devices.split(',') if d.strip()]

            print(f"Starting multi-process pool on devices: {target_devices}")
            pool = model.start_multi_process_pool(target_devices=target_devices)
            try:
                for chunk in iter_chunks(to_encode, max(1, chunk_size)):
                    texts = [r.get('content', '') for r in chunk]
                    emb = model.encode_multi_process(
                        texts,
                        pool,
                        batch_size=max(1, batch_size),
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=True,
                    )
                    for r, vec in zip(chunk, emb):
                        cache[r['response_id']] = {
                            'conversation_id': r.get('conversation_id'),
                            'response_id': r.get('response_id'),
                            'conversation_title': r.get('conversation_title', ''),
                            'timestamp': r.get('timestamp'),
                            'embedding': np.asarray(vec, dtype=np.float32),
                        }
                    encoded_count += len(chunk)
                    if save_every and (encoded_count % save_every == 0):
                        save_cache(out_pkl, cache)
                        print(f"Progress: saved {encoded_count}/{len(to_encode)} to cache")
            finally:
                SentenceTransformer.stop_multi_process_pool(pool)
        else:
            for chunk in iter_chunks(to_encode, max(1, chunk_size)):
                for r in chunk:
                    text = r.get('content', '')
                    vec = encode_single_text(text)
                    cache[r['response_id']] = {
                        'conversation_id': r.get('conversation_id'),
                        'response_id': r.get('response_id'),
                        'conversation_title': r.get('conversation_title', ''),
                        'timestamp': r.get('timestamp'),
                        'embedding': vec,
                    }
                encoded_count += len(chunk)
                if save_every and (encoded_count % save_every == 0):
                    save_cache(out_pkl, cache)
                    print(f"Progress: saved {encoded_count}/{len(to_encode)} to cache")

        save_cache(out_pkl, cache)
        print(f"Encoded and cached: {len(to_encode)} embeddings")
    else:
        print("No new responses to encode. Cache up-to-date.")

    print(f"Embeddings cache size: {len(cache)}")


def main():
    p = argparse.ArgumentParser(description='Extract SBERT embeddings (incremental)')
    p.add_argument('--input', type=str, default='test/output_full/s1_ai_responses.json')
    p.add_argument('--out-pkl', type=str, default='test/output_full/response_embeddings.pkl')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', type=str, default='auto', help="cpu, cuda, or auto")
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--chunk-size', type=int, default=4096)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--num-workers', type=int, default=0, help='DataLoader workers for tokenization')
    p.add_argument('--multi-process', action='store_true', help='Use sentence-transformers multiprocess pool')
    p.add_argument('--mp-devices', type=str, default='auto', help='Comma-separated devices for pool or auto')
    p.add_argument('--long-strategy', type=str, default='truncate', choices=['truncate', 'chunk-mean'], help='Handle long responses by truncation or chunk-averaged embeddings')
    args = p.parse_args()

    run(
        Path(args.input),
        Path(args.out_pkl),
        args.model,
        args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        normalize_embeddings=args.normalize,
        chunk_size=args.chunk_size,
        save_every=args.save_every,
        num_workers=args.num_workers,
        multi_process=args.multi_process,
        mp_devices=args.mp_devices,
    )


if __name__ == '__main__':
    main()
