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


def run(input_path: Path, out_pkl: Path, model_name: str, cache_dir: str):
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

        texts = [r.get('content', '') for r in to_encode]
        emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False)
        for r, vec in zip(to_encode, emb):
            cache[r['response_id']] = {
                'conversation_id': r.get('conversation_id'),
                'response_id': r.get('response_id'),
                'conversation_title': r.get('conversation_title', ''),
                'timestamp': r.get('timestamp'),
                'embedding': np.asarray(vec, dtype=np.float32),
            }
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
    args = p.parse_args()

    run(Path(args.input), Path(args.out_pkl), args.model, args.cache_dir)


if __name__ == '__main__':
    main()
