"""
Chunked conversation-level keyword extraction
- Combine all responses per conversation
- Chunk by 512 tokens using the model tokenizer
- Extract keywords per chunk with KeyBERT
- Aggregate across chunks, dedupe by max score
- Keep top N per conversation (default 5)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    import jieba
    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False


def clean_minimal(text: str) -> str:
    import re
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def group_by_conversation(rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r['conversation_id']].append(r)
    return grouped


def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 512) -> List[str]:
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )
        ids = enc['input_ids'] if isinstance(enc, dict) else enc.input_ids
        if not ids:
            return []
        chunks_ids = [ids[i:i + max_tokens] for i in range(0, len(ids), max_tokens)]
        return [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    except Exception:
        # Fallback by chars roughly
        size = 3000
        return [text[i:i + size] for i in range(0, len(text), size)]


def extract_for_conv(
    conv_id: Any,
    rows: List[Dict[str, Any]],
    kw_model: KeyBERT,
    tokenizer,
    per_chunk_topn: int,
    nr_candidates: int,
    diversity: float,
    ngram_max: int,
    final_topn: int,
) -> Dict[str, Any]:
    # Combine
    combined_parts: List[str] = []
    for r in rows:
        t = clean_minimal(r.get('content', ''))
        if t:
            combined_parts.append(t)
    combined_text = ' '.join(combined_parts).strip()

    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"
    vectorizer = CountVectorizer(token_pattern=token_pattern, ngram_range=(1, ngram_max))

    all_keywords: List[Tuple[str, float]] = []
    if combined_text:
        chunks = chunk_by_tokens(combined_text, tokenizer, 512)
        for chunk in tqdm(chunks, desc=f"conv {conv_id} (chunks)", leave=False):
            text = chunk
            has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
            if has_cjk and JIEBA_AVAILABLE:
                text = ' '.join(jieba.cut(text))
            try:
                kws = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, ngram_max),
                    stop_words=None,
                    use_mmr=True,
                    diversity=diversity,
                    nr_candidates=nr_candidates,
                    top_n=per_chunk_topn,
                    vectorizer=vectorizer,
                )
                for kw, score in kws:
                    all_keywords.append((kw, float(score)))
            except Exception:
                continue

    # dedupe by max score and take final top N
    best: Dict[str, float] = {}
    for kw, sc in all_keywords:
        if kw not in best or sc > best[kw]:
            best[kw] = sc
    top = sorted(best.items(), key=lambda x: x[1], reverse=True)[:final_topn]
    final = [{'keyword': k, 'score': v} for k, v in top]

    return {
        'conversation_id': conv_id,
        'conversation_title': rows[0].get('conversation_title', '') if rows else '',
        'response_count': len(rows),
        'chunks': len(all_keywords) > 0,
        'keywords': final,
    }


essage = """Chunked keyword extraction
- Input must be s1_ai_responses.json like rows with fields: response_id, conversation_id, content, conversation_title
- Output: list of conversations with top keywords
"""

def run(
    input_path: Path,
    output_path: Path,
    model_name: str,
    cache_dir: str,
    per_chunk_topn: int,
    nr_candidates: int,
    diversity: float,
    ngram_max: int,
    final_topn: int,
):
    with input_path.open('r', encoding='utf-8') as f:
        rows = json.load(f)
    if not rows:
        print(f"No rows in {input_path}")
        return

    grouped = group_by_conversation(rows)

    # KeyBERT + tokenizer (use SentenceTransformer's tokenizer to avoid HF repo id mismatch)
    if cache_dir:
        st_model = SentenceTransformer(model_name, cache_folder=cache_dir)
    else:
        st_model = SentenceTransformer(model_name)
    kw_model = KeyBERT(model=st_model)
    tokenizer = st_model.tokenizer

    results: List[Dict[str, Any]] = []
    items = list(grouped.items())
    for conv_id, items_rows in tqdm(items, desc="키워드 추출 (conversations, chunked)"):
        results.append(
            extract_for_conv(
                conv_id,
                items_rows,
                kw_model,
                tokenizer,
                per_chunk_topn,
                nr_candidates,
                diversity,
                ngram_max,
                final_topn,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Extract keywords per conversation with 512-token chunking')
    p.add_argument('--input', type=str, default='test/output_full/s1_ai_responses.json')
    p.add_argument('--output', type=str, default='test/output_full/s2_keywords_pipeline_chunked.json')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--per-chunk-topn', type=int, default=10, help='top_n per chunk before aggregation')
    p.add_argument('--nr-candidates', type=int, default=30)
    p.add_argument('--diversity', type=float, default=0.5)
    p.add_argument('--ngram-max', type=int, default=3)
    p.add_argument('--final-topn', type=int, default=5, help='final top N per conversation after aggregation')
    args = p.parse_args()

    run(
        Path(args.input),
        Path(args.output),
        args.model,
        args.cache_dir,
        args.per_chunk_topn,
        args.nr_candidates,
        args.diversity,
        args.ngram_max,
        args.final_topn,
    )


if __name__ == '__main__':
    main()
