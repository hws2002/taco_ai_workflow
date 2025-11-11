"""
Step 4: Extract keywords per conversation
- Input: s1 json (default: test/output_full/s1_ai_responses.json)
- Output: test/output_full/s2_keywords_pipeline_test.json
- For each conversation:
  - per response: extract at least 3 keywords (keywords_per_response >= 3)
  - aggregate across responses, dedupe by max score
  - pick final top 5 per conversation
- Uses KeyBERT with SBERT (default: paraphrase-multilingual-mpnet-base-v2)
- Multilingual vectorizer token pattern for CJK/Hangul/Latin
- Optional Chinese segmentation via jieba when CJK detected
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import jieba
    JIEBA_AVAILABLE = True
except Exception:
    JIEBA_AVAILABLE = False


def clean_minimal(text: str) -> str:
    # URLs and markdown cleanup minimal (preprocessed already handled heavy cleaning)
    import re
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def group_by_conversation(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['conversation_id']].append(r)
    return grouped


def extract_for_conv(conv_id: int, rows: List[Dict[str, Any]], kw_model: KeyBERT, tokenizer: AutoTokenizer, kpr: int, top_n: int) -> Dict[str, Any]:
    token_pattern = r"(?u)(?:[\u4E00-\u9FFF]{1,}|[\u3131-\u318E\uAC00-\uD7A3]{2,}|[A-Za-z0-9_]{2,})"
    vectorizer = CountVectorizer(token_pattern=token_pattern, ngram_range=(1, 3))

    # 1) Combine all responses' cleaned text
    combined_parts: List[str] = []
    for r in rows:
        t = clean_minimal(r.get('content', ''))
        if t:
            combined_parts.append(t)
    combined_text = ' '.join(combined_parts).strip()

    all_keywords = []
    if not combined_text:
        # Return empty keywords if nothing to process
        return {
            'conversation_id': conv_id,
            'conversation_title': rows[0].get('conversation_title', ''),
            'response_count': len(rows),
            'keywords': []
        }

    # 2) Tokenize and chunk into 512-token windows
    try:
        enc = tokenizer(combined_text, add_special_tokens=False, return_attention_mask=False, return_offsets_mapping=False)
        ids = enc['input_ids'] if isinstance(enc, dict) else enc.input_ids
        chunks_ids = [ids[i:i+512] for i in range(0, len(ids), 512)]
        chunk_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks_ids]
    except Exception:
        # Fallback: split by characters approx every 3000 chars
        chunk_size_chars = 3000
        chunk_texts = [combined_text[i:i+chunk_size_chars] for i in range(0, len(combined_text), chunk_size_chars)]

    # 3) Extract keywords per chunk
    for chunk in tqdm(chunk_texts, desc=f"conv {conv_id} (chunks)", leave=False):
        text = chunk
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in text)
        if has_cjk and JIEBA_AVAILABLE:
            text = ' '.join(jieba.cut(text))
        try:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words=None,
                use_mmr=True,
                diversity=0.5,
                nr_candidates=30,
                top_n=max(3, kpr),
                vectorizer=vectorizer
            )
            for kw, score in kws:
                all_keywords.append((kw, float(score)))
        except Exception:
            continue

    # dedupe by max score
    best: Dict[str, float] = {}
    for kw, sc in all_keywords:
        if kw not in best or sc > best[kw]:
            best[kw] = sc

    top = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_n]
    final = [{'keyword': k, 'score': v} for k, v in top]

    return {
        'conversation_id': conv_id,
        'conversation_title': rows[0].get('conversation_title', ''),
        'response_count': len(rows),
        'keywords': final
    }


def run(input_path: Path, output_path: Path, model_name: str, cache_dir: str, kpr: int, top_n: int):
    with input_path.open('r', encoding='utf-8') as f:
        rows = json.load(f)
    if not rows:
        print(f"No rows in {input_path}")
        return

    grouped = group_by_conversation(rows)

    # KeyBERT init
    if cache_dir:
        st_model = SentenceTransformer(model_name, cache_folder=cache_dir)
        kw_model = KeyBERT(model=st_model)
    else:
        kw_model = KeyBERT(model=model_name)

    # Tokenizer for 512-token chunking
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir) if cache_dir else AutoTokenizer.from_pretrained(model_name)

    results: List[Dict[str, Any]] = []
    items_list = list(grouped.items())
    for conv_id, items in tqdm(items_list, desc="키워드 추출 (conversations)"):
        results.append(extract_for_conv(conv_id, items, kw_model, tokenizer, kpr, top_n))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description='Extract keywords per conversation')
    p.add_argument('--input', type=str, default='test/output_full/s1_ai_responses.json')
    p.add_argument('--output', type=str, default='test/output_full/s2_keywords_pipeline_test.json')
    p.add_argument('--model', type=str, default='paraphrase-multilingual-mpnet-base-v2')
    p.add_argument('--cache-dir', type=str, default='models_cache')
    p.add_argument('--keywords-per-response', type=int, default=3)
    p.add_argument('--top-n', type=int, default=5)
    args = p.parse_args()

    run(Path(args.input), Path(args.output), args.model, args.cache_dir, args.keywords_per_response, args.top_n)


if __name__ == '__main__':
    main()