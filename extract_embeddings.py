import json
import time
from typing import List, Dict, Any
from pathlib import Path

from sentence_transformers import SentenceTransformer
import argparse
import pickle
import re


def load_ai_responses(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_embeddings_pkl(payload: Dict[str, Any], output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)


def extract_embeddings(
    responses: List[Dict[str, Any]],
    model_name: str = "thenlper/gte-base",
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    content_max_chars: int = None,
    cache_dir: str = None,
) -> List[Dict[str, Any]]:
    model = SentenceTransformer(model_name, cache_folder=cache_dir)

    texts = []
    meta = []
    for r in responses:
        content = r.get("content", "")
        if content_max_chars is not None:
            content = content[:content_max_chars]
        texts.append(content)
        meta.append({
            "response_id": r.get("response_id"),
            "conversation_title": r.get("conversation_title"),
        })

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    elapsed = time.time() - t0

    items = []
    for m, emb in zip(meta, embeddings):
        items.append({
            "response_id": m["response_id"],
            "conversation_title": m["conversation_title"],
            "embedding": emb,  # numpy array 그대로 저장 (pickle)
        })

    print(f"임베딩 총 소요 시간: {elapsed:.2f}초, 문서당 평균: {elapsed/len(texts):.3f}초")
    return items


def main():
    project_root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Extract embeddings with thenlper/gte-base and save as .pkl")
    parser.add_argument("--input", type=str, default=str(project_root / 'test' / 'output' / 's1_ai_responses.json'), help="입력 responses JSON 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 PKL 경로 (미지정 시 모델명이 포함된 기본 경로 사용)")
    parser.add_argument("--batch-size", type=int, default=32, help="인코딩 배치 크기")
    parser.add_argument("--no-normalize", action="store_true", help="임베딩 정규화 비활성화")
    parser.add_argument("--content-max-chars", type=int, default=None, help="본문 최대 길이 (문자)")
    parser.add_argument("--model", type=str, default="thenlper/gte-base", help="SentenceTransformer 모델명")
    parser.add_argument("--cache-dir", type=str, default=str(project_root / 'models_cache'), help="SentenceTransformer 캐시 디렉터리")
    args = parser.parse_args()

    input_path = Path(args.input)
    # 모델명을 파일명에 안전하게 넣기 위한 슬러그
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", args.model).strip('-')
    if args.output is None:
        output_path = project_root / 'test' / 'output' / f's2_ai_responses_embeddings_{model_slug}.pkl'
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("AI 응답 데이터 로딩 중...")
    responses = load_ai_responses(str(input_path))
    print(f"총 {len(responses)}개의 응답을 로드했습니다.\n")

    # 증분 캐시 로딩
    existing_payload = None
    if output_path.exists():
        try:
            with open(output_path, 'rb') as f:
                existing_payload = pickle.load(f)
        except Exception:
            existing_payload = None

    existing_items = {}
    if existing_payload and existing_payload.get('model') == args.model and existing_payload.get('normalized') == (not args.no_normalize):
        for it in existing_payload.get('items', []):
            existing_items[it['response_id']] = it

    # 새로 계산해야 할 응답만 필터링
    missing_responses = [r for r in responses if r.get('response_id') not in existing_items]
    print(f"캐시 상태: 기존 {len(existing_items)}개, 신규 {len(missing_responses)}개")

    items_all = list(existing_items.values())
    if missing_responses:
        print(f"임베딩 추출 중... (model={args.model})")
        t0 = time.time()
        print(f"모델 캐시 디렉터리: {args.cache_dir}")
        new_items = extract_embeddings(
            missing_responses,
            model_name=args.model,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
            content_max_chars=args.content_max_chars,
            cache_dir=args.cache_dir,
        )
        total_elapsed = time.time() - t0
        print(f"임베딩 추출 완료. 총 소요 시간: {total_elapsed:.2f}초\n")
        items_all.extend(new_items)
    else:
        print("추출할 신규 응답이 없습니다. 기존 캐시를 재사용합니다.")

    # response_id로 정렬하여 저장 일관성 유지
    items_all.sort(key=lambda x: (x['response_id'] is None, x['response_id']))

    payload = {
        "model": args.model,
        "normalized": not args.no_normalize,
        "count": len(items_all),
        "items": items_all,
    }

    print(f"결과를 {output_path}에 저장 중... (.pkl)")
    save_embeddings_pkl(payload, str(output_path))
    print("저장 완료.")


if __name__ == "__main__":
    main()
