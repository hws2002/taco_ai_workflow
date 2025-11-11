import json
import time
from typing import List, Dict, Any
from pathlib import Path

from sentence_transformers import SentenceTransformer
import argparse
import pickle
import re
import numpy as np
from tqdm import tqdm


def load_ai_responses(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_embeddings_pkl(payload: Dict[str, Any], output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)


def group_by_conversation(responses: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    응답을 conversation_id별로 그룹화

    Args:
        responses: 응답 리스트

    Returns:
        conversation_id를 키로 하는 응답 그룹
    """
    grouped = {}
    for resp in responses:
        conv_id = resp.get('conversation_id')
        if conv_id is not None:
            if conv_id not in grouped:
                grouped[conv_id] = []
            grouped[conv_id].append(resp)
    return grouped


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
            "conversation_id": r.get("conversation_id"),
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
            "conversation_id": m["conversation_id"],
            "conversation_title": m["conversation_title"],
            "embedding": emb,  # numpy array 그대로 저장 (pickle)
        })

    print(f"임베딩 총 소요 시간: {elapsed:.2f}초, 문서당 평균: {elapsed/len(texts):.3f}초")
    return items


def compute_conversation_embeddings(
    response_items: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """
    응답별 embedding으로부터 conversation별 average pooled embedding 계산

    Args:
        response_items: response_id, conversation_id, embedding을 포함하는 아이템 리스트

    Returns:
        conversation_id를 키로 하는 정보 딕셔너리
    """
    # conversation_id별로 그룹화
    grouped = {}
    for item in response_items:
        conv_id = item.get('conversation_id')
        if conv_id is not None:
            if conv_id not in grouped:
                grouped[conv_id] = []
            grouped[conv_id].append(item)

    conv_embeddings = {}

    print(f"\n총 {len(grouped)}개 conversation에 대해 average pooling 수행 중...")

    for conv_id, conv_items in tqdm(grouped.items(), desc="Average pooling", unit="conv"):
        # conversation의 모든 응답 embedding 수집
        embeddings = []
        response_ids = []

        for item in conv_items:
            emb = item.get('embedding')
            if emb is not None:
                embeddings.append(emb)
                response_ids.append(item.get('response_id'))

        if not embeddings:
            print(f"Warning: conversation {conv_id}에 유효한 embedding이 없습니다.")
            continue

        # Average pooling
        avg_embedding = np.mean(embeddings, axis=0)

        # 메타데이터
        first_item = conv_items[0]
        conv_title = first_item.get('conversation_title', f'Conversation {conv_id}')

        conv_embeddings[conv_id] = {
            'conversation_id': conv_id,
            'conversation_title': conv_title,
            'embedding': avg_embedding,
            'num_responses': len(embeddings),
            'response_ids': response_ids
        }

    return conv_embeddings


def save_conversation_embeddings_pkl(conv_embeddings: Dict[int, Dict[str, Any]], output_path: str, model_name: str, normalized: bool):
    """
    Conversation embedding을 PKL 형식으로 저장

    Args:
        conv_embeddings: conversation embedding 딕셔너리
        output_path: 출력 PKL 파일 경로
        model_name: 사용된 모델명
        normalized: 정규화 여부
    """
    items = []
    for conv_id, conv_data in conv_embeddings.items():
        items.append({
            'conversation_id': conv_id,
            'conversation_title': conv_data['conversation_title'],
            'embedding': conv_data['embedding'],
            'num_responses': conv_data['num_responses'],
            'response_ids': conv_data['response_ids']
        })

    payload = {
        'model': model_name,
        'normalized': normalized,
        'count': len(items),
        'items': items,
        'metadata': {
            'num_conversations': len(items),
            'embedding_dim': items[0]['embedding'].shape[0] if items else 0,
            'method': 'average_pooling'
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)


def main():
    project_root = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Extract embeddings with thenlper/gte-base and save as .pkl")
    parser.add_argument("--input", type=str, default=str(project_root / 'test' / 'output' / 's1_ai_responses.json'), help="입력 responses JSON 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 PKL 경로 (미지정 시 모델명이 포함된 기본 경로 사용)")
    parser.add_argument("--output-conv", type=str, default=None, help="Conversation embedding 출력 PKL 경로 (미지정 시 자동 생성)")
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

    # Conversation embedding 출력 경로 설정
    if args.output_conv is None:
        output_conv_path = project_root / 'test' / 'output' / f's2_conversation_embeddings_{model_slug}.pkl'
    else:
        output_conv_path = Path(args.output_conv)
    output_conv_path.parent.mkdir(parents=True, exist_ok=True)

    print("AI 응답 데이터 로딩 중...")
    responses = load_ai_responses(str(input_path))
    print(f"총 {len(responses)}개의 응답을 로드했습니다.\n")

    # ============================================================
    # [1/2] 응답별 embedding 캐시 로딩 및 처리
    # ============================================================
    print("[1/2] 응답별 embedding 처리 중...")
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
    print(f"  응답별 embedding 캐시 상태: 기존 {len(existing_items)}개, 신규 {len(missing_responses)}개")

    items_all = list(existing_items.values())
    if missing_responses:
        print(f"  임베딩 추출 중... (model={args.model})")
        t0 = time.time()
        print(f"  모델 캐시 디렉터리: {args.cache_dir}")
        new_items = extract_embeddings(
            missing_responses,
            model_name=args.model,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
            content_max_chars=args.content_max_chars,
            cache_dir=args.cache_dir,
        )
        total_elapsed = time.time() - t0
        print(f"  임베딩 추출 완료. 총 소요 시간: {total_elapsed:.2f}초")
        items_all.extend(new_items)
    else:
        print("  추출할 신규 응답이 없습니다. 기존 캐시를 재사용합니다.")

    # 기존 캐시에 conversation_id가 없는 경우 추가
    # (기존 캐시와의 호환성을 위해)
    for item in items_all:
        if 'conversation_id' not in item:
            # responses에서 해당 response_id를 찾아서 conversation_id 추가
            for resp in responses:
                if resp.get('response_id') == item.get('response_id'):
                    item['conversation_id'] = resp.get('conversation_id')
                    break

    # response_id로 정렬하여 저장 일관성 유지
    items_all.sort(key=lambda x: (x['response_id'] is None, x['response_id']))

    payload = {
        "model": args.model,
        "normalized": not args.no_normalize,
        "count": len(items_all),
        "items": items_all,
    }

    print(f"\n  응답별 embedding을 {output_path}에 저장 중... (.pkl)")
    save_embeddings_pkl(payload, str(output_path))
    print("  저장 완료.")

    # ============================================================
    # [2/2] Conversation별 average pooled embedding 캐시 로딩 및 처리
    # ============================================================
    print(f"\n[2/2] Conversation별 average pooled embedding 처리 중...")

    # Conversation embedding 캐시 로딩
    existing_conv_payload = None
    if output_conv_path.exists():
        try:
            with open(output_conv_path, 'rb') as f:
                existing_conv_payload = pickle.load(f)
        except Exception:
            existing_conv_payload = None

    existing_conv_embeddings = {}
    if existing_conv_payload and existing_conv_payload.get('model') == args.model and existing_conv_payload.get('normalized') == (not args.no_normalize):
        for it in existing_conv_payload.get('items', []):
            conv_id = it.get('conversation_id')
            if conv_id is not None:
                existing_conv_embeddings[conv_id] = it

    # 현재 데이터의 모든 conversation ID 수집
    all_conv_ids = set()
    for item in items_all:
        conv_id = item.get('conversation_id')
        if conv_id is not None:
            all_conv_ids.add(conv_id)

    # 캐시에 없는 conversation만 계산
    missing_conv_ids = all_conv_ids - set(existing_conv_embeddings.keys())
    print(f"  Conversation embedding 캐시 상태: 기존 {len(existing_conv_embeddings)}개, 신규 {len(missing_conv_ids)}개")

    conv_embeddings_dict = {}
    if missing_conv_ids:
        print(f"  신규 conversation에 대해 average pooling 수행 중...")
        # 신규 conversation에 속한 items만 필터링
        items_for_new_convs = [item for item in items_all if item.get('conversation_id') in missing_conv_ids]
        new_conv_embeddings = compute_conversation_embeddings(items_for_new_convs)

        # 기존 캐시와 병합
        for conv_id, conv_data in existing_conv_embeddings.items():
            conv_embeddings_dict[conv_id] = {
                'conversation_id': conv_data['conversation_id'],
                'conversation_title': conv_data['conversation_title'],
                'embedding': np.array(conv_data['embedding']),
                'num_responses': conv_data['num_responses'],
                'response_ids': conv_data['response_ids']
            }
        conv_embeddings_dict.update(new_conv_embeddings)
    else:
        print("  추출할 신규 conversation이 없습니다. 기존 캐시를 재사용합니다.")
        for conv_id, conv_data in existing_conv_embeddings.items():
            conv_embeddings_dict[conv_id] = {
                'conversation_id': conv_data['conversation_id'],
                'conversation_title': conv_data['conversation_title'],
                'embedding': np.array(conv_data['embedding']),
                'num_responses': conv_data['num_responses'],
                'response_ids': conv_data['response_ids']
            }

    print(f"\n  Conversation embedding을 {output_conv_path}에 저장 중... (.pkl)")
    save_conversation_embeddings_pkl(conv_embeddings_dict, str(output_conv_path), args.model, not args.no_normalize)
    print("  저장 완료.")

    # 요약 정보 출력
    print(f"\n{'='*80}")
    print("처리 완료 요약")
    print(f"{'='*80}")
    print(f"총 응답 수: {len(items_all)}")
    print(f"총 conversation 수: {len(conv_embeddings_dict)}")
    if conv_embeddings_dict:
        avg_responses_per_conv = sum(d['num_responses'] for d in conv_embeddings_dict.values()) / len(conv_embeddings_dict)
        print(f"Conversation당 평균 응답 수: {avg_responses_per_conv:.2f}")
    print(f"\n출력 파일:")
    print(f"  - 응답별 embedding: {output_path}")
    print(f"  - Conversation embedding: {output_conv_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
