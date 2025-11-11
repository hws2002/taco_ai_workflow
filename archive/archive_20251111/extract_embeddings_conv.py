"""
Conversation 단위로 average pooled embedding을 생성하는 스크립트
각 conversation의 모든 응답 embedding을 평균내어 하나의 대표 벡터 생성
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    기존 응답별 embedding PKL 파일 로드

    Args:
        embeddings_path: embedding PKL 파일 경로

    Returns:
        response_id를 키로 하는 embedding 딕셔너리
    """
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)

    emb_dict = {}
    for item in data.get('items', []):
        response_id = item.get('response_id')
        embedding = item.get('embedding')
        if response_id and embedding is not None:
            emb_dict[response_id] = np.array(embedding)

    return emb_dict


def load_responses(responses_path: str) -> List[Dict[str, Any]]:
    """
    응답 JSON 파일 로드

    Args:
        responses_path: responses JSON 파일 경로

    Returns:
        응답 리스트
    """
    with open(responses_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def compute_conversation_embeddings(
    responses: List[Dict[str, Any]],
    response_embeddings: Dict[str, np.ndarray]
) -> Dict[int, Dict[str, Any]]:
    """
    Conversation별 average pooled embedding 계산

    Args:
        responses: 응답 리스트
        response_embeddings: response_id를 키로 하는 embedding 딕셔너리

    Returns:
        conversation_id를 키로 하는 정보 딕셔너리
        {
            conv_id: {
                'conversation_id': int,
                'conversation_title': str,
                'embedding': np.ndarray (average pooled),
                'num_responses': int,
                'response_ids': List[str]
            }
        }
    """
    grouped = group_by_conversation(responses)
    conv_embeddings = {}

    print(f"총 {len(grouped)}개 conversation에 대해 embedding 계산 중...")

    for conv_id, conv_responses in tqdm(grouped.items(), desc="Average pooling", unit="conv"):
        # conversation의 모든 응답 embedding 수집
        embeddings = []
        valid_response_ids = []

        for resp in conv_responses:
            resp_id = resp.get('response_id')
            if resp_id in response_embeddings:
                embeddings.append(response_embeddings[resp_id])
                valid_response_ids.append(resp_id)

        if not embeddings:
            print(f"Warning: conversation {conv_id}에 유효한 embedding이 없습니다.")
            continue

        # Average pooling
        avg_embedding = np.mean(embeddings, axis=0)

        # 메타데이터
        first_resp = conv_responses[0]
        conv_title = first_resp.get('conversation_title', f'Conversation {conv_id}')

        conv_embeddings[conv_id] = {
            'conversation_id': conv_id,
            'conversation_title': conv_title,
            'embedding': avg_embedding,
            'num_responses': len(embeddings),
            'response_ids': valid_response_ids
        }

    return conv_embeddings


def save_embeddings_pkl(conv_embeddings: Dict[int, Dict[str, Any]], output_path: str):
    """
    Conversation embedding을 PKL 형식으로 저장

    Args:
        conv_embeddings: conversation embedding 딕셔너리
        output_path: 출력 PKL 파일 경로
    """
    items = []
    for conv_id, conv_data in conv_embeddings.items():
        items.append({
            'conversation_id': conv_id,
            'conversation_title': conv_data['conversation_title'],
            'embedding': conv_data['embedding'].tolist(),
            'num_responses': conv_data['num_responses'],
            'response_ids': conv_data['response_ids']
        })

    payload = {
        'items': items,
        'metadata': {
            'num_conversations': len(items),
            'embedding_dim': items[0]['embedding'].__len__() if items else 0,
            'method': 'average_pooling'
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)


def save_embeddings_json(conv_embeddings: Dict[int, Dict[str, Any]], output_path: str):
    """
    Conversation embedding을 JSON 형식으로 저장 (embedding 포함)

    Args:
        conv_embeddings: conversation embedding 딕셔너리
        output_path: 출력 JSON 파일 경로
    """
    items = []
    for conv_id, conv_data in conv_embeddings.items():
        items.append({
            'conversation_id': conv_id,
            'conversation_title': conv_data['conversation_title'],
            'embedding': conv_data['embedding'].tolist(),
            'num_responses': conv_data['num_responses'],
            'response_ids': conv_data['response_ids']
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def print_summary(conv_embeddings: Dict[int, Dict[str, Any]]):
    """결과 요약 출력"""
    print(f"\n{'='*80}")
    print("Conversation Embedding 생성 결과")
    print(f"{'='*80}\n")

    total_convs = len(conv_embeddings)
    total_responses = sum(data['num_responses'] for data in conv_embeddings.values())

    if conv_embeddings:
        first_emb = next(iter(conv_embeddings.values()))['embedding']
        emb_dim = len(first_emb)
    else:
        emb_dim = 0

    print(f"총 conversation 수: {total_convs}")
    print(f"총 응답 수: {total_responses}")
    print(f"Embedding 차원: {emb_dim}")
    print(f"Conversation당 평균 응답 수: {total_responses / max(1, total_convs):.2f}")

    # 샘플 출력
    print(f"\n샘플 (처음 5개):")
    for i, (conv_id, data) in enumerate(list(conv_embeddings.items())[:5]):
        print(f"  [{i+1}] Conv ID: {conv_id}")
        print(f"      Title: {data['conversation_title']}")
        print(f"      응답 수: {data['num_responses']}")
        print(f"      Embedding shape: {data['embedding'].shape}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Conversation 단위 average pooled embedding 생성")
    parser.add_argument("--responses", type=str, required=True, help="응답 JSON 파일 경로 (s2_ai_responses_with_keywords.json)")
    parser.add_argument("--embeddings", type=str, required=True, help="응답별 embedding PKL 파일 경로")
    parser.add_argument("--output-pkl", type=str, default=None, help="출력 PKL 파일 경로")
    parser.add_argument("--output-json", type=str, default=None, help="출력 JSON 파일 경로 (선택)")
    args = parser.parse_args()

    # 기본 출력 경로 설정
    if args.output_pkl is None:
        base_dir = Path(args.embeddings).parent
        args.output_pkl = str(base_dir / 'conversation_embeddings.pkl')

    print("응답 데이터 로딩 중...")
    responses = load_responses(args.responses)
    print(f"총 {len(responses)}개 응답 로드")

    print("\n응답별 embedding 로딩 중...")
    response_embeddings = load_embeddings(args.embeddings)
    print(f"총 {len(response_embeddings)}개 embedding 로드")

    print("\nConversation별 average pooling 수행 중...")
    conv_embeddings = compute_conversation_embeddings(responses, response_embeddings)

    # PKL 저장
    print(f"\nPKL 형식으로 저장 중: {args.output_pkl}")
    save_embeddings_pkl(conv_embeddings, args.output_pkl)
    print("PKL 저장 완료")

    # JSON 저장 (선택)
    if args.output_json:
        print(f"\nJSON 형식으로 저장 중: {args.output_json}")
        save_embeddings_json(conv_embeddings, args.output_json)
        print("JSON 저장 완료")

    # 요약 출력
    print_summary(conv_embeddings)

    print(f"\n{'='*80}")
    print("처리 완료!")
    print(f"  - PKL: {args.output_pkl}")
    if args.output_json:
        print(f"  - JSON: {args.output_json}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()