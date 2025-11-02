"""
S2 단계 테스트: 임베딩 생성
(새로운 워크플로우: AI 답변 기반 문서 분류)
"""

import sys
from pathlib import Path
import time
import json
import pickle

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
from analyze.semantic_analyzer import SemanticAnalyzer
from analyze.incremental_cache import IncrementalCache


def test_s2_new(sample_size=50, use_cache=True):
    """
    S2 테스트: 임베딩 생성

    새 워크플로우:
    1. AI 답변 로드
    2. 임베딩 생성 (증분 캐싱 사용)
    3. 결과 저장
    """
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S2 단계 테스트: 임베딩 생성")
    print("(새로운 워크플로우: AI 답변 기반)")
    print("=" * 80)

    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # ============================================================
    # 1. AI 답변 로드
    # ============================================================
    print("\n[1] AI 답변 로드")
    print("-" * 80)

    load_start = time.time()

    # S1 결과 파일이 있으면 로드, 없으면 새로 생성
    s1_output = output_dir / "s1_ai_responses.json"

    if s1_output.exists():
        print(f"S1 결과 파일 로드 중: {s1_output}")
        with open(s1_output, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)

        # AIResponse 객체로 변환
        from analyze.parser import AIResponse
        ai_responses = []
        for data in responses_data:
            ai_responses.append(AIResponse(
                response_id=data['response_id'],
                conversation_id=data['conversation_id'],
                conversation_title=data['conversation_title'],
                message_index=data['message_index'],
                content=data['content'],
                timestamp=data.get('timestamp')  # 선택적으로 가져오기
            ))

        print(f"✓ {len(ai_responses)}개의 AI 답변 로드 완료")
    else:
        print("S1 결과 파일 없음. 새로 생성 중...")
        loader = ConversationLoader()
        conversations = loader.load_sample(n=sample_size)

        parser = NoteParser(min_content_length=20)
        ai_responses = parser.parse_ai_responses(conversations)

        print(f"✓ {len(ai_responses)}개의 AI 답변 추출 완료")

    times['loading'] = time.time() - load_start
    print(f"⏱️  소요 시간: {times['loading']:.2f}초")

    # ============================================================
    # 2. 임베딩 생성 (증분 캐싱)
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 임베딩 생성 (증분 캐싱 사용)")
    print("-" * 80)

    embedding_start = time.time()

    # 증분 캐시 초기화
    cache = IncrementalCache(cache_dir="cache")

    if use_cache:
        # 캐시 로드
        embeddings_cache = cache.load_embeddings_cache()

        # 모든 응답 ID
        all_response_ids = [resp.response_id for resp in ai_responses]

        # 캐시에 없는 응답 ID 찾기
        missing_ids = cache.get_missing_embeddings(all_response_ids, embeddings_cache)

        print(f"전체 응답: {len(all_response_ids)}개")
        print(f"캐시에 있음: {len(all_response_ids) - len(missing_ids)}개")
        print(f"새로 계산 필요: {len(missing_ids)}개")

        if missing_ids:
            # 없는 것만 계산
            print(f"\n{len(missing_ids)}개의 새로운 임베딩 계산 중...")

            analyzer = SemanticAnalyzer(
                model_name="thenlper/gte-base",  # 더 빠른 모델
                use_keybert=True
            )

            # missing_ids에 해당하는 응답만 필터링
            missing_responses = [resp for resp in ai_responses if resp.response_id in missing_ids]
            new_embeddings = analyzer.analyze_ai_responses(missing_responses)

            # 캐시 업데이트
            embeddings_cache = cache.update_embeddings_cache(embeddings_cache, new_embeddings)
            cache.save_embeddings_cache(embeddings_cache)

            print(f"✓ {len(new_embeddings)}개의 새로운 임베딩 추가")
        else:
            print("✓ 모든 임베딩이 캐시에 있음 (계산 건너뜀)")

        # 최종 결과: 현재 필요한 응답들의 임베딩만 추출
        response_embeddings = {rid: embeddings_cache[rid] for rid in all_response_ids}

    else:
        # 캐시 사용 안 함
        print("캐시 사용 안 함 - 전체 계산")
        analyzer = SemanticAnalyzer(
            model_name="thenlper/gte-base",
            use_keybert=True
        )

        response_embeddings = analyzer.analyze_ai_responses(ai_responses)
        print(f"✓ 총 {len(response_embeddings)}개의 임베딩 생성 완료")

    times['embedding'] = time.time() - embedding_start
    print(f"⏱️  소요 시간: {times['embedding']:.2f}초")

    # 임베딩 차원 확인
    first_embedding = list(response_embeddings.values())[0]['embedding']
    print(f"  임베딩 차원: {first_embedding.shape[0]}")

    # ============================================================
    # 3. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 결과 저장")
    print("-" * 80)

    save_start = time.time()

    # 임베딩 저장 (pickle)
    embeddings_file = output_dir / "s2_embeddings.pkl"
    with open(embeddings_file, 'wb') as f:
        pickle.dump(response_embeddings, f)

    print(f"✓ 임베딩 저장: {embeddings_file}")
    print(f"  - 총 {len(response_embeddings)}개")
    print(f"  - 파일 크기: {embeddings_file.stat().st_size / 1024 / 1024:.2f} MB")

    # AI 답변별 개념 저장 (JSON)
    concepts_dict = {
        rid: data['concepts']
        for rid, data in response_embeddings.items()
    }

    concepts_file = output_dir / "s2_concepts.json"
    with open(concepts_file, 'w', encoding='utf-8') as f:
        json.dump(concepts_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ AI 답변별 개념 저장: {concepts_file}")
    print(f"  - 총 {len(concepts_dict)}개")

    # ============================================================
    # 3-1. Conversation별 키워드 집계 (LLM 대분류용)
    # ============================================================
    print("\n" + "-" * 80)
    print("[3-1] Conversation별 키워드 집계")
    print("-" * 80)

    from collections import defaultdict

    # conversation_id별로 키워드 수집
    conversation_keywords = defaultdict(set)
    conversation_titles = {}

    for rid, data in response_embeddings.items():
        conv_id = data['conversation_id']
        conv_title = data['conversation_title']
        keywords = data['concepts']

        # 키워드 추가
        conversation_keywords[conv_id].update(keywords)
        # 제목 저장
        conversation_titles[conv_id] = conv_title

    # set을 list로 변환하고 conversation 정보 포함
    conversation_keywords_list = {
        conv_id: {
            'title': conversation_titles[conv_id],
            'keywords': sorted(list(keywords))  # 정렬해서 저장
        }
        for conv_id, keywords in conversation_keywords.items()
    }

    # Conversation별 키워드 저장
    conv_keywords_file = output_dir / "s2_conversation_keywords.json"
    with open(conv_keywords_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_keywords_list, f, ensure_ascii=False, indent=2)

    print(f"✓ Conversation별 키워드 저장: {conv_keywords_file}")
    print(f"  - 총 {len(conversation_keywords_list)}개 채팅 기록")

    # 키워드 통계
    keyword_counts = [len(data['keywords']) for data in conversation_keywords_list.values()]
    if keyword_counts:
        print(f"  - 평균 키워드 수: {sum(keyword_counts) / len(keyword_counts):.1f}개")
        print(f"  - 최소 키워드 수: {min(keyword_counts)}개")
        print(f"  - 최대 키워드 수: {max(keyword_counts)}개")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    # 임베딩 통계
    embedding_dims = [data['embedding'].shape[0] for data in response_embeddings.values()]
    concept_counts = [len(concepts) for concepts in concepts_dict.values()]

    stats = {
        "total_embeddings": len(response_embeddings),
        "embedding_dimension": embedding_dims[0] if embedding_dims else 0,
        "avg_concepts_per_response": sum(concept_counts) / len(concept_counts) if concept_counts else 0,
        "total_concepts": sum(concept_counts),
        "total_conversations": len(conversation_keywords_list),
        "avg_keywords_per_conversation": sum(keyword_counts) / len(keyword_counts) if keyword_counts else 0,
        "cache_used": use_cache,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "embedding_seconds": round(times['embedding'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s2_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 통계 저장: {stats_file}")

    print("\n" + "=" * 80)
    print("S2 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - AI 답변 로드: {times['loading']:.2f}초")
    print(f"  - 임베딩 생성: {times['embedding']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return response_embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S2 테스트: 임베딩 생성")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")
    args = parser.parse_args()

    try:
        response_embeddings = test_s2_new(
            sample_size=args.sample_size,
            use_cache=not args.no_cache
        )
        print(f"\n✅ S2 테스트 성공!")
        print(f"생성된 임베딩: {len(response_embeddings)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
