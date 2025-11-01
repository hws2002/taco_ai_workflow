"""
전체 임베딩 워크플로우 테스트
AI 답변 추출 -> 임베딩 생성 -> BERTopic 클러스터링 -> 문서별 풀링 -> 유사도 계산
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import timedelta

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.loader import ConversationLoader
from analyze.parser import NoteParser
from analyze.semantic_analyzer import SemanticAnalyzer
from analyze.embedding_processor import EmbeddingProcessor
from analyze.cache_manager import CacheManager
from analyze.incremental_cache import IncrementalCache


def format_time(seconds: float) -> str:
    """시간을 읽기 쉬운 형식으로 변환"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}초"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}분 {secs:.1f}초"


def main(use_cache: bool = False, clear_cache: bool = False, sample_size: int = 50, use_incremental: bool = True):
    # 전체 시작 시간
    total_start = time.time()

    # 각 스텝별 시간 기록
    step_times = {}

    print("=" * 80)
    print("AI 답변 기반 문서 임베딩 워크플로우 테스트")
    if use_incremental:
        print("(증분 캐싱 모드)")
    print("=" * 80)

    # 캐시 관리자 초기화
    if use_incremental:
        cache_manager = IncrementalCache(cache_dir="cache")
    else:
        cache_manager = CacheManager(cache_dir="cache")

    # 캐시 초기화 요청 시
    if clear_cache:
        cache_manager.clear_cache()
        print("✓ 캐시 초기화 완료\n")

    # 캐시 정보 출력
    if use_cache and use_incremental:
        cache_manager.print_cache_info()

    # ============================================================
    # STEP 1: 데이터 로딩
    # ============================================================
    print("\n[STEP 1] 데이터 로딩")
    print("-" * 80)

    step_start = time.time()

    loader = ConversationLoader()

    # 샘플 데이터로 테스트 (전체 데이터는 너무 클 수 있음)
    print(f"샘플 {sample_size}개 대화 로딩 중...")
    conversations = loader.load_sample(n=sample_size)

    step_times['데이터 로딩'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['데이터 로딩'])}")

    # ============================================================
    # STEP 2: AI 답변만 추출
    # ============================================================
    print("\n[STEP 2] AI 답변만 추출")
    print("-" * 80)

    step_start = time.time()

    parser = NoteParser(min_content_length=20)
    ai_responses = parser.parse_ai_responses(conversations)

    print(f"✓ 총 {len(ai_responses)}개의 AI 답변 추출 완료")

    step_times['AI 답변 추출'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['AI 답변 추출'])}")

    # 샘플 출력
    if ai_responses:
        print(f"\n샘플 AI 답변:")
        sample_resp = ai_responses[0]
        print(f"  ID: {sample_resp.response_id}")
        print(f"  대화 제목: {sample_resp.conversation_title}")
        print(f"  내용 (앞 100자): {sample_resp.content[:100]}...")

    # ============================================================
    # STEP 3: 임베딩 생성
    # ============================================================
    print("\n[STEP 3] 임베딩 벡터 생성")
    print("-" * 80)

    step_start = time.time()

    if use_cache and use_incremental:
        # 증분 캐싱: ID 기반으로 필요한 것만 계산
        embeddings_cache = cache_manager.load_embeddings_cache()

        # 모든 응답 ID
        all_response_ids = [resp.response_id for resp in ai_responses]

        # 캐시에 없는 응답 ID 찾기
        missing_ids = cache_manager.get_missing_embeddings(all_response_ids, embeddings_cache)

        print(f"전체 응답: {len(all_response_ids)}개")
        print(f"캐시에 있음: {len(all_response_ids) - len(missing_ids)}개")
        print(f"새로 계산 필요: {len(missing_ids)}개")

        if missing_ids:
            # 없는 것만 계산
            print(f"\n{len(missing_ids)}개의 새로운 임베딩 계산 중...")

            analyzer = SemanticAnalyzer(
                model_name="BAAI/bge-m3",
                use_keybert=True
            )

            # missing_ids에 해당하는 응답만 필터링
            missing_responses = [resp for resp in ai_responses if resp.response_id in missing_ids]
            new_embeddings = analyzer.analyze_ai_responses(missing_responses)

            # 캐시 업데이트
            embeddings_cache = cache_manager.update_embeddings_cache(embeddings_cache, new_embeddings)
            cache_manager.save_embeddings_cache(embeddings_cache)

            print(f"✓ {len(new_embeddings)}개의 새로운 임베딩 추가")
        else:
            print("✓ 모든 임베딩이 캐시에 있음 (계산 건너뜀)")

        # 최종 결과: 현재 필요한 응답들의 임베딩만 추출
        response_embeddings = {rid: embeddings_cache[rid] for rid in all_response_ids}

    elif use_cache and not use_incremental:
        # 기존 방식: 전체 저장/로드
        response_embeddings = cache_manager.load_embeddings()

        if response_embeddings is None:
            analyzer = SemanticAnalyzer(
                model_name="BAAI/bge-m3",
                use_keybert=True
            )

            response_embeddings = analyzer.analyze_ai_responses(ai_responses)
            print(f"✓ 총 {len(response_embeddings)}개의 임베딩 생성 완료")

            cache_manager.save_embeddings(response_embeddings)
        else:
            print("✓ 캐시에서 임베딩 로드 완료 (계산 건너뜀)")

    else:
        # 캐시 사용 안 함
        analyzer = SemanticAnalyzer(
            model_name="BAAI/bge-m3",
            use_keybert=True
        )

        response_embeddings = analyzer.analyze_ai_responses(ai_responses)
        print(f"✓ 총 {len(response_embeddings)}개의 임베딩 생성 완료")

    step_times['임베딩 생성'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['임베딩 생성'])}")

    # 임베딩 차원 확인
    first_embedding = list(response_embeddings.values())[0]['embedding']
    print(f"  임베딩 차원: {first_embedding.shape[0]}")

    # ============================================================
    # STEP 4: BERTopic 클러스터링
    # ============================================================
    print("\n[STEP 4] BERTopic을 사용한 클러스터링")
    print("-" * 80)

    step_start = time.time()

    # 캐시 확인
    cached_result = None
    if use_cache:
        cached_result = cache_manager.load_clustered_responses()

    if cached_result is not None:
        clustered_responses, topic_keywords = cached_result
        print("✓ 캐시에서 클러스터링 결과 로드 완료 (계산 건너뜀)")
    else:
        processor = EmbeddingProcessor(
            min_topic_size=3,  # 작은 샘플이므로 최소 크기 줄임
            nr_topics=None,  # 자동 결정
            language="multilingual",
            verbose=True
        )

        # response_embeddings에 content 추가 (BERTopic이 문서 텍스트 필요)
        for response_id, emb_data in response_embeddings.items():
            # ai_responses에서 해당 response 찾기
            for resp in ai_responses:
                if resp.response_id == response_id:
                    emb_data['content'] = resp.content
                    break

        # BERTopic 클러스터링
        clustered_responses, topic_keywords = processor.cluster_with_bertopic(response_embeddings)

        print(f"\n✓ 클러스터링 완료")

        # 캐시에 저장
        if use_cache:
            cache_manager.save_clustered_responses(clustered_responses, topic_keywords)

    step_times['BERTopic 클러스터링'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['BERTopic 클러스터링'])}")
    print(f"  생성된 토픽 수: {len(set(cr.topic_id for cr in clustered_responses.values()))}")

    # ============================================================
    # STEP 5: 문서별 풀링
    # ============================================================
    print("\n[STEP 5] 대화별 토픽 풀링")
    print("-" * 80)

    step_start = time.time()

    # 캐시 확인
    document_embeddings = None
    if use_cache:
        document_embeddings = cache_manager.load_document_embeddings()

    if document_embeddings is None:
        # processor가 없으면 생성 (캐시에서 clustered_responses를 로드한 경우)
        if 'processor' not in locals():
            processor = EmbeddingProcessor(
                min_topic_size=3,
                nr_topics=None,
                language="multilingual",
                verbose=True
            )

        document_embeddings = processor.pool_by_conversation(
            clustered_responses,
            response_embeddings
        )

        print(f"✓ {len(document_embeddings)}개의 대화에 대한 주제 벡터 생성 완료")

        # 캐시에 저장
        if use_cache:
            cache_manager.save_document_embeddings(document_embeddings)
    else:
        print("✓ 캐시에서 문서 임베딩 로드 완료 (계산 건너뜀)")
        print(f"  {len(document_embeddings)}개의 대화")

    step_times['문서별 풀링'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['문서별 풀링'])}")

    # 샘플 문서 정보 출력
    if document_embeddings:
        print(f"\n샘플 문서:")
        sample_conv_id = list(document_embeddings.keys())[0]
        sample_doc = document_embeddings[sample_conv_id]
        print(f"  대화 ID: {sample_doc.conversation_id}")
        print(f"  제목: {sample_doc.conversation_title}")
        print(f"  주제 수: {len(sample_doc.topic_embeddings)}")
        print(f"  주제 목록:")
        for topic_id, keywords in sample_doc.topic_keywords.items():
            keywords_str = ", ".join(keywords[:5])
            print(f"    토픽 {topic_id}: {keywords_str}")

    # ============================================================
    # STEP 6: 유사도 계산
    # ============================================================
    print("\n[STEP 6] 문서 간 유사도 계산")
    print("-" * 80)

    step_start = time.time()

    # 캐시 확인
    similarities = None
    if use_cache:
        similarities = cache_manager.load_similarities()

    if similarities is None:
        # processor가 없으면 생성
        if 'processor' not in locals():
            processor = EmbeddingProcessor(
                min_topic_size=3,
                nr_topics=None,
                language="multilingual",
                verbose=True
            )

        similarities = processor.compute_all_document_similarities(document_embeddings)

        print(f"\n✓ 유사도 계산 완료")

        # 캐시에 저장
        if use_cache:
            cache_manager.save_similarities(similarities)
    else:
        print("✓ 캐시에서 유사도 로드 완료 (계산 건너뜀)")

    step_times['유사도 계산'] = time.time() - step_start
    print(f"⏱️  소요 시간: {format_time(step_times['유사도 계산'])}")

    # 가장 유사한 문서 쌍 10개 출력
    print(f"\n가장 유사한 문서 쌍 TOP 10:")
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for i, ((conv_id_1, conv_id_2), sim) in enumerate(sorted_sims[:10], 1):
        doc1 = document_embeddings[conv_id_1]
        doc2 = document_embeddings[conv_id_2]
        print(f"  {i}. 유사도 {sim:.4f}")
        print(f"     [{conv_id_1}] {doc1.conversation_title}")
        print(f"     [{conv_id_2}] {doc2.conversation_title}")

    # ============================================================
    # 완료
    # ============================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("전체 워크플로우 테스트 완료!")
    print("=" * 80)

    print("\n📊 데이터 요약:")
    print(f"  - 대화 수: {len(conversations)}")
    print(f"  - AI 답변 수: {len(ai_responses)}")
    print(f"  - 생성된 토픽 수: {len(topic_keywords)}")
    print(f"  - 문서별 평균 주제 수: {sum(len(d.topic_embeddings) for d in document_embeddings.values()) / len(document_embeddings):.2f}")
    print(f"  - 유사도 쌍 수: {len(similarities)}")

    print("\n⏱️  실행 시간 상세:")
    print("-" * 80)
    for step_name, step_time in step_times.items():
        percentage = (step_time / total_time) * 100
        print(f"  {step_name:20s} : {format_time(step_time):>10s}  ({percentage:5.1f}%)")
    print("-" * 80)
    print(f"  {'전체 시간':20s} : {format_time(total_time):>10s}  (100.0%)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="문서 임베딩 워크플로우 테스트")

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="캐시 사용 (이전 계산 결과 재사용)"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="캐시 초기화 후 실행"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="샘플 대화 개수 (기본: 50)"
    )

    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="증분 캐싱 사용 안 함 (전체 저장/로드 방식)"
    )

    args = parser.parse_args()

    try:
        main(
            use_cache=args.use_cache,
            clear_cache=args.clear_cache,
            sample_size=args.sample_size,
            use_incremental=not args.no_incremental
        )
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
