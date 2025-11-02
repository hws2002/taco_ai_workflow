"""
전체 파이프라인 테스트 (S1 → S2 → S3 → S4)
(새로운 워크플로우: AI 답변 기반 문서 분류)
"""

import sys
from pathlib import Path
import time
import argparse

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_s1_new import test_s1_new
from test_s2_new import test_s2_new
from test_s3_new import test_s3_new
from test_s4_new import test_s4_new
from test_s5_llm_clustering import test_s5_llm_clustering


def test_full_pipeline(
    sample_size=50,
    use_cache=True,
    n_clusters=10,
    similarity_threshold=0.7,
    llm_provider="openai",
    llm_model="gpt-4",
    llm_n_clusters=None
):
    """
    전체 파이프라인 실행

    S1: AI 답변 추출
    S2: 임베딩 생성
    S3: BERTopic 클러스터링 및 문서별 풀링
    S4: 유사도 계산
    S5: LLM 기반 대분류 클러스터링

    Args:
        sample_size: 샘플 대화 개수
        use_cache: 증분 캐싱 사용 여부
        n_clusters: K-Means 클러스터 개수
        similarity_threshold: 유사도 임계값
        llm_provider: LLM 제공자 (openai/anthropic)
        llm_model: LLM 모델명
        llm_n_clusters: LLM 클러스터 개수 (None이면 자동)
    """
    total_start = time.time()

    print("=" * 80)
    print("전체 파이프라인 테스트")
    print("(새로운 워크플로우: AI 답변 기반 문서 분류)")
    print("=" * 80)
    print(f"\n설정:")
    print(f"  - 샘플 크기: {sample_size}개 대화")
    print(f"  - 증분 캐싱: {'사용' if use_cache else '사용 안 함'}")
    print(f"  - BERTopic 클러스터 개수: {n_clusters}개")
    print(f"  - 유사도 임계값: {similarity_threshold}")
    print(f"  - LLM 제공자: {llm_provider}")
    print(f"  - LLM 모델: {llm_model}")
    print(f"  - LLM 클러스터: {llm_n_clusters if llm_n_clusters else '자동 결정'}")
    print("=" * 80)

    # ============================================================
    # S1: AI 답변 추출
    # ============================================================
    print("\n\n" + "🔵" * 40)
    print("S1: AI 답변 추출")
    print("🔵" * 40)

    s1_start = time.time()
    ai_responses = test_s1_new(sample_size=sample_size, use_cache=use_cache)
    s1_time = time.time() - s1_start

    if not ai_responses:
        print("\n❌ S1 실패")
        return

    print(f"\n✅ S1 완료 ({s1_time:.1f}초)")

    # ============================================================
    # S2: 임베딩 생성
    # ============================================================
    print("\n\n" + "🟢" * 40)
    print("S2: 임베딩 생성")
    print("🟢" * 40)

    s2_start = time.time()
    response_embeddings = test_s2_new(sample_size=sample_size, use_cache=use_cache)
    s2_time = time.time() - s2_start

    if not response_embeddings:
        print("\n❌ S2 실패")
        return

    print(f"\n✅ S2 완료 ({s2_time:.1f}초)")

    # ============================================================
    # S3: BERTopic 클러스터링 및 문서별 풀링
    # ============================================================
    print("\n\n" + "🟡" * 40)
    print("S3: BERTopic 클러스터링 및 문서별 풀링")
    print("🟡" * 40)

    s3_start = time.time()
    document_embeddings = test_s3_new(
        sample_size=sample_size,
        use_cache=use_cache,
        n_clusters=n_clusters
    )
    s3_time = time.time() - s3_start

    if not document_embeddings:
        print("\n❌ S3 실패")
        return

    print(f"\n✅ S3 완료 ({s3_time:.1f}초)")

    # ============================================================
    # S4: 유사도 계산
    # ============================================================
    print("\n\n" + "🔴" * 40)
    print("S4: 유사도 계산")
    print("🔴" * 40)

    s4_start = time.time()
    similarities = test_s4_new(
        sample_size=sample_size,
        use_cache=use_cache,
        similarity_threshold=similarity_threshold
    )
    s4_time = time.time() - s4_start

    if not similarities:
        print("\n❌ S4 실패")
        return

    print(f"\n✅ S4 완료 ({s4_time:.1f}초)")

    # ============================================================
    # S5: LLM 기반 대분류 클러스터링
    # ============================================================
    print("\n\n" + "🟣" * 40)
    print("S5: LLM 기반 대분류 클러스터링")
    print("🟣" * 40)

    s5_start = time.time()
    llm_clusters = test_s5_llm_clustering(
        sample_size=sample_size,
        use_cache=use_cache,
        provider=llm_provider,
        model=llm_model,
        n_clusters=llm_n_clusters
    )
    s5_time = time.time() - s5_start

    if not llm_clusters:
        print("\n❌ S5 실패")
        return

    print(f"\n✅ S5 완료 ({s5_time:.1f}초)")

    # ============================================================
    # 전체 요약
    # ============================================================
    total_time = time.time() - total_start

    print("\n\n" + "=" * 80)
    print("전체 파이프라인 완료!")
    print("=" * 80)

    print(f"\n⏱️  단계별 소요 시간:")
    print(f"  S1 (AI 답변 추출)      : {s1_time:6.1f}초  ({s1_time/total_time*100:5.1f}%)")
    print(f"  S2 (임베딩 생성)       : {s2_time:6.1f}초  ({s2_time/total_time*100:5.1f}%)")
    print(f"  S3 (클러스터링/풀링)   : {s3_time:6.1f}초  ({s3_time/total_time*100:5.1f}%)")
    print(f"  S4 (유사도 계산)       : {s4_time:6.1f}초  ({s4_time/total_time*100:5.1f}%)")
    print(f"  S5 (LLM 대분류)        : {s5_time:6.1f}초  ({s5_time/total_time*100:5.1f}%)")
    print(f"  " + "-" * 60)
    print(f"  전체                   : {total_time:6.1f}초  (100.0%)")

    print(f"\n📊 최종 결과:")
    print(f"  - AI 답변: {len(ai_responses)}개")
    print(f"  - 임베딩: {len(response_embeddings)}개")
    print(f"  - 문서: {len(document_embeddings)}개")
    print(f"  - 유사도 쌍: {len(similarities)}개")
    print(f"  - LLM 클러스터: {len(llm_clusters.get('cluster_definitions', {}))}개")

    print(f"\n📁 출력 파일:")
    output_dir = project_root / "test" / "output"
    print(f"  - {output_dir / 's1_ai_responses.json'}")
    print(f"  - {output_dir / 's2_embeddings.pkl'}")
    print(f"  - {output_dir / 's3_document_embeddings.pkl'}")
    print(f"  - {output_dir / 's4_similarities.pkl'}")
    print(f"  - {output_dir / 's4_high_similarities.json'}")
    print(f"  - {output_dir / 's5_keywords.json'}")
    print(f"  - {output_dir / 's5_llm_clustering_result.json'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 파이프라인 테스트")

    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="샘플 대화 개수 (기본: 50)"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="증분 캐싱 사용 안 함"
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="K-Means 클러스터 개수 (기본: 10)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="유사도 임계값 (기본: 0.7)"
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM 제공자 (기본: openai)"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help="LLM 모델 (기본: gpt-4)"
    )

    parser.add_argument(
        "--llm-n-clusters",
        type=int,
        default=None,
        help="LLM 클러스터 개수 (미지정시 자동 결정)"
    )

    args = parser.parse_args()

    try:
        test_full_pipeline(
            sample_size=args.sample_size,
            use_cache=not args.no_cache,
            n_clusters=args.n_clusters,
            similarity_threshold=args.threshold,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_n_clusters=args.llm_n_clusters
        )

        print("\n✅ 전체 파이프라인 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
