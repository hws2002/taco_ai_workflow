"""
S4 단계 테스트: 유사도 계산 및 결과 분석
(새로운 워크플로우: AI 답변 기반 문서 분류)
"""

import sys
from pathlib import Path
import time
import json
import pickle
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.embedding_processor import EmbeddingProcessor
from analyze.incremental_cache import IncrementalCache


def test_s4_new(sample_size=50, use_cache=True, similarity_threshold=0.7):
    """
    S4 테스트: 유사도 계산 및 결과 분석

    새 워크플로우:
    1. 문서 임베딩 로드
    2. 유사도 계산 (증분 캐싱)
    3. 유사한 문서 쌍 분석
    4. 결과 저장
    """
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S4 단계 테스트: 유사도 계산 및 결과 분석")
    print("(새로운 워크플로우: AI 답변 기반)")
    print("=" * 80)

    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # ============================================================
    # 1. 문서 임베딩 로드
    # ============================================================
    print("\n[1] 문서 임베딩 로드")
    print("-" * 80)

    load_start = time.time()

    # S3 결과 파일 로드
    doc_embeddings_file = output_dir / "s3_document_embeddings.pkl"

    if not doc_embeddings_file.exists():
        print(f"❌ S3 결과 파일이 없습니다: {doc_embeddings_file}")
        print("먼저 test_s3_new.py를 실행하세요.")
        return None

    with open(doc_embeddings_file, 'rb') as f:
        document_embeddings = pickle.load(f)

    print(f"✓ {len(document_embeddings)}개의 문서 임베딩 로드 완료")

    times['loading'] = time.time() - load_start
    print(f"⏱️  소요 시간: {times['loading']:.2f}초")

    # ============================================================
    # 2. 유사도 계산 (증분 캐싱)
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 유사도 계산 (증분 캐싱 사용)")
    print("-" * 80)

    similarity_start = time.time()

    processor = EmbeddingProcessor(
        min_topic_size=3,
        nr_topics=None,
        language="multilingual",
        verbose=False
    )

    if use_cache:
        # 증분 캐시 사용
        cache = IncrementalCache(cache_dir="cache")
        similarities_cache = cache.load_similarities_cache()

        # 모든 대화 ID
        all_conv_ids = list(document_embeddings.keys())

        # 캐시에 없는 쌍 찾기
        missing_pairs = cache.get_missing_similarities(all_conv_ids, similarities_cache)

        total_pairs = len(all_conv_ids) * (len(all_conv_ids) - 1) // 2

        print(f"전체 쌍: {total_pairs}개")
        print(f"캐시에 있음: {total_pairs - len(missing_pairs)}개")
        print(f"새로 계산 필요: {len(missing_pairs)}개")

        if missing_pairs:
            print(f"\n{len(missing_pairs)}개의 새로운 유사도 계산 중...")

            # 없는 쌍만 계산
            new_similarities = {}
            for conv_id_1, conv_id_2 in missing_pairs:
                doc1 = document_embeddings[conv_id_1]
                doc2 = document_embeddings[conv_id_2]
                sim = processor.compute_document_similarity(doc1, doc2)
                new_similarities[(conv_id_1, conv_id_2)] = sim

            # 캐시 업데이트
            similarities_cache = cache.update_similarities_cache(similarities_cache, new_similarities)
            cache.save_similarities_cache(similarities_cache)

            print(f"✓ {len(new_similarities)}개의 새로운 유사도 추가")
        else:
            print("✓ 모든 유사도가 캐시에 있음 (계산 건너뜀)")

        # 최종 결과: 현재 필요한 쌍들의 유사도만 추출
        similarities = {}
        for i, conv_id_1 in enumerate(all_conv_ids):
            for conv_id_2 in all_conv_ids[i + 1:]:
                pair = tuple(sorted([conv_id_1, conv_id_2]))
                if pair in similarities_cache:
                    similarities[(conv_id_1, conv_id_2)] = similarities_cache[pair]

    else:
        # 캐시 사용 안 함
        print("캐시 사용 안 함 - 전체 계산")
        similarities = processor.compute_all_document_similarities(document_embeddings)

    times['similarity'] = time.time() - similarity_start
    print(f"⏱️  소요 시간: {times['similarity']:.2f}초")

    # 유사도 통계
    sim_values = list(similarities.values())
    print(f"\n유사도 통계:")
    print(f"  평균: {np.mean(sim_values):.4f}")
    print(f"  중앙값: {np.median(sim_values):.4f}")
    print(f"  표준편차: {np.std(sim_values):.4f}")
    print(f"  최대: {np.max(sim_values):.4f}")
    print(f"  최소: {np.min(sim_values):.4f}")

    # ============================================================
    # 3. 유사한 문서 쌍 분석
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 유사한 문서 쌍 분석")
    print("-" * 80)

    analysis_start = time.time()

    # 유사도 임계값 이상인 쌍 필터링
    high_similarity_pairs = [
        (pair, sim)
        for pair, sim in similarities.items()
        if sim >= similarity_threshold
    ]

    print(f"유사도 {similarity_threshold} 이상인 쌍: {len(high_similarity_pairs)}개")

    # 가장 유사한 쌍 TOP 10
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print(f"\n가장 유사한 문서 쌍 TOP 10:")
    for i, ((conv_id_1, conv_id_2), sim) in enumerate(sorted_sims[:10], 1):
        doc1 = document_embeddings[conv_id_1]
        doc2 = document_embeddings[conv_id_2]
        print(f"\n  {i}. 유사도 {sim:.4f}")
        print(f"     [{conv_id_1}] {doc1.conversation_title}")
        print(f"     [{conv_id_2}] {doc2.conversation_title}")

    times['analysis'] = time.time() - analysis_start

    # ============================================================
    # 4. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 결과 저장")
    print("-" * 80)

    save_start = time.time()

    # 유사도 저장 (pickle)
    similarities_file = output_dir / "s4_similarities.pkl"
    with open(similarities_file, 'wb') as f:
        pickle.dump(similarities, f)

    print(f"✓ 유사도 저장: {similarities_file}")
    print(f"  - 총 {len(similarities)}개 쌍")
    print(f"  - 파일 크기: {similarities_file.stat().st_size / 1024:.2f} KB")

    # 고유사도 쌍 저장 (JSON - 가독성)
    high_sim_data = []
    for (conv_id_1, conv_id_2), sim in sorted_sims[:50]:  # TOP 50
        doc1 = document_embeddings[conv_id_1]
        doc2 = document_embeddings[conv_id_2]
        high_sim_data.append({
            'conversation_1': {
                'id': conv_id_1,
                'title': doc1.conversation_title,
                'response_count': doc1.response_count,
                'topic_count': len(doc1.topic_embeddings)
            },
            'conversation_2': {
                'id': conv_id_2,
                'title': doc2.conversation_title,
                'response_count': doc2.response_count,
                'topic_count': len(doc2.topic_embeddings)
            },
            'similarity': float(sim)
        })

    high_sim_file = output_dir / "s4_high_similarities.json"
    with open(high_sim_file, 'w', encoding='utf-8') as f:
        json.dump(high_sim_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 고유사도 쌍 저장: {high_sim_file}")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    stats = {
        "total_documents": len(document_embeddings),
        "total_similarity_pairs": len(similarities),
        "high_similarity_pairs": len(high_similarity_pairs),
        "similarity_threshold": similarity_threshold,
        "similarity_statistics": {
            "mean": float(np.mean(sim_values)),
            "median": float(np.median(sim_values)),
            "std": float(np.std(sim_values)),
            "min": float(np.min(sim_values)),
            "max": float(np.max(sim_values))
        },
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "similarity_seconds": round(times['similarity'], 2),
            "analysis_seconds": round(times['analysis'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s4_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 통계 저장: {stats_file}")

    print("\n" + "=" * 80)
    print("S4 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 문서 임베딩 로드: {times['loading']:.2f}초")
    print(f"  - 유사도 계산: {times['similarity']:.2f}초")
    print(f"  - 결과 분석: {times['analysis']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    print(f"\n📊 결과 요약:")
    print(f"  - 문서 수: {len(document_embeddings)}개")
    print(f"  - 유사도 쌍 수: {len(similarities)}개")
    print(f"  - 고유사도 쌍 (≥{similarity_threshold}): {len(high_similarity_pairs)}개")
    print(f"  - 평균 유사도: {np.mean(sim_values):.4f}")

    return similarities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S4 테스트: 유사도 계산")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--threshold", type=float, default=0.7, help="유사도 임계값")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")
    args = parser.parse_args()

    try:
        similarities = test_s4_new(
            sample_size=args.sample_size,
            use_cache=not args.no_cache,
            similarity_threshold=args.threshold
        )

        if similarities:
            print(f"\n✅ S4 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
