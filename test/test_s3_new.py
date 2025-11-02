"""
S3 단계 테스트: BERTopic 클러스터링 및 문서별 풀링
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

from analyze.embedding_processor import EmbeddingProcessor
from analyze.incremental_cache import IncrementalCache


def test_s3_new(sample_size=50, use_cache=True, n_clusters=10):
    """
    S3 테스트: BERTopic 클러스터링 및 문서별 풀링

    새 워크플로우:
    1. 임베딩 로드
    2. BERTopic 클러스터링
    3. 대화별 토픽 풀링
    4. 결과 저장
    """
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S3 단계 테스트: BERTopic 클러스터링 및 문서별 풀링")
    print("(새로운 워크플로우: AI 답변 기반)")
    print("=" * 80)

    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # ============================================================
    # 1. 임베딩 로드
    # ============================================================
    print("\n[1] 임베딩 로드")
    print("-" * 80)

    load_start = time.time()

    # S2 결과 파일 로드
    embeddings_file = output_dir / "s2_embeddings.pkl"

    if not embeddings_file.exists():
        print(f"❌ S2 결과 파일이 없습니다: {embeddings_file}")
        print("먼저 test_s2_new.py를 실행하세요.")
        return None

    with open(embeddings_file, 'rb') as f:
        response_embeddings = pickle.load(f)

    print(f"✓ {len(response_embeddings)}개의 임베딩 로드 완료")

    # AI 답변 로드 (content 필요)
    s1_output = output_dir / "s1_ai_responses.json"
    with open(s1_output, 'r', encoding='utf-8') as f:
        responses_data = json.load(f)

    # response_embeddings에 content 추가
    for data in responses_data:
        rid = data['response_id']
        if rid in response_embeddings:
            response_embeddings[rid]['content'] = data['content']

    times['loading'] = time.time() - load_start
    print(f"⏱️  소요 시간: {times['loading']:.2f}초")

    # ============================================================
    # 2. BERTopic 클러스터링
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] BERTopic 클러스터링")
    print("-" * 80)

    clustering_start = time.time()

    processor = EmbeddingProcessor(
        min_topic_size=3,  # 작은 샘플이므로 최소 크기 줄임
        nr_topics=None,  # 자동 결정
        n_clusters=n_clusters,  # K-Means 클러스터 개수
        language="multilingual",
        verbose=True,
        use_kmeans=True  # K-Means 사용 (빠름)
    )

    # BERTopic 클러스터링
    clustered_responses, topic_keywords = processor.cluster_with_bertopic(response_embeddings)

    times['clustering'] = time.time() - clustering_start
    print(f"⏱️  소요 시간: {times['clustering']:.2f}초")
    print(f"  생성된 토픽 수: {len(set(cr.topic_id for cr in clustered_responses.values()))}")

    # ============================================================
    # 3. 대화별 토픽 풀링
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] 대화별 토픽 풀링")
    print("-" * 80)

    pooling_start = time.time()

    document_embeddings = processor.pool_by_conversation(
        clustered_responses,
        response_embeddings
    )

    times['pooling'] = time.time() - pooling_start
    print(f"⏱️  소요 시간: {times['pooling']:.2f}초")
    print(f"  {len(document_embeddings)}개의 대화")

    # 샘플 문서 정보 출력
    if document_embeddings:
        print(f"\n샘플 문서 (처음 3개):")
        for i, (conv_id, doc) in enumerate(list(document_embeddings.items())[:3], 1):
            print(f"\n{i}. 대화 {conv_id}:")
            print(f"   제목: {doc.conversation_title}")
            print(f"   주제 수: {len(doc.topic_embeddings)}개")
            print(f"   응답 수: {doc.response_count}개")
            print(f"   주제 목록:")
            for topic_id, keywords in doc.topic_keywords.items():
                keywords_str = ", ".join(keywords[:5])
                print(f"     - 토픽 {topic_id}: {keywords_str}")

    # ============================================================
    # 4. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 결과 저장")
    print("-" * 80)

    save_start = time.time()

    # 클러스터링 결과 저장 (pickle)
    clustering_file = output_dir / "s3_clustered_responses.pkl"
    with open(clustering_file, 'wb') as f:
        pickle.dump({
            'clustered_responses': clustered_responses,
            'topic_keywords': topic_keywords
        }, f)

    print(f"✓ 클러스터링 결과 저장: {clustering_file}")
    print(f"  - 파일 크기: {clustering_file.stat().st_size / 1024:.2f} KB")

    # 문서 임베딩 저장 (pickle)
    doc_embeddings_file = output_dir / "s3_document_embeddings.pkl"
    with open(doc_embeddings_file, 'wb') as f:
        pickle.dump(document_embeddings, f)

    print(f"✓ 문서 임베딩 저장: {doc_embeddings_file}")
    print(f"  - 파일 크기: {doc_embeddings_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 토픽 정보 저장 (JSON - 가독성 위해)
    topics_info = {}
    for topic_id, keywords in topic_keywords.items():
        topics_info[str(topic_id)] = {
            'keywords': keywords,
            'count': sum(1 for cr in clustered_responses.values() if cr.topic_id == topic_id)
        }

    topics_file = output_dir / "s3_topics.json"
    with open(topics_file, 'w', encoding='utf-8') as f:
        json.dump(topics_info, f, ensure_ascii=False, indent=2)

    print(f"✓ 토픽 정보 저장: {topics_file}")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    # 토픽별 분포
    from collections import Counter
    topic_counts = Counter(cr.topic_id for cr in clustered_responses.values())

    stats = {
        "total_responses": len(clustered_responses),
        "total_documents": len(document_embeddings),
        "total_topics": len(topic_keywords),
        "avg_topics_per_document": sum(len(d.topic_embeddings) for d in document_embeddings.values()) / len(document_embeddings) if document_embeddings else 0,
        "topic_distribution": {
            str(topic_id): count
            for topic_id, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "clustering_seconds": round(times['clustering'], 2),
            "pooling_seconds": round(times['pooling'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s3_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 통계 저장: {stats_file}")

    print("\n" + "=" * 80)
    print("S3 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 임베딩 로드: {times['loading']:.2f}초")
    print(f"  - BERTopic 클러스터링: {times['clustering']:.2f}초")
    print(f"  - 대화별 풀링: {times['pooling']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return document_embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S3 테스트: BERTopic 클러스터링 및 문서별 풀링")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--n-clusters", type=int, default=10, help="K-Means 클러스터 개수")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")
    args = parser.parse_args()

    try:
        document_embeddings = test_s3_new(
            sample_size=args.sample_size,
            use_cache=not args.no_cache,
            n_clusters=args.n_clusters
        )

        if document_embeddings:
            print(f"\n✅ S3 테스트 성공!")
            print(f"생성된 문서 임베딩: {len(document_embeddings)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
