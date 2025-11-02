"""
S5 단계 테스트: LLM 기반 대분류 클러스터링
(키워드 기반으로 LLM이 의미론적 클러스터를 생성)
"""

import sys
from pathlib import Path
import time
import json
import pickle

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.llm_clustering import LLMClusterer, extract_keywords_from_embeddings


def test_s5_llm_clustering(
    sample_size=50,
    use_cache=True,
    provider="openai",
    model="gpt-4",
    n_clusters=None
):
    """
    S5 테스트: LLM 기반 대분류 클러스터링

    워크플로우:
    1. S3 결과(document_embeddings, response_embeddings) 로드
    2. 키워드 추출
    3. LLM에게 클러스터링 요청
    4. 결과 저장
    """
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S5 단계 테스트: LLM 기반 대분류 클러스터링")
    print("=" * 80)

    output_dir = project_root / "test" / "output"
    output_dir.mkdir(exist_ok=True)

    # ============================================================
    # 1. S3 결과 로드
    # ============================================================
    print("\n[1] S3 결과 로드")
    print("-" * 80)

    load_start = time.time()

    # Document embeddings 로드
    doc_embeddings_file = output_dir / "s3_document_embeddings.pkl"
    if not doc_embeddings_file.exists():
        print(f"❌ S3 결과 파일이 없습니다: {doc_embeddings_file}")
        print("먼저 test_s3_new.py를 실행하세요.")
        return None

    with open(doc_embeddings_file, 'rb') as f:
        document_embeddings = pickle.load(f)

    print(f"✓ {len(document_embeddings)}개의 문서 임베딩 로드")

    # Response embeddings 로드
    embeddings_file = output_dir / "s2_embeddings.pkl"
    if not embeddings_file.exists():
        print(f"❌ S2 결과 파일이 없습니다: {embeddings_file}")
        print("먼저 test_s2_new.py를 실행하세요.")
        return None

    with open(embeddings_file, 'rb') as f:
        response_embeddings = pickle.load(f)

    print(f"✓ {len(response_embeddings)}개의 응답 임베딩 로드")

    times['loading'] = time.time() - load_start
    print(f"⏱️  소요 시간: {times['loading']:.2f}초")

    # ============================================================
    # 2. 키워드 추출
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 키워드 추출")
    print("-" * 80)

    extract_start = time.time()

    # document_embeddings에서 키워드 추출
    chats_data = extract_keywords_from_embeddings(
        document_embeddings=document_embeddings,
        response_embeddings=response_embeddings,
        top_k=10  # 각 채팅당 상위 10개 키워드
    )

    print(f"✓ {len(chats_data['chats'])}개 채팅의 키워드 추출 완료")

    # 샘플 출력
    if chats_data['chats']:
        print(f"\n샘플 (처음 3개 채팅):")
        for i, chat in enumerate(chats_data['chats'][:3], 1):
            print(f"\n{i}. {chat['chat_id']} - {chat.get('title', 'N/A')}")
            keywords_str = ", ".join([f"{kw['word']}({kw['score']:.2f})"
                                     for kw in chat['keywords'][:5]])
            print(f"   키워드: {keywords_str}")

    times['extraction'] = time.time() - extract_start
    print(f"\n⏱️  소요 시간: {times['extraction']:.2f}초")

    # 키워드 데이터 저장 (중간 결과)
    keywords_file = output_dir / "s5_keywords.json"
    with open(keywords_file, 'w', encoding='utf-8') as f:
        json.dump(chats_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 키워드 데이터 저장: {keywords_file}")

    # ============================================================
    # 3. LLM 클러스터링
    # ============================================================
    print("\n" + "=" * 80)
    print("[3] LLM 클러스터링")
    print("-" * 80)

    clustering_start = time.time()

    try:
        # LLM 클러스터러 초기화
        clusterer = LLMClusterer(
            provider=provider,
            model=model
        )

        # 클러스터링 실행
        result = clusterer.cluster_chats(
            chats_data=chats_data,
            n_clusters=n_clusters,
            temperature=0.3  # 낮은 온도로 일관성 확보
        )

        times['clustering'] = time.time() - clustering_start
        print(f"\n⏱️  소요 시간: {times['clustering']:.2f}초")

    except Exception as e:
        print(f"❌ LLM 클러스터링 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ============================================================
    # 4. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 결과 저장")
    print("-" * 80)

    save_start = time.time()

    # 클러스터링 결과 저장
    result_file = output_dir / "s5_llm_clustering_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ 클러스터링 결과 저장: {result_file}")
    print(f"  - 파일 크기: {result_file.stat().st_size / 1024:.2f} KB")

    # 통계 정보 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    # 클러스터 분포 계산
    from collections import Counter
    cluster_counts = Counter(assignment['cluster_id']
                            for assignment in result['chat_assignments'])

    stats = {
        "total_chats": len(chats_data['chats']),
        "total_clusters": len(result.get('cluster_definitions', {})),
        "cluster_distribution": dict(cluster_counts),
        "llm_provider": provider,
        "llm_model": model,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "extraction_seconds": round(times['extraction'], 2),
            "clustering_seconds": round(times['clustering'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s5_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 통계 저장: {stats_file}")

    # ============================================================
    # 결과 요약
    # ============================================================
    print("\n" + "=" * 80)
    print("S5 단계 테스트 완료!")
    print("=" * 80)

    print(f"\n⏱️  소요 시간:")
    print(f"  - 데이터 로드: {times['loading']:.2f}초")
    print(f"  - 키워드 추출: {times['extraction']:.2f}초")
    print(f"  - LLM 클러스터링: {times['clustering']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    print(f"\n📊 결과 요약:")
    print(f"  - 채팅 수: {len(chats_data['chats'])}개")
    print(f"  - 클러스터 수: {len(result.get('cluster_definitions', {}))}개")
    print(f"  - LLM: {provider}/{model}")

    print(f"\n📁 출력 파일:")
    print(f"  - {keywords_file}")
    print(f"  - {result_file}")
    print(f"  - {stats_file}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S5 테스트: LLM 기반 대분류 클러스터링")
    parser.add_argument("--sample-size", type=int, default=50, help="샘플 대화 개수")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic"], help="LLM 제공자")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="LLM 모델 (예: gpt-4, gpt-3.5-turbo, claude-3-opus-20240229)")
    parser.add_argument("--n-clusters", type=int, default=None,
                       help="클러스터 개수 (미지정시 LLM이 자동 결정)")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함")

    args = parser.parse_args()

    try:
        result = test_s5_llm_clustering(
            sample_size=args.sample_size,
            use_cache=not args.no_cache,
            provider=args.provider,
            model=args.model,
            n_clusters=args.n_clusters
        )

        if result:
            print(f"\n✅ S5 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
