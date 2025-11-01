"""
S3 단계 테스트: 관계 추론
"""

import sys
from pathlib import Path
import json
import pickle
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.relationship_builder import RelationshipBuilder

def load_data_from_s2():
    """S2 단계에서 생성한 데이터 로드"""
    output_dir = project_root / "test" / "output"

    # 1. 유사도 로드
    similarities_file = output_dir / "s2_similarities.json"
    if not similarities_file.exists():
        raise FileNotFoundError(
            f"S2 결과 파일이 없습니다: {similarities_file}\n"
            "먼저 test_s2.py를 실행하세요."
        )

    with open(similarities_file, 'r', encoding='utf-8') as f:
        similarities_serializable = json.load(f)

    # string key를 tuple로 변환
    similarities = {
        tuple(map(int, k.split(','))): v
        for k, v in similarities_serializable.items()
    }

    # 2. 개념 로드
    concepts_file = output_dir / "s2_concepts.json"
    with open(concepts_file, 'r', encoding='utf-8') as f:
        concepts_dict_str_keys = json.load(f)

    # string key를 int로 변환
    concepts_dict = {int(k): v for k, v in concepts_dict_str_keys.items()}

    return similarities, concepts_dict


def test_s3(similarity_threshold=0.5, use_conceptnet=True):
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S3 단계 테스트: 관계 추론")
    print("=" * 80)

    # ============================================================
    # 1. S2 결과 로드
    # ============================================================
    print("\n[1] S2 결과 로드")
    print("-" * 80)

    load_start = time.time()
    similarities, concepts_dict = load_data_from_s2()
    times['loading'] = time.time() - load_start

    print(f"✓ {len(similarities)}개의 유사도 쌍 로드")
    print(f"✓ {len(concepts_dict)}개의 노트 개념 로드")
    print(f"  (소요 시간: {times['loading']:.2f}초)\n")

    # 통계 출력
    avg_sim = sum(similarities.values()) / len(similarities) if similarities else 0
    print(f"유사도 통계:")
    print(f"  - 평균: {avg_sim:.4f}")
    print(f"  - 최대: {max(similarities.values()):.4f}")
    print(f"  - 최소: {min(similarities.values()):.4f}")

    # ============================================================
    # 2. 관계 추론
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] 관계 추론 시작")
    print("-" * 80)

    builder = RelationshipBuilder(
        similarity_threshold=similarity_threshold,
        use_conceptnet=use_conceptnet
    )

    # 간단 버전 사용 (ConceptNet 없이)
    print(f"\n임계값: {similarity_threshold}")
    print(f"ConceptNet 사용: {use_conceptnet}")

    inference_start = time.time()
    if use_conceptnet:
        # ConceptNet 포함 버전 (느림)
        edges = builder.build_relationships(similarities, concepts_dict)
    else:
        # 간단 버전 (빠름)
        edges = builder.build_simple_relationships(similarities)
    times['inference'] = time.time() - inference_start

    print(f"\n✓ {len(edges)}개의 엣지 생성 완료 ({times['inference']:.2f}초)\n")

    # ============================================================
    # 3. 엣지 분석
    # ============================================================
    print("=" * 80)
    print("[3] 엣지 분석")
    print("-" * 80)

    if edges:
        # 가중치 통계
        weights = [e.weight for e in edges]
        print(f"\n가중치 통계:")
        print(f"  - 평균: {sum(weights)/len(weights):.4f}")
        print(f"  - 최대: {max(weights):.4f}")
        print(f"  - 최소: {min(weights):.4f}")

        # 상위 엣지 출력
        top_edges = builder.get_top_edges(edges, top_k=10)
        print(f"\n상위 {len(top_edges)}개 엣지:")
        for i, edge in enumerate(top_edges, 1):
            print(f"  {i}. {edge}")
            if edge.reason:
                print(f"     이유: {edge.reason}")

        # 다양한 임계값으로 필터링
        print(f"\n임계값별 엣지 수:")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            filtered = builder.filter_edges_by_weight(edges, min_weight=threshold)
            print(f"  - {threshold:.1f} 이상: {len(filtered)}개")

    # ============================================================
    # 4. 결과 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 결과 저장")
    print("-" * 80)

    save_start = time.time()
    output_dir = project_root / "test" / "output"

    # 1) 엣지 저장 (JSON)
    edges_dict = builder.export_edges_to_dict(edges)
    edges_file = output_dir / "s3_edges.json"
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ 엣지 저장: {edges_file}")

    # 2) 통계 저장
    times['saving'] = time.time() - save_start
    times['total'] = time.time() - start_time

    stats = {
        "total_edges": len(edges),
        "similarity_threshold": similarity_threshold,
        "use_conceptnet": use_conceptnet,
        "avg_weight": sum(e.weight for e in edges) / len(edges) if edges else 0,
        "max_weight": max(e.weight for e in edges) if edges else 0,
        "min_weight": min(e.weight for e in edges) if edges else 0,
        "edges_with_label": sum(1 for e in edges if e.label) if edges else 0,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "inference_seconds": round(times['inference'], 2),
            "saving_seconds": round(times['saving'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s3_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ 통계 저장: {stats_file}")

    print("\n통계 요약:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 3) 그래프 데이터 저장 (Neo4j 입력용)
    # 노드는 S1에서, 엣지는 여기서
    graph_data = {
        "edges": edges_dict,
        "metadata": {
            "total_edges": len(edges),
            "threshold": similarity_threshold
        }
    }

    graph_file = output_dir / "s3_graph_data.json"
    with open(graph_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 그래프 데이터 저장: {graph_file}")

    print("\n" + "=" * 80)
    print("S3 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 데이터 로딩: {times['loading']:.2f}초")
    print(f"  - 관계 추론: {times['inference']:.2f}초")
    print(f"  - 결과 저장: {times['saving']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return edges


if __name__ == "__main__":
    try:
        # 파라미터 조정 가능
        edges = test_s3(
            similarity_threshold=0.5,  # 임계값 (0~1)
            use_conceptnet=True  # ConceptNet 사용 여부
        )

        print("\n✅ S3 테스트 성공!")
        print(f"생성된 엣지: {len(edges)}개")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
