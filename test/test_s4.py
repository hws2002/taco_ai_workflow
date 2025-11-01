"""
S4 단계 테스트: Neo4j 그래프 DB 저장
"""

import sys
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyze.graph_storage import Neo4jStorage
from analyze.parser import Note
from analyze.relationship_builder import Edge

# .env 파일 로드
load_dotenv(project_root / ".env")


def load_notes_from_s1():
    """S1 결과에서 노트 로드"""
    output_dir = project_root / "test" / "output"
    notes_file = output_dir / "s1_notes.json"

    if not notes_file.exists():
        raise FileNotFoundError(
            f"S1 결과 파일이 없습니다: {notes_file}\n"
            "먼저 test_s1.py를 실행하세요."
        )

    with open(notes_file, 'r', encoding='utf-8') as f:
        notes_dict = json.load(f)

    notes = [
        Note(
            note_id=n['note_id'],
            title=n['title'],
            content=n['content'],
            create_time=n['create_time'],
            update_time=n.get('update_time'),
            message_count=n.get('message_count', 0)
        )
        for n in notes_dict
    ]

    return notes


def load_edges_from_s3():
    """S3 결과에서 엣지 로드"""
    output_dir = project_root / "test" / "output"
    edges_file = output_dir / "s3_edges.json"

    if not edges_file.exists():
        raise FileNotFoundError(
            f"S3 결과 파일이 없습니다: {edges_file}\n"
            "먼저 test_s3.py를 실행하세요."
        )

    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_dict = json.load(f)

    edges = [
        Edge(
            source_id=e['source_id'],
            target_id=e['target_id'],
            weight=e['weight'],
            label=e.get('label'),
            reason=e.get('reason')
        )
        for e in edges_dict
    ]

    return edges


def load_concepts_from_s2():
    """S2 결과에서 개념 로드"""
    output_dir = project_root / "test" / "output"
    concepts_file = output_dir / "s2_concepts.json"

    if not concepts_file.exists():
        raise FileNotFoundError(
            f"S2 결과 파일이 없습니다: {concepts_file}\n"
            "먼저 test_s2.py를 실행하세요."
        )

    with open(concepts_file, 'r', encoding='utf-8') as f:
        concepts_dict_str = json.load(f)

    # string key를 int로 변환
    concepts_dict = {int(k): v for k, v in concepts_dict_str.items()}

    return concepts_dict


def test_s4(
    clear_db: bool = True,
    save_concepts: bool = True,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
):
    start_time = time.time()
    times = {}

    print("=" * 80)
    print("S4 단계 테스트: Neo4j 그래프 DB 저장")
    print("=" * 80)

    # ============================================================
    # 1. 이전 단계 결과 로드
    # ============================================================
    print("\n[1] 이전 단계 결과 로드")
    print("-" * 80)

    load_start = time.time()
    notes = load_notes_from_s1()
    print(f"✓ {len(notes)}개의 노트 로드")

    edges = load_edges_from_s3()
    print(f"✓ {len(edges)}개의 엣지 로드")

    concepts_dict = load_concepts_from_s2()
    print(f"✓ {len(concepts_dict)}개의 노트 개념 로드")
    times['loading'] = time.time() - load_start
    print(f"  (소요 시간: {times['loading']:.2f}초)")

    # ============================================================
    # 2. Neo4j 연결
    # ============================================================
    print("\n" + "=" * 80)
    print("[2] Neo4j 연결")
    print("-" * 80)

    # 환경 변수 또는 기본값 사용
    uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

    print(f"\nNeo4j URI: {uri}")
    print(f"사용자: {user}")

    connect_start = time.time()
    try:
        storage = Neo4jStorage(uri=uri, user=user, password=password)
        times['connection'] = time.time() - connect_start
        print(f"✓ 연결 완료 ({times['connection']:.2f}초)")
    except Exception as e:
        print(f"\n❌ Neo4j 연결 실패: {e}")
        print("\n해결 방법:")
        print("1. Neo4j가 실행 중인지 확인")
        print("2. .env 파일에서 연결 정보 확인")
        print("3. Neo4j Desktop 또는 docker로 Neo4j 시작")
        raise

    # ============================================================
    # 3. 데이터베이스 초기화 (선택적)
    # ============================================================
    if clear_db:
        print("\n" + "=" * 80)
        print("[3] 데이터베이스 초기화")
        print("-" * 80)
        clear_start = time.time()
        storage.clear_database()
        times['clear_db'] = time.time() - clear_start
        print(f"✓ 초기화 완료 ({times['clear_db']:.2f}초)")

    # ============================================================
    # 4. 제약 조건 생성
    # ============================================================
    print("\n" + "=" * 80)
    print("[4] 제약 조건 생성")
    print("-" * 80)
    constraint_start = time.time()
    storage.create_constraints()
    times['constraints'] = time.time() - constraint_start
    print(f"✓ 제약 조건 생성 완료 ({times['constraints']:.2f}초)")

    # ============================================================
    # 5. 노트 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[5] 노트 저장")
    print("-" * 80)
    notes_start = time.time()
    storage.save_notes(notes)
    times['save_notes'] = time.time() - notes_start
    print(f"✓ 노트 저장 완료 ({times['save_notes']:.2f}초)")

    # ============================================================
    # 6. 엣지 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[6] 엣지 저장")
    print("-" * 80)
    edges_start = time.time()
    storage.save_edges(edges)
    times['save_edges'] = time.time() - edges_start
    print(f"✓ 엣지 저장 완료 ({times['save_edges']:.2f}초)")

    # ============================================================
    # 7. 개념 저장 (선택적)
    # ============================================================
    if save_concepts:
        print("\n" + "=" * 80)
        print("[7] 개념 저장")
        print("-" * 80)
        concepts_start = time.time()
        storage.save_concepts(concepts_dict)
        times['save_concepts'] = time.time() - concepts_start
        print(f"✓ 개념 저장 완료 ({times['save_concepts']:.2f}초)")

    # ============================================================
    # 8. 통계 확인
    # ============================================================
    print("\n" + "=" * 80)
    print("[8] 그래프 통계")
    print("-" * 80)

    stats_start = time.time()
    graph_stats = storage.get_graph_stats()
    times['get_stats'] = time.time() - stats_start

    print("\n그래프 통계:")
    for key, value in graph_stats.items():
        print(f"  {key}: {value}")

    # ============================================================
    # 9. 쿼리 테스트
    # ============================================================
    print("\n" + "=" * 80)
    print("[9] 쿼리 테스트")
    print("-" * 80)

    query_start = time.time()

    if notes:
        # 첫 번째 노트와 유사한 노트 조회
        first_note_id = notes[0].note_id
        print(f"\n노트 {first_note_id}와 유사한 노트 조회:")
        similar_notes = storage.query_similar_notes(first_note_id, min_weight=0.5, limit=5)

        if similar_notes:
            for i, note in enumerate(similar_notes, 1):
                print(f"  {i}. 노트 {note['id']}: {note['title']}")
                print(f"     유사도: {note['weight']:.4f}, 관계: {note['label']}")
        else:
            print("  유사한 노트가 없습니다.")

    if concepts_dict:
        # 첫 번째 개념으로 노트 조회
        first_concepts = list(concepts_dict.values())[0]
        if first_concepts:
            first_concept = first_concepts[0]
            print(f"\n개념 '{first_concept}'을 포함하는 노트 조회:")
            concept_notes = storage.query_notes_by_concept(first_concept, limit=5)

            if concept_notes:
                for i, note in enumerate(concept_notes, 1):
                    print(f"  {i}. 노트 {note['id']}: {note['title']}")
            else:
                print("  해당 개념을 포함하는 노트가 없습니다.")

    times['query'] = time.time() - query_start
    print(f"\n✓ 쿼리 테스트 완료 ({times['query']:.2f}초)")

    # ============================================================
    # 10. 시각화 데이터 추출
    # ============================================================
    print("\n" + "=" * 80)
    print("[10] 시각화 데이터 추출")
    print("-" * 80)

    viz_start = time.time()
    graph_data = storage.export_graph_for_visualization()
    print(f"\n✓ 노드: {len(graph_data['nodes'])}개")
    print(f"✓ 엣지: {len(graph_data['edges'])}개")

    # 파일로 저장
    output_dir = project_root / "test" / "output"
    viz_file = output_dir / "s4_visualization_data.json"
    with open(viz_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"✓ 시각화 데이터 저장: {viz_file}")
    times['visualization'] = time.time() - viz_start
    print(f"  (소요 시간: {times['visualization']:.2f}초)")

    # ============================================================
    # 11. 통계 파일 저장
    # ============================================================
    print("\n" + "=" * 80)
    print("[11] 통계 파일 저장")
    print("-" * 80)

    times['total'] = time.time() - start_time

    stats = {
        "total_notes": len(notes),
        "total_edges": len(edges),
        "total_concepts": len(concepts_dict),
        "clear_db": clear_db,
        "save_concepts": save_concepts,
        "graph_statistics": graph_stats,
        "elapsed_time": {
            "loading_seconds": round(times['loading'], 2),
            "connection_seconds": round(times['connection'], 2),
            "clear_db_seconds": round(times.get('clear_db', 0), 2),
            "constraints_seconds": round(times['constraints'], 2),
            "save_notes_seconds": round(times['save_notes'], 2),
            "save_edges_seconds": round(times['save_edges'], 2),
            "save_concepts_seconds": round(times.get('save_concepts', 0), 2),
            "get_stats_seconds": round(times['get_stats'], 2),
            "query_seconds": round(times['query'], 2),
            "visualization_seconds": round(times['visualization'], 2),
            "total_seconds": round(times['total'], 2)
        }
    }

    stats_file = output_dir / "s4_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ 통계 저장: {stats_file}")

    # ============================================================
    # 12. 연결 종료
    # ============================================================
    print("\n" + "=" * 80)
    print("[12] 연결 종료")
    print("-" * 80)
    storage.close()
    print("✓ 연결 종료 완료")

    print("\n" + "=" * 80)
    print("S4 단계 테스트 완료!")
    print("=" * 80)
    print(f"\n⏱️  소요 시간:")
    print(f"  - 데이터 로딩: {times['loading']:.2f}초")
    print(f"  - Neo4j 연결: {times['connection']:.2f}초")
    if 'clear_db' in times:
        print(f"  - DB 초기화: {times['clear_db']:.2f}초")
    print(f"  - 제약 조건 생성: {times['constraints']:.2f}초")
    print(f"  - 노트 저장: {times['save_notes']:.2f}초")
    print(f"  - 엣지 저장: {times['save_edges']:.2f}초")
    if 'save_concepts' in times:
        print(f"  - 개념 저장: {times['save_concepts']:.2f}초")
    print(f"  - 통계 조회: {times['get_stats']:.2f}초")
    print(f"  - 쿼리 테스트: {times['query']:.2f}초")
    print(f"  - 시각화 데이터: {times['visualization']:.2f}초")
    print(f"  - 전체: {times['total']:.2f}초")

    return storage, graph_data


if __name__ == "__main__":
    try:
        # 파라미터 조정 가능
        storage, graph_data = test_s4(
            clear_db=True,  # 데이터베이스 초기화 여부
            save_concepts=True,  # 개념 저장 여부 (시간이 오래 걸릴 수 있음)
        )

        print("\n✅ S4 테스트 성공!")
        print("\nNeo4j Browser에서 확인:")
        print("  http://localhost:7474")
        print("\n쿼리 예시:")
        print("  MATCH (n:Note) RETURN n LIMIT 10")
        print("  MATCH (n:Note)-[r:RELATED_TO]->(m:Note) RETURN n, r, m LIMIT 20")
        print("  MATCH (n:Note)-[:HAS_CONCEPT]->(c:Concept) RETURN n, c LIMIT 20")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
